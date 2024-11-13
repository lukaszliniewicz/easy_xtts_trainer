import argparse
import os
import json
import random
import shutil
import subprocess
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
import csv
import torch
import torchaudio
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
from trainer import Trainer, TrainerArgs
import gc
from datetime import datetime
import requests
from tqdm import tqdm
import re
import logging
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Literal
import json
from pathlib import Path
from scipy import signal
import numpy as np
from df.enhance import enhance, init_df, load_audio, save_audio
import soundfile as sf
import librosa
import traceback

conda_path = Path("../conda/Scripts/conda.exe")

class AudioProcessor:
    def __init__(self, target_sr: int = 22050):
        self.target_sr = target_sr
        self.deepfilter_model = None
        self.df_state = None
        
        # Compression profiles optimized for different voice types
        self.compression_profiles: Dict = {
            'male': {
                'threshold': -18,
                'ratio': 3.0,
                'attack': 0.020,
                'release': 0.150,
                'knee_width': 6
            },
            'female': {
                'threshold': -16,
                'ratio': 2.5,
                'attack': 0.015,
                'release': 0.120,
                'knee_width': 4
            },
            'neutral': {
                'threshold': -17,
                'ratio': 2.75,
                'attack': 0.018,
                'release': 0.135,
                'knee_width': 5
            }
        }
        
        # De-essing profiles
        self.dess_profiles: Dict = {
            'low': {
                'threshold': 0.15,
                'ratio': 2.5,
                'makeup_gain': 0.5,
                'reduction_factor': 0.3,
                'freq_range': (4500, 9000)
            },
            'high': {
                'threshold': 0.25,
                'ratio': 3.5,
                'makeup_gain': 0.4,
                'reduction_factor': 0.5,
                'freq_range': (5000, 10000)
            }
        }
        self.segment_window_ms = 200  # Window for finding cut points
        self.analysis_window_ms = 2  # Window for energy analysis

    def apply_fades(self, audio: np.ndarray, sr: int, fade_in_ms: Optional[int] = None, 
                    fade_out_ms: Optional[int] = None) -> np.ndarray:
        """
        Apply fade-in and/or fade-out effects to audio.
        
        Args:
            audio: Input audio array
            sr: Sample rate
            fade_in_ms: Duration of fade-in in milliseconds
            fade_out_ms: Duration of fade-out in milliseconds
        Returns:
            Audio with fades applied
        """
        try:
            # Make a copy to avoid modifying the original array
            audio = audio.copy()
            
            # Convert ms to samples
            fade_in_samples = int((fade_in_ms / 1000) * sr) if fade_in_ms else 0
            fade_out_samples = int((fade_out_ms / 1000) * sr) if fade_out_ms else 0
            
            # Apply fade-in
            if fade_in_samples > 0:
                if fade_in_samples >= len(audio):
                    fade_in_samples = len(audio) // 2  # Limit to half the audio length
                fade_in = np.linspace(0, 1, fade_in_samples)
                audio[:fade_in_samples] *= fade_in
            
            # Apply fade-out
            if fade_out_samples > 0:
                if fade_out_samples >= len(audio):
                    fade_out_samples = len(audio) // 2  # Limit to half the audio length
                fade_out = np.linspace(1, 0, fade_out_samples)
                audio[-fade_out_samples:] *= fade_out
            
            return audio
            
        except Exception as e:
            print(f"Error applying fades: {str(e)}")
            return audio


    def remove_trailing_silence(self, audio: np.ndarray, sr: int, silence_threshold=-50.0, chunk_size=10) -> np.ndarray:
        """
        Remove trailing silence using pydub's detect_leading_silence function.
        
        Args:
            audio: Input audio array
            sr: Sample rate
            silence_threshold: The threshold (in dB) below which is considered silence
            chunk_size: Size of chunks to analyze (in ms)
        Returns:
            Trimmed audio array with trailing silence removed
        """
        try:
            # Convert numpy array to AudioSegment
            audio_segment = AudioSegment(
                audio.tobytes(), 
                frame_rate=sr,
                sample_width=audio.dtype.itemsize,
                channels=1 if len(audio.shape) == 1 else audio.shape[1]
            )

            # Detect trailing silence by reversing the audio
            trailing_silence = detect_leading_silence(
                audio_segment.reverse(),
                silence_threshold=silence_threshold,
                chunk_size=chunk_size
            )

            # Convert milliseconds to samples
            samples_to_remove = int((trailing_silence / 1000) * sr)

            # Trim the audio
            if samples_to_remove > 0:
                return audio[:-samples_to_remove]
            return audio

        except Exception as e:
            print(f"Error in remove_trailing_silence: {str(e)}")
            return audio

    def find_lowest_energy_point(self, audio: AudioSegment, start_time: float, end_time: float, is_start_cut: bool = True) -> float:
        """
        Find the absolute quietest point using 2ms subwindows.
        is_start_cut: True if looking for start cut (100ms max), False for end cut (200ms max)
        """
        try:
            # Convert times to milliseconds
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            if end_ms <= start_ms:
                return start_time
                
            # Apply maximum window constraints
            window_duration_ms = end_ms - start_ms
            max_window_ms = 100 if is_start_cut else 200
            
            if window_duration_ms > max_window_ms:
                if is_start_cut:
                    start_ms = end_ms - max_window_ms
                else:
                    end_ms = start_ms + max_window_ms
                window_duration_ms = max_window_ms
                
            window = audio[start_ms:end_ms]
            
            # Fixed 2ms subwindows
            subwindow_size_ms = 2
            num_subwindows = window_duration_ms // subwindow_size_ms
            
            print(f"\nAnalyzing {window_duration_ms}ms window from {start_time:.3f}s to {end_time:.3f}s")
            print(f"Dividing into {num_subwindows} subwindows of {subwindow_size_ms}ms each")
            
            if subwindow_size_ms < 2 or num_subwindows < 1:
                print(f"Window too small ({window_duration_ms}ms), using midpoint")
                return (start_time + end_time) / 2
                
            samples = np.array(window.get_array_of_samples(), dtype=np.float32)
            sample_silence_threshold = 1e-4
            
            # Analyze each subwindow
            subwindow_data = []
            
            for i in range(num_subwindows):
                start_idx = i * len(samples) // num_subwindows
                end_idx = (i + 1) * len(samples) // num_subwindows
                subwindow = samples[start_idx:end_idx]
                
                # Check for true silence
                silence_ratio = np.sum(np.abs(subwindow) < sample_silence_threshold) / len(subwindow)
                is_silent = silence_ratio > 0.95
                
                # Calculate RMS energy
                rms = np.sqrt(np.mean(subwindow**2))
                db = 20 * np.log10(max(rms, 1e-5))
                
                subwindow_start_time = start_time + (i * window_duration_ms / num_subwindows) / 1000
                subwindow_center = subwindow_start_time + (subwindow_size_ms / 2000)
                
                print(f"Subwindow {i+1}: {db:.1f}dB, silence_ratio: {silence_ratio:.2f}, center at {subwindow_center:.3f}s")
                
                subwindow_data.append({
                    'index': i,
                    'db': db,
                    'center_time': subwindow_center,
                    'is_silent': is_silent,
                    'start_time': subwindow_start_time,
                    'end_time': subwindow_start_time + (subwindow_size_ms / 1000),
                    'samples': subwindow
                })
            
            if not subwindow_data:
                return (start_time + end_time) / 2

            # First check for true silence
            silent_windows = [w for w in subwindow_data if w['is_silent']]
            
            if silent_windows:
                # Find the silent window with lowest energy
                optimal_window = min(silent_windows, key=lambda x: x['db'])
                # Find the absolute lowest point within this window
                min_sample_idx = np.argmin(np.abs(optimal_window['samples']))
                relative_time = min_sample_idx / len(optimal_window['samples']) * subwindow_size_ms / 1000
                optimal_time = optimal_window['start_time'] + relative_time
                print(f"Found true silence at {optimal_time:.3f}s")
            else:
                # Find the window with lowest energy
                optimal_window = min(subwindow_data, key=lambda x: x['db'])
                # Find the absolute lowest point within this window
                min_sample_idx = np.argmin(np.abs(optimal_window['samples']))
                relative_time = min_sample_idx / len(optimal_window['samples']) * subwindow_size_ms / 1000
                optimal_time = optimal_window['start_time'] + relative_time
                print(f"Found lowest energy point ({optimal_window['db']:.1f}dB) at {optimal_time:.3f}s")
            
            final_time = max(start_time, min(end_time, optimal_time))
            if final_time != optimal_time:
                print(f"Adjusted time to {final_time:.3f}s to stay within bounds")
            
            return final_time
            
        except Exception as e:
            print(f"Error in find_lowest_energy_point: {str(e)}")
            return (start_time + end_time) / 2
            
    def find_optimal_cut_points(self, segments: List[Dict], audio: AudioSegment) -> List[Dict[str, Tuple[float, Optional[float], Optional[float]]]]:
        """
        Find optimal cut points between segments by analyzing the quietest point 
        between contextual words and segments.
        """
        if not segments:
            return []
            
        cut_points = []
        audio_length_sec = len(audio) / 1000  # Convert to seconds
        
        # Define window sizes at the start of the method
        max_start_window = 0.1  # 100ms for start cuts
        max_end_window = 0.3    # 300ms for end cuts
        
        # Create debug log file in the same directory as the audio
        debug_log_path = Path(segments[0]['words'][0].get('audio_file', 'debug')).parent / 'cut_points_debug.log'
        
        with open(debug_log_path, 'w', encoding='utf-8') as debug_log:
            debug_log.write("Cut Points Analysis Debug Log\n")
            debug_log.write("===========================\n")
            debug_log.write(f"Audio duration: {audio_length_sec:.3f}s\n")
            debug_log.write(f"Number of segments: {len(segments)}\n")
            debug_log.write(f"Max start window: {max_start_window*1000:.0f}ms\n")
            debug_log.write(f"Max end window: {max_end_window*1000:.0f}ms\n\n")
            
            for i in range(len(segments)):
                curr_segment = segments[i]
                curr_first_word = next((w for w in curr_segment["words"] if "start" in w), None)
                curr_last_word = next((w for w in reversed(curr_segment["words"]) if "end" in w), None)
                
                if not curr_first_word or not curr_last_word:
                    debug_log.write(f"\nSkipping segment {i} - Missing word timestamps\n")
                    continue
                    
                # Log segment details
                debug_log.write(f"\n{'='*20} Segment {i} {'='*20}\n")
                debug_log.write(f"Text: {' '.join(w['word'] for w in curr_segment['words'])}\n")
                debug_log.write(f"Duration: {curr_last_word['end'] - curr_first_word['start']:.3f}s\n")
                debug_log.write(f"Words: {len(curr_segment['words'])}\n")
                debug_log.write("\nWord timings:\n")
                for word in curr_segment['words']:
                    if 'start' in word and 'end' in word:
                        duration = word['end'] - word['start']
                        debug_log.write(f"  '{word['word']}': {word['start']:.3f}s - {word['end']:.3f}s ({duration*1000:.0f}ms)\n")
                
                # Context - previous word's end time and next word's start time
                prev_word_end = None
                next_word_start = None
                
                # Get previous segment info
                if i > 0:
                    prev_segment = segments[i-1]
                    prev_last_word = next((w for w in reversed(prev_segment["words"]) if "end" in w), None)
                    if prev_last_word:
                        prev_word_end = prev_last_word["end"]
                        debug_log.write(f"\nPrevious context: '{prev_last_word['word']}' ending at {prev_word_end:.3f}s\n")
                else:
                    debug_log.write(f"\nFirst segment - no previous word\n")
                
                # Get next segment info
                if i < len(segments) - 1:
                    next_segment = segments[i+1]
                    next_first_word = next((w for w in next_segment["words"] if "start" in w), None)
                    if next_first_word:
                        next_word_start = next_first_word["start"]
                        debug_log.write(f"Next context: '{next_first_word['word']}' starting at {next_word_start:.3f}s\n")
                else:
                    debug_log.write(f"Last segment - no next word\n")
                
                # Calculate search windows
                start_search_end = curr_first_word["start"]
                if prev_word_end is not None and (start_search_end - prev_word_end) < max_start_window:
                    # If previous word is closer than 100ms, look back to that word
                    # But ensure at least 50ms window
                    start_search_start = min(prev_word_end, start_search_end - 0.05)
                else:
                    # Otherwise only look back 100ms
                    start_search_start = max(0, start_search_end - max_start_window)

                end_search_start = curr_last_word["end"]
                if next_word_start is not None and (next_word_start - end_search_start) < max_end_window:
                    # If next word is closer than 200ms, look forward to that word
                    end_search_end = next_word_start
                else:
                    # Otherwise only look forward 200ms
                    end_search_end = min(audio_length_sec, end_search_start + max_end_window)
                
                debug_log.write("\nSearching for cut points:\n")
                debug_log.write(f"Start window: {start_search_start:.3f}s - {start_search_end:.3f}s ({(start_search_end-start_search_start)*1000:.0f}ms)\n")
                debug_log.write(f"End window: {end_search_start:.3f}s - {end_search_end:.3f}s ({(end_search_end-end_search_start)*1000:.0f}ms)\n")
                
                # Find optimal points
                start_time = self.find_lowest_energy_point(audio, start_search_start, start_search_end, is_start_cut=True)
                end_time = self.find_lowest_energy_point(audio, end_search_start, end_search_end, is_start_cut=False)
                
                cut_points.append({
                    "start": start_time,
                    "end": end_time,
                    "prev_word_end": prev_word_end,
                    "next_word_start": next_word_start
                })
                
                # Log final results
                debug_log.write("\nFinal cut points:\n")
                debug_log.write(f"Start cut at: {start_time:.3f}s\n")
                debug_log.write(f"  Distance from search start: {(start_time-start_search_start)*1000:.0f}ms\n")
                debug_log.write(f"  Distance to first word: {(curr_first_word['start']-start_time)*1000:.0f}ms\n")
                if prev_word_end is not None:
                    debug_log.write(f"  Gap from previous word: {(start_time-prev_word_end)*1000:.0f}ms\n")
                
                debug_log.write(f"End cut at: {end_time:.3f}s\n")
                debug_log.write(f"  Distance from search start: {(end_time-end_search_start)*1000:.0f}ms\n")
                debug_log.write(f"  Distance from last word: {(end_time-curr_last_word['end'])*1000:.0f}ms\n")
                if next_word_start is not None:
                    debug_log.write(f"  Gap to next word: {(next_word_start-end_time)*1000:.0f}ms\n")
                
                debug_log.write(f"Final segment duration: {end_time-start_time:.3f}s\n")
                debug_log.write("="*50 + "\n")
        
        return cut_points

    def check_for_abrupt_ending(self, audio: np.ndarray, sr: int, threshold_ms: int = 50) -> bool:
        """
        Check if audio segment ends abruptly using multiple factors with more sensitive thresholds.
        """
        try:
            # Convert audio to float32 and normalize
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            # Analyze the final portion of audio
            tail_ms = 100  # Look at final 100ms
            tail_samples = min(int((tail_ms / 1000) * sr), len(audio) // 3)
            tail = audio[-tail_samples:]
            
            # Calculate RMS energy in small windows
            window_ms = 5  # 5ms windows for finer resolution
            window_size = int((window_ms / 1000) * sr)
            windows = np.array_split(tail, tail_samples // window_size)
            energy = np.array([np.sqrt(np.mean(w**2)) for w in windows])
            
            # Calculate metrics
            end_energy = np.mean(energy[-3:])
            peak_energy = np.max(energy)
            energy_ratio = end_energy / peak_energy
            
            # Calculate energy slope at the end
            if len(energy) >= 4:
                end_slope = (energy[-1] - energy[-4]) / 3
            else:
                end_slope = 0
                
            # Zero-crossing analysis
            zcr_windows = np.array_split(tail, 4)
            zcr_rates = [np.sum(np.abs(np.diff(np.signbit(w)))) / len(w) for w in zcr_windows]
            zcr_variance = np.var(zcr_rates)
            zcr_change = abs(zcr_rates[-1] - np.mean(zcr_rates[:-1])) if len(zcr_rates) > 1 else 0
            
            # Check for sudden amplitude drop
            amplitude_envelope = np.abs(tail)
            final_samples = amplitude_envelope[-int(0.01 * sr):]  # Last 10ms
            sudden_drop = np.mean(final_samples) < 0.1 * np.mean(amplitude_envelope)
            
            # More sensitive thresholds
            is_abrupt = (
                (energy_ratio > 0.5 or  # Energy doesn't drop enough
                 end_slope > 0 or       # Energy increases at the end
                 zcr_variance > 0.05 or # Irregular zero crossings
                 zcr_change > 0.3 or    # Sudden change in zero-crossing rate
                 sudden_drop)           # Sudden amplitude drop
                and
                end_energy > 0.1        # Ensure there's significant energy at the end
            )
            
            # Print detailed analysis if it's abrupt
            if is_abrupt:
                print("\nAbrupt ending detected:")
                print(f"Energy ratio: {energy_ratio:.3f} (> 0.5 suggests abrupt)")
                print(f"End slope: {end_slope:.3f} (positive suggests abrupt)")
                print(f"ZCR variance: {zcr_variance:.3f} (> 0.05 suggests abrupt)")
                print(f"ZCR change: {zcr_change:.3f} (> 0.3 suggests abrupt)")
                print(f"Sudden drop: {sudden_drop}")
                print(f"End energy: {end_energy:.3f}")
                
            return is_abrupt
            
        except Exception as e:
            print(f"Error in check_for_abrupt_ending: {str(e)}")
            print(f"Audio shape: {audio.shape if isinstance(audio, np.ndarray) else 'invalid'}")
            print(f"Audio type: {audio.dtype if isinstance(audio, np.ndarray) else 'invalid'}")
            return False
        
    def _calculate_zcr_variance(self, audio: np.ndarray) -> float:
        """Calculate zero-crossing rate variance with adaptive windowing."""
        window_size = len(audio) // 4
        zcr_groups = [
            np.sum(np.abs(np.diff(np.signbit(g)))) / len(g)
            for g in np.array_split(audio, max(4, len(audio) // window_size))
        ]
        return np.var(zcr_groups)

    def process_segments(self, input_path: str, segments: List[Dict], output_dir: Path, args) -> List[Dict[str, str]]:
        try:
            # Load audio file
            audio = AudioSegment.from_wav(input_path)
            
            # Track statistics for final report
            total_segments = len(segments)
            discarded_segments = []
            
            # Apply negative offset to last words
            adjusted_segments = []
            for segment in segments:
                if segment['words']:
                    # Create a deep copy of the segment
                    adjusted_segment = {
                        'words': segment['words'][:-1],  # All words except last
                        'text': segment['text']
                    }
                    
                    # Adjust the last word's end time
                    last_word = segment['words'][-1].copy()
                    if 'end' in last_word:
                        base_offset = args.negative_offset_last_word / 1000
                        if any(last_word['word'].strip().endswith(p) for p in '.!?'):
                            offset = base_offset * 2
                        else:
                            offset = base_offset
                            
                        last_word['end'] = max(
                            last_word['start'],
                            last_word['end'] - offset
                        )
                    adjusted_segment['words'].append(last_word)
                    adjusted_segments.append(adjusted_segment)
            
            cut_points = self.find_optimal_cut_points(adjusted_segments, audio)
            processed_segments = []
            
            print(f"\nProcessing {total_segments} segments from {Path(input_path).name}...")
            
            # Process each segment
            for i, cut_point in enumerate(cut_points):
                start_ms = int(cut_point["start"] * 1000)
                end_ms = int(cut_point["end"] * 1000)
                
                # Extract segment
                segment_audio = audio[start_ms:end_ms]
                segment_duration = len(segment_audio) / 1000.0  # duration in seconds
                segment_text = " ".join(w["word"] for w in adjusted_segments[i]["words"]).strip()
                
                # Check for abrupt ending if flag is set
                if args.discard_abrupt:
                    samples = np.array(segment_audio.get_array_of_samples(), dtype=np.float32)
                    samples = samples / max(np.max(np.abs(samples)), 1)  # Normalize to [-1, 1]
                    if self.check_for_abrupt_ending(samples, segment_audio.frame_rate):
                        discarded_segments.append({
                            "index": i,
                            "duration": segment_duration,
                            "text": segment_text,
                            "reason": "abrupt ending"
                        })
                        print(f"\n⚠️  Discarding segment {i}:")
                        print(f"   Duration: {segment_duration:.2f}s")
                        print(f"   Text: \"{segment_text}\"")
                        continue
                
                # Generate output filename
                output_path = output_dir / f"{Path(input_path).stem}_segment_{i}.wav"
                
                # Export raw segment
                segment_audio.export(str(output_path), format="wav")
                
                # Apply audio processing chain
                if any([args.normalize is not None, args.dess, args.denoise, 
                       args.compress, args.fade_in > 0, args.fade_out > 0, args.trim]):
                    success = self.process_audio(
                        str(output_path),
                        str(output_path),
                        normalize_target=-float(args.normalize) if args.normalize else None,
                        dess_profile='high' if args.dess else None,
                        denoise=args.denoise,
                        compress_profile=args.compress if args.compress else None,
                        fade_in_ms=args.fade_in,
                        fade_out_ms=args.fade_out,
                        trim=args.trim
                    )
                    
                    if not success:
                        discarded_segments.append({
                            "index": i,
                            "duration": segment_duration,
                            "text": segment_text,
                            "reason": "processing failed"
                        })
                        print(f"\n⚠️  Discarding segment {i}:")
                        print(f"   Duration: {segment_duration:.2f}s")
                        print(f"   Text: \"{segment_text}\"")
                        print(f"   Reason: Audio processing failed")
                        continue
                
                # Store segment information
                processed_segments.append({
                    "audio_file": str(output_path),
                    "text": segment_text,
                    "speaker_name": "001"
                })
            
            # Print final statistics
            print("\nSegment Processing Summary:")
            print(f"Total segments: {total_segments}")
            print(f"Successfully processed: {len(processed_segments)}")
            if discarded_segments:
                print(f"Discarded segments: {len(discarded_segments)}")
                print("\nDiscarded segments details:")
                for disc in discarded_segments:
                    print(f"\nSegment {disc['index']}:")
                    print(f"Duration: {disc['duration']:.2f}s")
                    print(f"Text: \"{disc['text']}\"")
                    print(f"Reason: {disc['reason']}")
            
            return processed_segments
            
        except Exception as e:
            print(f"Error processing segments: {str(e)}")
            return []

    def _init_deepfilter(self):
        """Initialize DeepFilterNet model lazily"""
        if self.deepfilter_model is None:
            self.deepfilter_model, self.df_state, _ = init_df()

    def process_audio(self, input_path: str, output_path: str, 
                     normalize_target: Optional[float] = None,
                     dess_profile: Optional[str] = None,
                     denoise: bool = False,
                     compress_profile: Optional[str] = None,
                     fade_in_ms: Optional[int] = None,
                     fade_out_ms: Optional[int] = None,
                     trim: bool = False) -> bool:
        try:
            # Load and pre-process audio
            if denoise:
                # Use DeepFilterNet's own loading and processing functions
                self._init_deepfilter()
                audio, _ = load_audio(input_path, sr=self.df_state.sr())
                # Denoise the audio
                audio = enhance(self.deepfilter_model, self.df_state, audio)
                # Convert tensor to numpy array
                audio = audio.cpu().numpy()
                if audio.ndim == 2:  # If 2D array (channels, samples)
                    audio = audio[0]  # Take first channel
                sr = self.df_state.sr()
            else:
                # Load audio normally if no denoising needed
                audio, sr = sf.read(input_path)
                
            # Validate audio array
            if audio is None or len(audio) == 0:
                print(f"Error: Empty or invalid audio loaded from {input_path}")
                return False
                
            # Ensure audio is floating point with correct range
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))

            # Resample if needed
            if sr != self.target_sr:
                print(f"Resampling from {sr}Hz to {self.target_sr}Hz")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            # Process order matters:
            # 1. Normalize before compression (if requested)
            if normalize_target:
                print(f"Normalizing to {normalize_target} LUFS")
                audio = self._normalize(audio, target_lufs=normalize_target)
            
            # 2. Compress after normalization
            if compress_profile:
                print(f"Applying {compress_profile} compression profile")
                audio = self._compress(audio, compress_profile)
            
            # 3. De-ess after compression
            if dess_profile:
                print(f"Applying {dess_profile} de-essing profile")
                audio = self._deess(audio, dess_profile)
            
            # 4. Trim silence before fades
            if trim:
                print("Removing trailing silence")
                audio = self.remove_trailing_silence(audio, sr)
            
            # 5. Apply fades after trimming
            if fade_in_ms or fade_out_ms:
                print(f"Applying fades: in={fade_in_ms}ms, out={fade_out_ms}ms")
                audio = self.apply_fades(audio, sr, fade_in_ms, fade_out_ms)
            
            # Final peak normalization to prevent clipping
            peak = np.max(np.abs(audio))
            if peak > 0.99:
                print("Applying final peak normalization")
                audio = audio * (0.99 / peak)
            
            # Validate final audio
            if np.isnan(audio).any() or np.isinf(audio).any():
                print("Error: Audio contains NaN or Inf values after processing")
                return False
                
            # Save processed audio
            try:
                sf.write(output_path, audio, sr)
                if not os.path.exists(output_path):
                    print(f"Error: File not written to {output_path}")
                    return False
            except Exception as e:
                print(f"Error saving audio file: {str(e)}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            print(f"Audio shape: {audio.shape if 'audio' in locals() else 'not loaded'}")
            print(f"Audio type: {type(audio) if 'audio' in locals() else 'not loaded'}")
            print(f"Sample rate: {sr if 'sr' in locals() else 'not loaded'}")
            return False
        finally:
            # Clear any GPU memory if used
            if denoise and hasattr(self, 'deepfilter_model'):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
    def _normalize(self, audio: np.ndarray, target_lufs: float = -16.0) -> np.ndarray:
        """Normalize audio to target LUFS with true peak limiting"""
        # Calculate current LUFS
        rms = np.sqrt(np.mean(audio**2))
        current_lufs = 20 * np.log10(rms) - 0.691
        
        # Calculate required gain
        gain = 10 ** ((target_lufs - current_lufs) / 20)
        
        # Apply true peak limiting
        peak = np.max(np.abs(audio * gain))
        if peak > 0.99:
            gain *= 0.99 / peak
        
        return audio * gain

    def _compress(self, audio: np.ndarray, profile: Literal['male', 'female', 'neutral']) -> np.ndarray:
        """Dynamic range compression with voice-optimized profiles"""
        params = self.compression_profiles[profile]
        
        # Convert threshold to linear
        threshold_linear = 10 ** (params['threshold'] / 20)
        knee_lower = threshold_linear * (10 ** (-params['knee_width'] / 40))
        knee_upper = threshold_linear * (10 ** (params['knee_width'] / 40))
        
        # Calculate gain reduction
        gain_reduction = np.zeros_like(audio)
        
        # Below knee
        mask_below = np.abs(audio) <= knee_lower
        gain_reduction[mask_below] = 0
        
        # Above knee
        mask_above = np.abs(audio) >= knee_upper
        gain_above = (1 - 1/params['ratio']) * (20 * np.log10(np.abs(audio[mask_above])) - params['threshold'])
        gain_reduction[mask_above] = gain_above
        
        # In knee
        mask_knee = ~mask_below & ~mask_above
        knee_curve = (1 - 1/params['ratio']) * ((20 * np.log10(np.abs(audio[mask_knee])) - params['threshold'] + params['knee_width']/2)**2 / (2 * params['knee_width']))
        gain_reduction[mask_knee] = knee_curve
        
        # Apply attack/release envelope
        attack_coef = np.exp(-1 / (params['attack'] * self.target_sr))
        release_coef = np.exp(-1 / (params['release'] * self.target_sr))
        
        gain_reduction_smoothed = np.zeros_like(gain_reduction)
        for i in range(1, len(gain_reduction)):
            if gain_reduction[i] <= gain_reduction_smoothed[i-1]:
                # Attack phase
                gain_reduction_smoothed[i] = attack_coef * gain_reduction_smoothed[i-1] + (1 - attack_coef) * gain_reduction[i]
            else:
                # Release phase
                gain_reduction_smoothed[i] = release_coef * gain_reduction_smoothed[i-1] + (1 - release_coef) * gain_reduction[i]
        
        # Convert gain reduction to linear domain and apply
        gain_reduction_linear = 10 ** (gain_reduction_smoothed / 20)
        return audio * gain_reduction_linear

    def _deess(self, audio: np.ndarray, profile: Literal['low', 'high']) -> np.ndarray:
        """Enhanced de-essing with two profiles"""
        params = self.dess_profiles[profile]
        
        # Create bandpass filter for sibilance detection
        nyquist = self.target_sr / 2
        low_freq = params['freq_range'][0] / nyquist
        high_freq = params['freq_range'][1] / nyquist
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        
        # Extract and process sibilance
        sibilants = signal.filtfilt(b, a, audio)
        
        # Calculate adaptive threshold
        rms = np.sqrt(np.mean(sibilants**2))
        adaptive_threshold = params['threshold'] * rms
        
        # Apply compression to sibilants
        mask = np.abs(sibilants) > adaptive_threshold
        sibilants[mask] = (
            adaptive_threshold + 
            (np.abs(sibilants[mask]) - adaptive_threshold) / params['ratio']
        ) * np.sign(sibilants[mask])
        
        # Apply makeup gain
        sibilants *= params['makeup_gain']
        
        # Smooth transitions
        env_b, env_a = signal.butter(2, 150 / nyquist, btype='low')
        smoothed_sibilants = signal.filtfilt(env_b, env_a, sibilants)
        
        # Mix processed sibilants back with original
        return audio - smoothed_sibilants * params['reduction_factor']

def process_audio_files(input_files, output_dir, args):
    """Process audio files with the specified effects"""
    processor = AudioProcessor(target_sr=args.sample_rate)
    
    for input_file in input_files:
        try:
            input_path = Path(input_file)
            output_path = Path(output_dir) / input_path.name
            
            # Process with selected effects
            success = processor.process_audio(
                str(input_path),
                str(output_path),
                normalize_target=-float(args.normalize) if args.normalize else None,
                dess_profile='high' if args.dess else None,
                denoise=args.denoise,
                compress_profile=args.compress if args.compress else None
            )
            
            if success:
                print(f"Successfully processed: {input_path.name}")
            else:
                print(f"Failed to process: {input_path.name}")
                
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")

@dataclass
class TrainingMetrics:
    model_name: str
    start_time: float
    end_time: float = 0
    total_audio_duration_minutes: float = 0  # in minutes
    total_training_minutes: float = 0
    num_samples: int = 0
    num_batches: int = 0
    num_epochs: int = 0
    gradient_accumulation: int = 0
    language: str = ""
    sample_rate: int = 0
    text_losses: List[float] = None
    mel_losses: List[float] = None
    
    def __post_init__(self):
        self.text_losses = self.text_losses or []
        self.mel_losses = self.mel_losses or []
    
    def update_training_time(self):
        """Calculate training time in minutes"""
        if self.end_time:
            self.total_training_minutes = round((self.end_time - self.start_time) / 60, 2)
    
    def parse_loss_values(self, log_text: str, loss_type: str) -> List[float]:
        losses = []
        lines = log_text.split('\n')
        for line in lines:
            if f"avg_loss_{loss_type}_ce:" in line:
                log_position = log_text.find(line)
                context_before = log_text[max(0, log_position-1000):log_position]
                
                if '> EVALUATION' in context_before:
                    number_match = re.search(r'\x1b$$\d+m\s*([\d.]+)', line)
                    if not number_match:
                        number_match = re.search(r'ce:\s*([\d.]+)', line)
                    
                    if number_match:
                        value = float(number_match.group(1))
                        losses.append(value)
        return losses
    
    def update_from_log(self, log_path: Path):
        with open(log_path, 'r', encoding='utf-8') as f:
            log_text = f.read()
        
        self.text_losses = self.parse_loss_values(log_text, "text")
        self.mel_losses = self.parse_loss_values(log_text, "mel")
    
    def calculate_audio_stats(self, audio_segments: List[Dict]):
        """Calculate audio statistics from the training segments"""
        self.num_samples = len(audio_segments)
        self.num_batches = len(audio_segments) // 2  # assuming batch_size=2
        total_seconds = sum(
            AudioSegment.from_wav(seg['audio_file']).duration_seconds 
            for seg in audio_segments
        )
        self.total_audio_duration_minutes = round(total_seconds / 60, 2)
    
    # def save(self, output_dir: Path):
        # metrics_file = output_dir / "training_metrics.json"
        # self.update_training_time()  # Update training time before saving
        
        # # Create a clean dict without internal tracking fields
        # metrics_dict = {
            # "model_name": self.model_name,
            # "total_training_minutes": self.total_training_minutes,
            # "total_audio_duration_minutes": self.total_audio_duration_minutes,
            # "num_samples": self.num_samples,
            # "num_batches": self.num_batches,
            # "num_epochs": self.num_epochs,
            # "gradient_accumulation": self.gradient_accumulation,
            # "language": self.language,
            # "sample_rate": self.sample_rate,
            # "text_losses": self.text_losses,
            # "mel_losses": self.mel_losses
        # }
        
        # with open(metrics_file, 'w', encoding='utf-8') as f:
            # json.dump(metrics_dict, f, indent=2)

def parse_arguments():
    parser = argparse.ArgumentParser(description="XTTS Model Training App")
    parser.add_argument("--source-language", choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu"], required=True, help="Source language for XTTS")
    parser.add_argument("--whisper-model", choices=["medium", "medium.en", "large-v2", "large-v3"], default="large-v3", help="Whisper model to use for transcription")
    parser.add_argument("--enhance", action="store_true", help="Enable audio enhancement")
    parser.add_argument("-i", "--input", required=True, help="Input folder or single file")
    parser.add_argument("--session", help="Name for the session folder")
    parser.add_argument("--separate", action="store_true", help="Enable speech separation")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--xtts-base-model", default="v2.0.2", help="XTTS base model version")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient", type=int, default=1, help="Gradient accumulation levels")
    parser.add_argument("--xtts-model-name", help="Name for the trained model")
    parser.add_argument("--sample-method", choices=["maximise-punctuation", "punctuation-only", "mixed"], default="maximise-punctuation", help="Method for preparing training samples")
    parser.add_argument("-conda_env", help="Name of the Conda environment to use")
    parser.add_argument("-conda_path", help="Path to the Conda installation folder")
    parser.add_argument("--sample-rate", type=int, choices=[22050, 44100], default=22050, 
                   help="Sample rate for WAV files (default: 22050, recommended for XTTS training)")
    parser.add_argument("--max-audio-time", type=float, default=11,
                       help="Maximum audio duration in seconds (default: 11.6)")
    parser.add_argument("--max-text-length", type=int, default=200,
                       help="Maximum text length in characters (default: 200)")
    parser.add_argument("--align-model", 
                       help="""Model to use for phoneme-level alignment. Common options for English include:
                       - WAV2VEC2_ASR_LARGE_LV60K_960H (largest, most accurate)
                       - WAV2VEC2_ASR_BASE_960H (medium size)
                       - WAV2VEC2_ASR_BASE_100H (smallest, fastest)""")
    parser.add_argument("--normalize", type=float, nargs='?', const=16.0,
                       help="Normalize audio to target LUFS (default: -16.0 if no value provided)")
    parser.add_argument("--dess", action="store_true",
                       help="Apply de-essing to reduce sibilance")
    parser.add_argument("--denoise", action="store_true",
                       help="Apply DeepFilterNet noise reduction")
    parser.add_argument("--compress", choices=['male', 'female', 'neutral'],
                       help="Apply dynamic range compression with voice-specific profile")
    
    parser.add_argument("--method-proportion", default="6_4",
                       help="For mixed method, proportion of maximise-punctuation to punctuation-only (e.g., '6_4' for 60-40 split)")
    parser.add_argument("--training-proportion", default="8_2",
                       help="Proportion of training to validation data (e.g., '8_2' for 80-20 split)")
    parser.add_argument("--negative-offset-last-word", type=int, default=50,
                   help="Subtract this many milliseconds from the end time of the last word in each segment (default: 50)")
    parser.add_argument("--breath", action="store_true",
                   help="Apply breath removal preprocessing")
    parser.add_argument("--trim", action="store_true",
                   help="Automatically trim trailing silence from segments while preserving word endings")
    parser.add_argument("--fade-in", type=int, metavar="MS", default=30,
                       help="Apply fade-in effect for specified milliseconds from start (default: 30ms)")
    parser.add_argument("--fade-out", type=int, metavar="MS", default=40,
                       help="Apply fade-out effect for specified milliseconds from end (default: 40ms)")
    parser.add_argument("--discard-abrupt", action="store_true",
                   help="Detect and discard segments with abrupt endings")
    
    args = parser.parse_args()

    # Set default alignment models based on language if not specified by user
    if not args.align_model:
        language_align_models = {
            "pl": "jonatasgrosman/wav2vec2-xls-r-1b-polish",
            "nl": "FvH14/wav2vec2-XLSR-53-DutchCommonVoice12",
            "de": "jonatasgrosman/wav2vec2-xls-r-1b-german",
            "en": "jonatasgrosman/wav2vec2-xls-r-1b-english"
        }
        args.align_model = language_align_models.get(args.source_language)
    
    # Validate and convert proportions
    if args.method_proportion:
        try:
            max_prop, punct_prop = map(int, args.method_proportion.split('_'))
            if max_prop + punct_prop != 10:
                raise ValueError("Method proportions must sum to 10")
            args.method_proportion = max_prop / 10
        except (ValueError, AttributeError):
            raise ValueError("Method proportion must be in format 'N_M' where N+M=10 (e.g., '6_4')")
    
    if args.training_proportion:
        try:
            train_prop, val_prop = map(int, args.training_proportion.split('_'))
            if train_prop + val_prop != 10:
                raise ValueError("Training proportions must sum to 10")
            args.training_proportion = train_prop / 10
        except (ValueError, AttributeError):
            raise ValueError("Training proportion must be in format 'N_M' where N+M=10 (e.g., '8_2')")
    
    if args.negative_offset_last_word is None:
        args.negative_offset_last_word = 0 if args.source_language == "en" else 50
    
    return args

def create_session_folder(session_name):
    if not session_name:
        session_name = f"xtts-finetune-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    session_path = Path(session_name)
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path

def transcribe_audio(audio_file, args, session_path):
    output_dir = (session_path / "transcriptions").resolve()
    output_dir.mkdir(exist_ok=True)
    
    audio_file_absolute = audio_file.resolve()
    
    def get_conda_path():
        if args.conda_path:
            return os.path.join(args.conda_path, "conda.exe")
        
        possible_paths = [
            os.path.join(os.environ.get('USERPROFILE', ''), "anaconda3", "Scripts", "conda.exe"),
            os.path.join(os.environ.get('USERPROFILE', ''), "miniconda3", "Scripts", "conda.exe"),
            r"C:\ProgramData\Anaconda3\Scripts\conda.exe",
            r"C:\ProgramData\Miniconda3\Scripts\conda.exe"
        ]
        
        if args.conda_env:
            for path in possible_paths:
                if os.path.exists(path):
                    return path
        
        return str(conda_path)
    
    conda_executable = get_conda_path()
    
    def run_whisperx(env):
        # Build base command without align_model
        command = [
            conda_executable,
            "run",
            "-n",
            env,
            "python",
            "-m",
            "whisperx",
            str(audio_file_absolute),
            "--language", args.source_language,
            "--model", args.whisper_model,
            "--output_dir", str(output_dir),
            "--output_format", "json"
        ]
        
        # Only add align_model if it was specified
        if hasattr(args, 'align_model') and args.align_model:
            command.extend(["--align_model", args.align_model])
        
        # Check if GPU is from Pascal generation
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            pascal_gpus = ['1060', '1070', '1080', '1660', '1650']
            if any(gpu in gpu_name for gpu in pascal_gpus):
                command.extend(["--compute_type", "int8"])
        
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running WhisperX with environment {env}: {e}")
            print(f"Standard output: {e.stdout}")
            print(f"Standard error: {e.stderr}")
            return False
    
    if args.conda_env:
        if not run_whisperx(args.conda_env):
            raise Exception(f"Failed to run WhisperX with provided environment: {args.conda_env}")
    else:
        # Try with hardcoded path and whisperx_installer
        if not run_whisperx("whisperx_installer"):
            # If it fails, try with "whisperx" and one of the usual Windows paths
            conda_executable = next((path for path in possible_paths if os.path.exists(path)), None)
            if conda_executable:
                if not run_whisperx("whisperx"):
                    raise Exception("Failed to run WhisperX with both whisperx_installer and whisperx environments")
            else:
                raise Exception("Could not find a valid conda path")
    
    json_file = output_dir / f"{audio_file.stem}.json"
    return json_file

def create_log_file(session_path):
    log_file = session_path / f"session-log--{datetime.now().strftime('%Y-%m-%d--%H-%M')}.txt"
    return log_file

def create_json_file(session_path):
    json_file = session_path / f"session-data--{datetime.now().strftime('%Y-%m-%d--%H-%M')}.json"
    return json_file

def convert_to_wav(input_path, output_path, target_sample_rate):
    # If input is already a WAV file, check its properties
    if input_path.suffix.lower() == '.wav':
        try:
            waveform, sample_rate = torchaudio.load(input_path)
            is_mono = waveform.shape[0] == 1
            
            # If the file already matches our requirements, just copy it
            if sample_rate == target_sample_rate and is_mono:
                shutil.copy2(input_path, output_path)
                return
            
        except Exception as e:
            print(f"Error reading WAV file {input_path}: {e}")
            # Continue to regular conversion if there's an error reading the WAV
    
    # Convert the file if it's not WAV or doesn't match requirements
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(target_sample_rate).set_channels(1).set_sample_width(2)
    audio.export(output_path, format="wav")

def process_audio(input_path, session_path, args):
    """Process audio files and prepare them for training"""
    # Create necessary directories
    audio_sources_dir = session_path / "audio_sources"
    audio_sources_dir.mkdir(exist_ok=True, parents=True)
    processed_dir = audio_sources_dir / "processed"
    processed_dir.mkdir(exist_ok=True, parents=True)

    if os.path.isfile(input_path):
        files = [Path(input_path)]
    else:
        input_path = Path(input_path)
        files = [f for f in input_path.rglob('*') if f.suffix.lower() in ('.mp3', '.wav', '.flac', '.ogg', '.webm')]

    # Check if breath removal is available if --breath is passed
    if args.breath:
        try:
            # Try to run breath-removal with --help or -h to check if it's available
            result = subprocess.run(["breath-removal", "--help"], 
                                 capture_output=True, 
                                 text=True)
            breath_removal_available = True
            print("Breath removal tool found and available.")
        except FileNotFoundError:
            breath_removal_available = False
            print("Warning: breath-removal command not found. Continuing without breath removal.")
            args.breath = False  # Disable breath removal
        except subprocess.CalledProcessError as e:
            # Command exists but returned error - might be okay if --help isn't supported
            breath_removal_available = True
            print("Breath removal tool found.")

    # Create a temporary directory for breath removal if needed
    breath_removal_dir = None
    if args.breath and breath_removal_available:
        breath_removal_dir = audio_sources_dir / "breath_removal_temp"
        breath_removal_dir.mkdir(exist_ok=True)
        print(f"Created temporary directory for breath removal: {breath_removal_dir}")
        
    processed_files = []
    
    for file in files:
        try:
            if args.breath and breath_removal_available:
                # Run breath removal
                print(f"Processing {file.name} with breath removal...")
                breath_output = breath_removal_dir / f"breath_removal_{file.name}"
                
                try:
                    cmd = [
                        "breath-removal",  # Changed from breath_removal to breath-removal
                        "-i", str(file.absolute()),
                        "-o", str(breath_removal_dir)
                    ]
                    print(f"Running command: {' '.join(cmd)}")
                    
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    
                    # Check if the breath removal output exists
                    if breath_output.exists():
                        print(f"Breath removal successful for {file.name}")
                        file_to_process = breath_output
                    else:
                        print(f"Warning: Breath removal output not found for {file.name}, using original file")
                        file_to_process = file
                except subprocess.CalledProcessError as e:
                    print(f"Error during breath removal for {file.name}:")
                    print(f"Command output: {e.stdout}")
                    print(f"Command error: {e.stderr}")
                    print("Using original file instead")
                    file_to_process = file
            else:
                file_to_process = file

            print(f"Converting {file_to_process.name} to WAV format...")
            # Convert to WAV with target sample rate
            output_file = audio_sources_dir / f"{file.stem}.wav"
            convert_to_wav(file_to_process, output_file, args.sample_rate)
            processed_files.append(output_file)
            print(f"Successfully processed {file.name} -> {output_file.name}")
            
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            continue

    # Clean up breath removal temporary directory if it was created
    if breath_removal_dir and breath_removal_dir.exists():
        try:
            shutil.rmtree(breath_removal_dir)
            print("Cleaned up breath removal temporary directory")
        except Exception as e:
            print(f"Warning: Could not remove temporary breath removal directory: {e}")

    if not processed_files:
        print("No files were successfully processed!")
        return None

    print(f"Successfully processed {len(processed_files)} files")
    return audio_sources_dir

def is_abbreviation(text, pos):
    """Check if punctuation is part of an abbreviation."""
    common_abbrev = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "et.", "al.", 
        "etc.", "e.g.", "i.e.", "vs.", "Ph.D.", "M.D.", "B.A.", "M.A.",
        "a.m.", "p.m.", "U.S.", "U.K.", "St."
    ]
    
    # Look back up to 5 characters to catch abbreviations
    start = max(0, pos - 5)
    text_slice = text[start:pos + 1].lower()
    
    return any(abbr.lower() in text_slice for abbr in common_abbrev)

def find_real_sentence_end(text, pos):
    """
    Determine if a punctuation mark is a real sentence ending.
    Handles cases like ." .) .") etc.
    """
    if pos >= len(text):
        return pos
    
    stack = []
    i = pos + 1
    
    while i < len(text):
        if text[i] in '"\'(':
            stack.append(text[i])
        elif text[i] in '"\')':
            if stack and ((stack[-1] == '"' and text[i] == '"') or 
                         (stack[-1] == "'" and text[i] == "'") or
                         (stack[-1] == "(" and text[i] == ")")):
                stack.pop()
        elif text[i] not in ' \t\n' and not stack:
            break
        i += 1
    
    return i - 1

def parse_transcription(json_file, audio_file, processed_dir, session_data, sample_method, args):
    """Parse transcription and extract training segments with clean cuts."""
    with open(json_file, 'r', encoding='utf-8') as f:
        transcription = json.load(f)
    
    processor = AudioProcessor(target_sr=args.sample_rate)
    qualifying_segments = session_data['qualifying_segments']
    
    # Extract words directly from word_segments
    all_words = transcription.get('word_segments', [])
    
    # Initialize containers for segment information
    pending_segments = []
    
    def is_valid_timestamps(word):
        """Check if word has valid timestamps."""
        if not all(key in word for key in ['start', 'end', 'word']):
            return False
        try:
            float(word['start'])
            float(word['end'])
            return True
        except (ValueError, TypeError):
            return False
            
    def find_next_valid_break(words, start_idx):
        """Find next valid punctuation break with timestamps."""
        for i in range(start_idx, len(words)):
            word = words[i]
            if is_valid_timestamps(word) and any(p in word['word'] for p in '.!?;,-'):
                if not is_abbreviation(word['word'], len(word['word'])-1):
                    return i
        return None
        
    def calculate_segment_duration(words):
        """Calculate duration between first and last word with valid timestamps."""
        valid_words = [w for w in words if is_valid_timestamps(w)]
        if not valid_words:
            return 0
        return float(valid_words[-1]['end']) - float(valid_words[0]['start'])

    def process_maximise_punctuation(current_segment=None):
        if current_segment is None:
            current_segment = {
                "words": [],
                "text": "",
                "break_points": []
            }
        
        segments_processed = 0
        try:
            words_to_process = current_segment["words"]
            
            i = 0
            while i < len(words_to_process):
                word = words_to_process[i]
                
                # Check if current word has valid timestamps
                if not is_valid_timestamps(word):
                    # If word has punctuation, we need to handle backtracking
                    if any(p in word['word'] for p in '.!?;,-'):
                        if current_segment["words"]:
                            # Try to build new segment from current segment's start up to last valid break
                            last_valid_break = None
                            current_duration = 0
                            
                            for idx, w in enumerate(current_segment["words"]):
                                if is_valid_timestamps(w):
                                    if any(p in w['word'] for p in '.!?;,-'):
                                        if not is_abbreviation(w['word'], len(w['word'])-1):
                                            temp_duration = calculate_segment_duration(current_segment["words"][:idx + 1])
                                            if temp_duration <= args.max_audio_time:
                                                last_valid_break = idx
                                                current_duration = temp_duration
                            
                            if last_valid_break is not None:
                                # Add segment to pending instead of saving immediately
                                valid_segment = current_segment["words"][:last_valid_break + 1]
                                pending_segments.append({
                                    "words": valid_segment,
                                    "text": " ".join(w['word'] for w in valid_segment)
                                })
                                segments_processed += 1
                        
                        # Whether we saved a segment or not, find next valid starting point
                        next_valid_idx = None
                        for j in range(i + 1, len(words_to_process)):
                            if (is_valid_timestamps(words_to_process[j]) and 
                                any(p in words_to_process[j]['word'] for p in '.!?;,-')):
                                next_valid_idx = j
                                break
                        
                        if next_valid_idx:
                            i = next_valid_idx + 1
                            current_segment["words"] = []
                            current_segment["text"] = ""
                            current_segment["break_points"] = []
                            continue
                        else:
                            break
                    
                    # Skip words without timestamps
                    i += 1
                    continue
                # Before adding word, check if current segment already exceeds limits
                if current_segment["words"]:
                    current_duration = calculate_segment_duration(current_segment["words"])
                    if current_duration > args.max_audio_time:
                        if current_segment["break_points"]:
                            optimal_break = current_segment["break_points"][-1]
                            pending_segments.append({
                                "words": optimal_break["words"],
                                "text": optimal_break["text"]
                            })
                            segments_processed += 1
                            # Reset segment
                            current_segment["words"] = []
                            current_segment["text"] = ""
                            current_segment["break_points"] = []
                        else:
                            # If no valid breaks, reset and continue
                            current_segment["words"] = []
                            current_segment["text"] = ""
                            current_segment["break_points"] = []
                
                # Add word to current segment
                current_segment["words"].append(word)
                current_segment["text"] += " " + word['word']
                
                duration = calculate_segment_duration(current_segment["words"])
                text_length = len(current_segment["text"])
                
                # Check if this word creates a potential break point
                if any(p in word['word'] for p in '.!?;,-'):
                    if not is_abbreviation(word['word'], len(word['word'])-1):
                        if duration <= args.max_audio_time and text_length <= args.max_text_length:
                            current_segment["break_points"].append({
                                "words": current_segment["words"].copy(),
                                "text": current_segment["text"],
                                "duration": duration
                            })
                
                # Check if we've exceeded limits
                if duration > args.max_audio_time or text_length > args.max_text_length:
                    if current_segment["break_points"]:
                        optimal_break = current_segment["break_points"][-1]
                        pending_segments.append({
                            "words": optimal_break["words"],
                            "text": optimal_break["text"]
                        })
                        segments_processed += 1
                        # Reset segment with remaining words
                        break_word_index = current_segment["words"].index(optimal_break["words"][-1])
                        current_segment["words"] = current_segment["words"][break_word_index + 1:]
                        current_segment["text"] = " ".join(word['word'] for word in current_segment["words"])
                        current_segment["break_points"] = []
                    else:
                        # If no valid breaks, reset and continue
                        current_segment["words"] = []
                        current_segment["text"] = ""
                        current_segment["break_points"] = []
                
                i += 1
            
            # Process any remaining segment
            if current_segment["words"] and current_segment["break_points"]:
                optimal_break = current_segment["break_points"][-1]
                pending_segments.append({
                    "words": optimal_break["words"],
                    "text": optimal_break["text"]
                })
                segments_processed += 1
            
            return segments_processed
        except Exception as e:
            print(f"Error in maximise_punctuation: {str(e)}")
            return 0
            
    def process_punctuation_only(current_segment=None):
        segments_processed = 0
        try:
            current_words = current_segment["words"] if current_segment else []
            i = 0
            
            words_to_process = current_segment["words"] if current_segment else []
            while i < len(words_to_process):  
                word = words_to_process[i]
                
                if not is_valid_timestamps(word):
                    if any(p in word['word'] for p in '.!?;,-'):
                        # Try to extend to next punctuation mark
                        next_break_idx = find_next_valid_break(all_words, i + 1)
                        if next_break_idx:
                            # Check if extended segment would be within limits
                            potential_segment = current_words + all_words[i:next_break_idx + 1]
                            duration = calculate_segment_duration(potential_segment)
                            text_length = len(" ".join(w['word'] for w in potential_segment))
                            
                            if duration <= args.max_audio_time and text_length <= args.max_text_length:
                                if duration >= 2.0:  # Minimum duration check
                                    pending_segments.append({
                                        "words": potential_segment,
                                        "text": " ".join(w['word'] for w in potential_segment)
                                    })
                                    segments_processed += 1
                                    current_words = []
                                    i = next_break_idx + 1
                                    continue
                        
                        # If extension failed, start fresh from next valid break
                        next_start = find_next_valid_break(all_words, i + 1)
                        if next_start:
                            i = next_start
                            current_words = []
                        else:
                            break
                    i += 1
                    continue
                
                current_words.append(word)
                
                if len(current_words) >= 2:
                    duration = calculate_segment_duration(current_words)
                    current_text = " ".join(w['word'] for w in current_words)
                    
                    if duration > args.max_audio_time or len(current_text) > args.max_text_length:
                        current_words = [word]
                        continue
                    
                    if duration >= 2.0 and any(p in word['word'] for p in '.!?;,-'):
                        if not is_abbreviation(word['word'], len(word['word'])-1):
                            pending_segments.append({
                                "words": current_words,
                                "text": current_text
                            })
                            segments_processed += 1
                            current_words = []
                
                i += 1
            
            return segments_processed
        except Exception as e:
            print(f"Error in process_punctuation_only: {str(e)}")
            return 0
            
    def process_mixed(method_proportion=0.6):
        if not all_words:
            return 0
            
        total_duration = calculate_segment_duration(all_words)
        split_target = total_duration * method_proportion
        
        # Find best split point
        split_index = None
        best_diff = float('inf')
        
        for i, word in enumerate(all_words):
            if is_valid_timestamps(word) and any(p in word['word'] for p in '.!?'):
                if not is_abbreviation(word['word'], len(word['word'])-1):
                    duration = calculate_segment_duration(all_words[:i + 1])
                    diff = abs(duration - split_target)
                    if diff < best_diff:
                        best_diff = diff
                        split_index = i
        
        segments_count = 0
        
        if split_index:
            # Split the words
            first_part = all_words[:split_index + 1]
            second_part = all_words[split_index + 1:]
            
            # Process first part with maximise-punctuation
            first_segment = {
                "words": first_part,
                "text": "",
                "break_points": []
            }
            segments_count += process_maximise_punctuation(first_segment)
            
            # Process second part with punctuation-only
            second_segment = {
                "words": second_part
            }
            segments_count += process_punctuation_only(second_segment)
        else:
            # If no good split point, use maximise-punctuation for whole file
            segments_count += process_maximise_punctuation()
        
        return segments_count

    # Main processing logic
    segments_count = 0
    if sample_method == "maximise-punctuation":
        segments_count = process_maximise_punctuation()
    elif sample_method == "punctuation-only":
        segments_count = process_punctuation_only()
    elif sample_method == "mixed":
        method_prop = args.method_proportion if hasattr(args, 'method_proportion') else 0.6
        segments_count = process_mixed(method_prop)
    
    # Process all pending segments with optimal cut points
    if pending_segments:
        processed_segments = processor.process_segments(
            str(audio_file),
            pending_segments,
            processed_dir,
            args
        )
        
        # Add processed segments to session data
        qualifying_segments.extend(processed_segments)
    
    return segments_count       

def create_metadata_files(session_data, session_path, training_proportion=0.8):  # Add parameter with default
    databases_dir = session_path / "databases"
    databases_dir.mkdir(exist_ok=True)

    all_samples = session_data['qualifying_segments']
    valid_samples = []

    print(f"Total samples in session_data: {len(all_samples)}")

    for sample in all_samples:
        audio_path = Path(sample['audio_file'])
        if audio_path.exists() and sample['text'].strip():
            valid_samples.append({
                "audio_file": str(audio_path.absolute()),
                "text": sample['text'].strip(),
                "speaker_name": sample['speaker_name']
            })
        else:
            print(f"Skipping invalid sample: file exists: {audio_path.exists()}, text: '{sample['text']}'")

    print(f"Valid samples: {len(valid_samples)}")

    if not valid_samples:
        print("No valid samples found. Check audio processing and transcription steps.")
        return None, None

    random.shuffle(valid_samples)

    train_size = int(training_proportion * len(valid_samples))  # Use the provided proportion
    train_samples = valid_samples[:train_size]
    eval_samples = valid_samples[train_size:]

    train_csv = databases_dir / "train_metadata.csv"
    eval_csv = databases_dir / "eval_metadata.csv"

    for csv_file, samples in [(train_csv, train_samples), (eval_csv, eval_samples)]:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='|')  # Use '|' as delimiter
            writer.writerow(["audio_file", "text", "speaker_name"])  # Write header
            for sample in samples:
                writer.writerow([sample['audio_file'], sample['text'], sample['speaker_name']])
        
        print(f"Created {csv_file} with {len(samples)} samples")

    return str(train_csv.absolute()), str(eval_csv.absolute())

def download_file(url, local_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as file, tqdm(
        desc=os.path.basename(local_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def download_models(base_path, xtts_base_model):
    os.makedirs(base_path, exist_ok=True)
    
    files_to_download = [
        ("https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth", "dvae.pth"),
        ("https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth", "mel_stats.pth"),
        (f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{xtts_base_model}/vocab.json", "vocab.json"),
        (f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{xtts_base_model}/model.pth", "model.pth"),
        (f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/{xtts_base_model}/config.json", "config.json"),
        ("https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/speakers_xtts.pth", "speakers_xtts.pth")
    ]
    
    for url, filename in files_to_download:
        local_path = os.path.join(base_path, filename)
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            download_file(url, local_path)
        else:
            print(f"{filename} already exists. Skipping download.")
        
        # Verify file
        if os.path.exists(local_path):
            print(f"File {filename} size: {os.path.getsize(local_path)} bytes")
            if filename.endswith('.json'):
                with open(local_path, 'r') as f:
                    print(f"First 100 characters of {filename}: {f.read(100)}")
        else:
            print(f"Error: {filename} does not exist after download attempt!")

def train_gpt(custom_model, version, language, num_epochs, batch_size, grad_acumm, 
              train_csv, eval_csv, output_path, sample_rate, model_name, max_audio_time,
              max_text_length):
    """
    Train the GPT model.
    """
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    OUT_PATH = os.path.join(output_path, "run", "training")
    CHECKPOINTS_OUT_PATH = os.path.join(Path.cwd(), "base_models", f"{version}")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    # # Initialize metrics tracking
    # metrics = TrainingMetrics(
        # model_name=model_name,
        # start_time=time.time(),
        # num_epochs=num_epochs,
        # gradient_accumulation=grad_acumm,
        # language=language,
        # sample_rate=sample_rate
    # )

    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=os.path.dirname(train_csv),
        meta_file_train=os.path.basename(train_csv),
        meta_file_val=os.path.basename(eval_csv),
        language=language,
    )
    DATASETS_CONFIG_LIST = [config_dataset]

    # Use local paths
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "dvae.pth")
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "mel_stats.pth")
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "vocab.json")
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "model.pth")
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "config.json")
    XTTS_SPEAKER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "speakers_xtts.pth")

    if custom_model and os.path.exists(custom_model) and custom_model.endswith('.pth'):
        XTTS_CHECKPOINT = custom_model

    # Calculate max lengths from max_audio_time and sample_rate
    max_wav_length = int((max_audio_time + 0.5) * sample_rate) # Convert seconds to samples
    max_conditioning_length = max_wav_length  # Same as max_wav_length for XTTS

    model_args = GPTArgs(
        max_conditioning_length=max_conditioning_length,
        min_conditioning_length=6615,
        max_wav_length=max_wav_length,
        max_text_length=max_text_length,  # Use the parameter directly
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
        debug_loading_failures=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=sample_rate,
        dvae_sample_rate=22050,
        output_sample_rate=24000
    )

    config = GPTTrainerConfig(
        epochs=num_epochs,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="GPT XTTS training",
        dashboard_logger=DASHBOARD_LOGGER,
        audio=audio_config,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        num_loader_workers=8 if language != "ja" else 0,
        print_step=50,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        optimizer="AdamW",
        optimizer_params={
            "betas": [0.9, 0.96],
            "eps": 1e-8,
            "weight_decay": 1e-2
        },
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
            "last_epoch": -1
        },
    )

    model = GPTTrainer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Update metrics with audio stats before training
    #metrics.calculate_audio_stats(train_samples)

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            grad_accum_steps=grad_acumm,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    # Train the model
    trainer.fit()

    # # Update metrics after training
    # metrics.end_time = time.time()
    
    # # Parse training logs
    # log_file = Path(OUT_PATH) / "log.txt"
    # if log_file.exists():
        # metrics.update_from_log(log_file)
    
    # # Save metrics
    # metrics.save(Path(output_path))

    del model, trainer, train_samples, eval_samples
    gc.collect()

    return XTTS_SPEAKER_FILE, XTTS_CONFIG_FILE, XTTS_CHECKPOINT, TOKENIZER_FILE, OUT_PATH

def optimize_model(out_path, base_model_path):
    run_dir = Path(out_path) / "run" / "training"
    best_model = None

    print(f"Searching for best model in: {run_dir}")

    for model_dir in run_dir.glob("*"):
        if model_dir.is_dir():
            print(f"Checking directory: {model_dir}")
            model_files = list(model_dir.glob("best_model_*.pth"))
            print(f"Found {len(model_files)} model files")
            if model_files:
                best_model = max(model_files, key=lambda x: int(x.stem.split('_')[-1]))
                print(f"Selected best model: {best_model}")
                break

    if best_model is None:
        return "Best model not found in training output", ""

    print(f"Best model found: {best_model}")

    # Load the checkpoint and remove unnecessary parts
    checkpoint = torch.load(best_model, map_location=torch.device("cpu"))
    if "optimizer" in checkpoint:
        del checkpoint["optimizer"]
    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    # Save the optimized model directly in out_path
    optimized_model = out_path / "model.pth"
    torch.save(checkpoint, optimized_model)

    # Copy config, speakers, and vocab files from base model
    files_to_copy = ["config.json", "speakers_xtts.pth", "vocab.json"]
    for file in files_to_copy:
        src = Path(base_model_path) / file
        dst = out_path / file
        if src.exists():
            shutil.copy(src, dst)
        else:
            print(f"Warning: {file} not found in base model path.")

    # Remove the run directory and its contents
    run_dir = Path(out_path) / "run"
    if run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except PermissionError:
            logging.warning(f"Unable to delete the run directory at {run_dir} due to a permission error. You may want to manually delete this directory to save disk space.")

    return f"Model optimized and saved at {optimized_model}!", str(optimized_model)

def copy_reference_samples(processed_dir, session_path, session_data):
    pandrator_voices_dir = Path("../Pandrator/tts_voices")
    if not pandrator_voices_dir.exists():
        print("Pandrator tts_voices directory does not exist. Skipping reference sample copying.")
        return

    # Get session name for file naming
    session_name = session_path.name

    # Get all wav files and their corresponding segments from session_data
    wav_files = []
    for wav_file in processed_dir.glob('*.wav'):
        # Find corresponding text in session_data
        segment = next((seg for seg in session_data['qualifying_segments'] 
                       if Path(seg['audio_file']).name == wav_file.name), None)
        if segment:
            wav_files.append({
                'path': wav_file,
                'size': wav_file.stat().st_size,
                'text': segment['text'],
                'duration': AudioSegment.from_wav(wav_file).duration_seconds
            })

    # Sort files by size in descending order
    wav_files.sort(key=lambda x: x['size'], reverse=True)

    # Select one random file from top 10%
    top_10_percent = wav_files[:max(1, len(wav_files) // 10)]
    random_long = random.choice(top_10_percent)
    
    # Copy the random long sample
    shutil.copy2(random_long['path'], pandrator_voices_dir / f"{session_name}.wav")

    # Get top 70% by length for speech rate calculation
    top_70_percent = wav_files[:int(len(wav_files) * 0.7)]
    
    # Calculate speech rate (characters per second) for each file
    for file in top_70_percent:
        file['speech_rate'] = len(file['text']) / file['duration']

    # Find the file with highest speech rate
    fastest = max(top_70_percent, key=lambda x: x['speech_rate'])
    
    # Copy the fastest sample
    shutil.copy2(fastest['path'], pandrator_voices_dir / f"{session_name}_fastest.wav")

    print(f"Copied reference samples to {pandrator_voices_dir}:")
    print(f"Long sample: {random_long['path'].name}")
    print(f"Fastest sample: {fastest['path'].name} (speech rate: {fastest['speech_rate']:.2f} chars/sec)")

def main():
    try:
        args = parse_arguments()
        session_path = create_session_folder(args.session)
        log_file = create_log_file(session_path)
        json_file = create_json_file(session_path)

        session_data = {'qualifying_segments': []}

        with open(json_file, 'w') as f:
            json.dump(session_data, f)

        # Process audio - this function now handles all audio preprocessing including
        # normalization, de-essing, denoising, and compression
        audio_sources_dir = process_audio(args.input, session_path, args)

        total_segments = 0
        for audio_file in audio_sources_dir.glob('*.wav'):
            json_file = transcribe_audio(audio_file, args, session_path)
            segments_count = parse_transcription(
                json_file, 
                audio_file, 
                audio_sources_dir / "processed", 
                session_data, 
                args.sample_method,
                args
            )
            total_segments += segments_count
            print(f"Total qualifying segments after processing {audio_file.name}: {segments_count}")

            # Save updated session_data to JSON file after each audio file
            with open(json_file, 'w') as f:
                json.dump(session_data, f)

        print(f"Total qualifying segments across all files: {total_segments}")

        train_csv, eval_csv = create_metadata_files(
            session_data, 
            session_path,
            args.training_proportion if hasattr(args, 'training_proportion') else 0.8  # Use new proportion if provided
        )

        if train_csv is None or eval_csv is None:
            print("Failed to create metadata files. Exiting.")
            return

        # Clear GPU memory before training
        print("\nClearing GPU memory before training...")
        if torch.cuda.is_available():
            # Clear CUDA cache
            torch.cuda.empty_cache()
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            # Force garbage collection
            gc.collect()
            # Clear cache again
            torch.cuda.empty_cache()
            
            # Print memory stats
            print("GPU Memory Status after clearing:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB\n")

        models_dir = session_path / "models"
        models_dir.mkdir(exist_ok=True)

        model_name = args.xtts_model_name or f"xtts_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_output_path = models_dir / model_name

        base_model_path = os.path.join(Path.cwd(), "base_models", args.xtts_base_model)
        download_models(base_model_path, args.xtts_base_model)

        speaker_file, config_file, checkpoint_file, tokenizer_file, training_output_path = train_gpt(
            custom_model="",
            version=args.xtts_base_model,
            language=args.source_language,
            num_epochs=args.epochs,
            batch_size=args.batch,
            grad_acumm=args.gradient,
            train_csv=str(train_csv),
            eval_csv=str(eval_csv),
            output_path=str(model_output_path),
            sample_rate=args.sample_rate,
            model_name=model_name,
            max_audio_time=args.max_audio_time,
            max_text_length=args.max_text_length
        )

        optimization_message, optimized_model_path = optimize_model(model_output_path, base_model_path)

        print(f"Training completed. Model saved at: {model_output_path}")
        print(optimization_message)
        print(f"Optimized model path: {optimized_model_path}")

        if optimized_model_path:  
            copy_reference_samples(audio_sources_dir / "processed", session_path, session_data)
            print("Reference samples copied successfully.")
        else:
            print("Warning: Optimized model path not found. Reference samples not copied.")

        return True  # Indicate successful completion

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("Training process completed successfully.")
            sys.exit(0)  # Exit with success code
        else:
            print("Training process failed.")
            sys.exit(1)  # Exit with error code
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)  # Exit with error code
