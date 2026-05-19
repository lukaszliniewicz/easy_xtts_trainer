from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from scipy import signal

from easy_xtts_trainer.config import DatasetRuntimeConfig


class AudioProcessor:
    def __init__(self, target_sr: int = 22050):
        self.target_sr = target_sr
        self.deepfilter_model = None
        self.df_state = None
        self._df_enhance = None
        self._df_init = None
        self._df_load_audio = None
        self._denoise_warning_emitted = False

        # Compression profiles optimized for different voice types
        self.compression_profiles: Dict = {
            "male": {
                "threshold": -18,
                "ratio": 3.0,
                "attack": 0.020,
                "release": 0.150,
                "knee_width": 6,
            },
            "female": {
                "threshold": -16,
                "ratio": 2.5,
                "attack": 0.015,
                "release": 0.120,
                "knee_width": 4,
            },
            "neutral": {
                "threshold": -17,
                "ratio": 2.75,
                "attack": 0.018,
                "release": 0.135,
                "knee_width": 5,
            },
        }

        # De-essing profiles
        self.dess_profiles: Dict = {
            "low": {
                "threshold": 0.15,
                "ratio": 2.5,
                "makeup_gain": 0.5,
                "reduction_factor": 0.3,
                "freq_range": (4500, 9000),
            },
            "high": {
                "threshold": 0.25,
                "ratio": 3.5,
                "makeup_gain": 0.4,
                "reduction_factor": 0.5,
                "freq_range": (5000, 10000),
            },
        }
        self.segment_window_ms = 200  # Window for finding cut points
        self.analysis_window_ms = 2  # Window for energy analysis

    def apply_fades(
        self,
        audio: np.ndarray,
        sr: int,
        fade_in_ms: Optional[int] = None,
        fade_out_ms: Optional[int] = None,
    ) -> np.ndarray:
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

        except Exception as exc:
            print(f"Error applying fades: {str(exc)}")
            return audio

    def remove_trailing_silence(
        self,
        audio: np.ndarray,
        sr: int,
        silence_threshold: float = -50.0,
        chunk_size: int = 10,
    ) -> np.ndarray:
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
                channels=1 if len(audio.shape) == 1 else audio.shape[1],
            )

            # Detect trailing silence by reversing the audio
            trailing_silence = detect_leading_silence(
                audio_segment.reverse(),
                silence_threshold=silence_threshold,
                chunk_size=chunk_size,
            )

            # Convert milliseconds to samples
            samples_to_remove = int((trailing_silence / 1000) * sr)

            # Trim the audio
            if samples_to_remove > 0:
                return audio[:-samples_to_remove]
            return audio

        except Exception as exc:
            print(f"Error in remove_trailing_silence: {str(exc)}")
            return audio

    def find_lowest_energy_point(
        self,
        audio: AudioSegment,
        start_time: float,
        end_time: float,
        is_start_cut: bool = True,
    ) -> float:
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
            max_window_ms = 100 if is_start_cut else 300

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

            for index in range(num_subwindows):
                start_idx = index * len(samples) // num_subwindows
                end_idx = (index + 1) * len(samples) // num_subwindows
                subwindow = samples[start_idx:end_idx]

                # Check for true silence
                silence_ratio = np.sum(np.abs(subwindow) < sample_silence_threshold) / len(subwindow)
                is_silent = silence_ratio > 0.95

                # Calculate RMS energy
                rms = np.sqrt(np.mean(subwindow**2))
                db = 20 * np.log10(max(rms, 1e-5))

                subwindow_start_time = start_time + (index * window_duration_ms / num_subwindows) / 1000
                subwindow_center = subwindow_start_time + (subwindow_size_ms / 2000)

                print(
                    f"Subwindow {index + 1}: {db:.1f}dB, "
                    f"silence_ratio: {silence_ratio:.2f}, center at {subwindow_center:.3f}s"
                )

                subwindow_data.append(
                    {
                        "index": index,
                        "db": db,
                        "center_time": subwindow_center,
                        "is_silent": is_silent,
                        "start_time": subwindow_start_time,
                        "end_time": subwindow_start_time + (subwindow_size_ms / 1000),
                        "samples": subwindow,
                    }
                )

            if not subwindow_data:
                return (start_time + end_time) / 2

            # First check for true silence
            silent_windows = [window for window in subwindow_data if window["is_silent"]]

            if silent_windows:
                # Find the silent window with lowest energy
                optimal_window = min(silent_windows, key=lambda item: item["db"])
                # Find the absolute lowest point within this window
                min_sample_idx = np.argmin(np.abs(optimal_window["samples"]))
                relative_time = min_sample_idx / len(optimal_window["samples"]) * subwindow_size_ms / 1000
                optimal_time = optimal_window["start_time"] + relative_time
                print(f"Found true silence at {optimal_time:.3f}s")
            else:
                # Find the window with lowest energy
                optimal_window = min(subwindow_data, key=lambda item: item["db"])
                # Find the absolute lowest point within this window
                min_sample_idx = np.argmin(np.abs(optimal_window["samples"]))
                relative_time = min_sample_idx / len(optimal_window["samples"]) * subwindow_size_ms / 1000
                optimal_time = optimal_window["start_time"] + relative_time
                print(f"Found lowest energy point ({optimal_window['db']:.1f}dB) at {optimal_time:.3f}s")

            final_time = max(start_time, min(end_time, optimal_time))
            if final_time != optimal_time:
                print(f"Adjusted time to {final_time:.3f}s to stay within bounds")

            return final_time

        except Exception as exc:
            print(f"Error in find_lowest_energy_point: {str(exc)}")
            return (start_time + end_time) / 2

    def find_optimal_cut_points(
        self,
        segments: List[Dict],
        audio: AudioSegment,
    ) -> List[Dict[str, Tuple[float, Optional[float], Optional[float]]]]:
        """
        Find optimal cut points between segments by analyzing the quietest point
        between contextual words and segments.
        """
        if not segments:
            return []

        cut_points = []
        audio_length_sec = len(audio) / 1000  # Convert to seconds

        # Define window sizes at the start of the method
        max_start_window = 0.15  # 100ms for start cuts
        max_end_window = 0.4  # 300ms for end cuts

        # Create debug log file in the same directory as the audio
        debug_log_path = Path(segments[0]["words"][0].get("audio_file", "debug")).parent / "cut_points_debug.log"

        with open(debug_log_path, "w", encoding="utf-8") as debug_log:
            debug_log.write("Cut Points Analysis Debug Log\n")
            debug_log.write("===========================\n")
            debug_log.write(f"Audio duration: {audio_length_sec:.3f}s\n")
            debug_log.write(f"Number of segments: {len(segments)}\n")
            debug_log.write(f"Max start window: {max_start_window * 1000:.0f}ms\n")
            debug_log.write(f"Max end window: {max_end_window * 1000:.0f}ms\n\n")

            for index in range(len(segments)):
                curr_segment = segments[index]
                curr_first_word = next((word for word in curr_segment["words"] if "start" in word), None)
                curr_last_word = next((word for word in reversed(curr_segment["words"]) if "end" in word), None)

                if not curr_first_word or not curr_last_word:
                    debug_log.write(f"\nSkipping segment {index} - Missing word timestamps\n")
                    continue

                # Log segment details
                debug_log.write(f"\n{'=' * 20} Segment {index} {'=' * 20}\n")
                debug_log.write(f"Text: {' '.join(word['word'] for word in curr_segment['words'])}\n")
                debug_log.write(f"Duration: {curr_last_word['end'] - curr_first_word['start']:.3f}s\n")
                debug_log.write(f"Words: {len(curr_segment['words'])}\n")
                debug_log.write("\nWord timings:\n")
                for word in curr_segment["words"]:
                    if "start" in word and "end" in word:
                        duration = word["end"] - word["start"]
                        debug_log.write(f"  '{word['word']}': {word['start']:.3f}s - {word['end']:.3f}s ({duration * 1000:.0f}ms)\n")

                # Context - previous word's end time and next word's start time
                prev_word_end = None
                next_word_start = None

                # Get previous segment info
                if index > 0:
                    prev_segment = segments[index - 1]
                    prev_last_word = next((word for word in reversed(prev_segment["words"]) if "end" in word), None)
                    if prev_last_word:
                        prev_word_end = prev_last_word["end"]
                        debug_log.write(f"\nPrevious context: '{prev_last_word['word']}' ending at {prev_word_end:.3f}s\n")
                else:
                    debug_log.write("\nFirst segment - no previous word\n")

                # Get next segment info
                if index < len(segments) - 1:
                    next_segment = segments[index + 1]
                    next_first_word = next((word for word in next_segment["words"] if "start" in word), None)
                    if next_first_word:
                        next_word_start = next_first_word["start"]
                        debug_log.write(f"Next context: '{next_first_word['word']}' starting at {next_word_start:.3f}s\n")
                else:
                    debug_log.write("Last segment - no next word\n")

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
                debug_log.write(
                    f"Start window: {start_search_start:.3f}s - "
                    f"{start_search_end:.3f}s ({(start_search_end - start_search_start) * 1000:.0f}ms)\n"
                )
                debug_log.write(
                    f"End window: {end_search_start:.3f}s - "
                    f"{end_search_end:.3f}s ({(end_search_end - end_search_start) * 1000:.0f}ms)\n"
                )

                # Find optimal points
                start_time = self.find_lowest_energy_point(audio, start_search_start, start_search_end, is_start_cut=True)
                end_time = self.find_lowest_energy_point(audio, end_search_start, end_search_end, is_start_cut=False)

                cut_points.append(
                    {
                        "start": start_time,
                        "end": end_time,
                        "prev_word_end": prev_word_end,
                        "next_word_start": next_word_start,
                    }
                )

                # Log final results
                debug_log.write("\nFinal cut points:\n")
                debug_log.write(f"Start cut at: {start_time:.3f}s\n")
                debug_log.write(f"  Distance from search start: {(start_time - start_search_start) * 1000:.0f}ms\n")
                debug_log.write(f"  Distance to first word: {(curr_first_word['start'] - start_time) * 1000:.0f}ms\n")
                if prev_word_end is not None:
                    debug_log.write(f"  Gap from previous word: {(start_time - prev_word_end) * 1000:.0f}ms\n")

                debug_log.write(f"End cut at: {end_time:.3f}s\n")
                debug_log.write(f"  Distance from search start: {(end_time - end_search_start) * 1000:.0f}ms\n")
                debug_log.write(f"  Distance from last word: {(end_time - curr_last_word['end']) * 1000:.0f}ms\n")
                if next_word_start is not None:
                    debug_log.write(f"  Gap to next word: {(next_word_start - end_time) * 1000:.0f}ms\n")

                debug_log.write(f"Final segment duration: {end_time - start_time:.3f}s\n")
                debug_log.write("=" * 50 + "\n")

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
            energy = np.array([np.sqrt(np.mean(window**2)) for window in windows])

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
            zcr_rates = [np.sum(np.abs(np.diff(np.signbit(window)))) / len(window) for window in zcr_windows]
            zcr_variance = np.var(zcr_rates)
            zcr_change = abs(zcr_rates[-1] - np.mean(zcr_rates[:-1])) if len(zcr_rates) > 1 else 0

            # Check for sudden amplitude drop
            amplitude_envelope = np.abs(tail)
            final_samples = amplitude_envelope[-int(0.01 * sr) :]  # Last 10ms
            sudden_drop = np.mean(final_samples) < 0.1 * np.mean(amplitude_envelope)

            # More sensitive thresholds
            is_abrupt = (
                (
                    energy_ratio > 0.5  # Energy doesn't drop enough
                    or end_slope > 0  # Energy increases at the end
                    or zcr_variance > 0.05  # Irregular zero crossings
                    or zcr_change > 0.3  # Sudden change in zero-crossing rate
                    or sudden_drop
                )
                and end_energy > 0.1  # Ensure there's significant energy at the end
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

        except Exception as exc:
            print(f"Error in check_for_abrupt_ending: {str(exc)}")
            print(f"Audio shape: {audio.shape if isinstance(audio, np.ndarray) else 'invalid'}")
            print(f"Audio type: {audio.dtype if isinstance(audio, np.ndarray) else 'invalid'}")
            return False

    def _calculate_zcr_variance(self, audio: np.ndarray) -> float:
        """Calculate zero-crossing rate variance with adaptive windowing."""
        window_size = len(audio) // 4
        zcr_groups = [
            np.sum(np.abs(np.diff(np.signbit(group)))) / len(group)
            for group in np.array_split(audio, max(4, len(audio) // window_size))
        ]
        return np.var(zcr_groups)

    def process_segments(
        self,
        input_path: str,
        segments: List[Dict],
        output_dir: Path,
        config: DatasetRuntimeConfig,
    ) -> List[Dict[str, str]]:
        try:
            # Load audio file
            audio = AudioSegment.from_wav(input_path)

            # Track statistics for final report
            total_segments = len(segments)
            discarded_segments = []

            # Apply negative offset to last words
            adjusted_segments = []
            for segment in segments:
                if segment["words"]:
                    # Create a deep copy of the segment
                    adjusted_segment = {
                        "words": segment["words"][:-1],  # All words except last
                        "text": segment["text"],
                    }

                    # Adjust the last word's end time
                    last_word = segment["words"][-1].copy()
                    if "end" in last_word:
                        base_offset = config.negative_offset_last_word / 1000
                        if any(last_word["word"].strip().endswith(punctuation) for punctuation in ".!?"):
                            offset = base_offset * 2
                        else:
                            offset = base_offset

                        last_word["end"] = max(last_word["start"], last_word["end"] - offset)
                    adjusted_segment["words"].append(last_word)
                    adjusted_segments.append(adjusted_segment)

            # Check gaps between segments if min-gap is specified
            valid_segments = []
            if config.min_gap:
                min_gap_seconds = config.min_gap / 1000  # Convert to seconds

                for index, segment in enumerate(adjusted_segments):
                    should_keep = True
                    segment_duration = len(" ".join(word["word"] for word in segment["words"]))

                    # Check gap with previous segment
                    if index > 0 and "start" in segment["words"][0] and "end" in adjusted_segments[index - 1]["words"][-1]:
                        prev_end = float(adjusted_segments[index - 1]["words"][-1]["end"])
                        curr_start = float(segment["words"][0]["start"])
                        if curr_start - prev_end < min_gap_seconds:
                            should_keep = False
                            discarded_segments.append(
                                {
                                    "index": index,
                                    "duration": float(segment["words"][-1]["end"]) - float(segment["words"][0]["start"]),
                                    "text": segment["text"],
                                    "reason": (
                                        "gap with previous segment "
                                        f"({(curr_start - prev_end) * 1000:.1f}ms) below minimum "
                                        f"({config.min_gap}ms)"
                                    ),
                                }
                            )
                            continue

                    # Check gap with next segment
                    if index < len(adjusted_segments) - 1 and "end" in segment["words"][-1] and "start" in adjusted_segments[index + 1]["words"][0]:
                        curr_end = float(segment["words"][-1]["end"])
                        next_start = float(adjusted_segments[index + 1]["words"][0]["start"])
                        if next_start - curr_end < min_gap_seconds:
                            should_keep = False
                            discarded_segments.append(
                                {
                                    "index": index,
                                    "duration": segment_duration,
                                    "text": segment["text"],
                                    "reason": (
                                        f"gap with next segment ({(next_start - curr_end) * 1000:.1f}ms) "
                                        f"below minimum ({config.min_gap}ms)"
                                    ),
                                }
                            )
                            continue

                    if should_keep:
                        valid_segments.append(segment)
            else:
                valid_segments = adjusted_segments

            cut_points = self.find_optimal_cut_points(valid_segments, audio)
            processed_segments = []

            print(f"\nProcessing {len(valid_segments)} segments from {Path(input_path).name}...")

            # Process each segment
            for index, cut_point in enumerate(cut_points):
                start_ms = int(cut_point["start"] * 1000)
                end_ms = int(cut_point["end"] * 1000)

                # Extract segment
                segment_audio = audio[start_ms:end_ms]
                segment_duration = len(segment_audio) / 1000.0  # duration in seconds
                segment_text = " ".join(word["word"] for word in valid_segments[index]["words"]).strip()

                # Check for abrupt ending if flag is set
                if config.discard_abrupt:
                    samples = np.array(segment_audio.get_array_of_samples(), dtype=np.float32)
                    samples = samples / max(np.max(np.abs(samples)), 1)  # Normalize to [-1, 1]
                    if self.check_for_abrupt_ending(samples, segment_audio.frame_rate):
                        discarded_segments.append(
                            {
                                "index": index,
                                "duration": segment_duration,
                                "text": segment_text,
                                "reason": "abrupt ending",
                            }
                        )
                        print(f"\nWarning: Discarding segment {index}:")
                        print(f"   Duration: {segment_duration:.2f}s")
                        print(f"   Text: \"{segment_text}\"")
                        continue

                # Generate output filename
                output_path = output_dir / f"{Path(input_path).stem}_segment_{index}.wav"

                # Export raw segment
                segment_audio.export(str(output_path), format="wav")

                # Apply audio processing chain
                if any(
                    [
                        config.normalize is not None,
                        config.dess,
                        config.denoise,
                        config.compress,
                        config.fade_in > 0,
                        config.fade_out > 0,
                        config.trim,
                    ]
                ):
                    success = self.process_audio(
                        str(output_path),
                        str(output_path),
                        normalize_target=-float(config.normalize) if config.normalize else None,
                        dess_profile="high" if config.dess else None,
                        denoise=config.denoise,
                        compress_profile=config.compress if config.compress else None,
                        fade_in_ms=config.fade_in,
                        fade_out_ms=config.fade_out,
                        trim=config.trim,
                    )

                    if not success:
                        discarded_segments.append(
                            {
                                "index": index,
                                "duration": segment_duration,
                                "text": segment_text,
                                "reason": "processing failed",
                            }
                        )
                        print(f"\nWarning: Discarding segment {index}:")
                        print(f"   Duration: {segment_duration:.2f}s")
                        print(f"   Text: \"{segment_text}\"")
                        print("   Reason: Audio processing failed")
                        continue

                # Store segment information
                processed_segments.append(
                    {
                        "audio_file": str(output_path),
                        "text": segment_text,
                        "speaker_name": "001",
                    }
                )

            # Print final statistics
            print("\nSegment Processing Summary:")
            print(f"Total segments: {total_segments}")
            print(f"Successfully processed: {len(processed_segments)}")
            if discarded_segments:
                print(f"Discarded segments: {len(discarded_segments)}")
                print("\nDiscarded segments details:")
                for discarded in discarded_segments:
                    print(f"\nSegment {discarded['index']}:")
                    print(f"Duration: {discarded['duration']:.2f}s")
                    print(f"Text: \"{discarded['text']}\"")
                    print(f"Reason: {discarded['reason']}")

            return processed_segments

        except Exception as exc:
            print(f"Error processing segments: {str(exc)}")
            return []

    def _warn_denoise_unavailable(self, reason: str) -> None:
        if not self._denoise_warning_emitted:
            print(f"Warning: --denoise requested but {reason}. Continuing without denoise.")
            self._denoise_warning_emitted = True

    def _import_deepfilter(self) -> bool:
        if self._df_enhance and self._df_init and self._df_load_audio:
            return True

        try:
            from df.enhance import enhance as df_enhance
            from df.enhance import init_df as df_init
            from df.enhance import load_audio as df_load_audio
        except Exception as exc:
            self._warn_denoise_unavailable(f"DeepFilterNet could not be imported ({exc})")
            return False

        self._df_enhance = df_enhance
        self._df_init = df_init
        self._df_load_audio = df_load_audio
        return True

    def _init_deepfilter(self) -> bool:
        """Initialize DeepFilterNet model lazily."""
        if not self._import_deepfilter():
            return False

        if self.deepfilter_model is None:
            try:
                self.deepfilter_model, self.df_state, _ = self._df_init()
            except Exception as exc:
                self._warn_denoise_unavailable(f"DeepFilterNet failed to initialize ({exc})")
                return False

        return self.df_state is not None

    def process_audio(
        self,
        input_path: str,
        output_path: str,
        normalize_target: Optional[float] = None,
        dess_profile: Optional[str] = None,
        denoise: bool = False,
        compress_profile: Optional[str] = None,
        fade_in_ms: Optional[int] = None,
        fade_out_ms: Optional[int] = None,
        trim: bool = False,
    ) -> bool:
        use_denoise = False
        try:
            # Load and pre-process audio
            use_denoise = denoise and self._init_deepfilter()

            if use_denoise:
                # Use DeepFilterNet's own loading and processing functions
                audio, _ = self._df_load_audio(input_path, sr=self.df_state.sr())
                # Denoise the audio
                audio = self._df_enhance(self.deepfilter_model, self.df_state, audio)
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
            except Exception as exc:
                print(f"Error saving audio file: {str(exc)}")
                return False

            return True

        except Exception as exc:
            print(f"Error processing audio: {str(exc)}")
            print(f"Audio shape: {audio.shape if 'audio' in locals() else 'not loaded'}")
            print(f"Audio type: {type(audio) if 'audio' in locals() else 'not loaded'}")
            print(f"Sample rate: {sr if 'sr' in locals() else 'not loaded'}")
            return False
        finally:
            # Clear any GPU memory if used
            if use_denoise and self.deepfilter_model is not None:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _normalize(self, audio: np.ndarray, target_lufs: float = -16.0) -> np.ndarray:
        """Normalize audio to target LUFS with true peak limiting."""
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

    def _compress(self, audio: np.ndarray, profile: Literal["male", "female", "neutral"]) -> np.ndarray:
        """Dynamic range compression with voice-optimized profiles."""
        params = self.compression_profiles[profile]

        # Convert threshold to linear
        threshold_linear = 10 ** (params["threshold"] / 20)
        knee_lower = threshold_linear * (10 ** (-params["knee_width"] / 40))
        knee_upper = threshold_linear * (10 ** (params["knee_width"] / 40))

        # Calculate gain reduction
        gain_reduction = np.zeros_like(audio)

        # Below knee
        mask_below = np.abs(audio) <= knee_lower
        gain_reduction[mask_below] = 0

        # Above knee
        mask_above = np.abs(audio) >= knee_upper
        gain_above = (1 - 1 / params["ratio"]) * (20 * np.log10(np.abs(audio[mask_above])) - params["threshold"])
        gain_reduction[mask_above] = gain_above

        # In knee
        mask_knee = ~mask_below & ~mask_above
        knee_curve = (1 - 1 / params["ratio"]) * (
            (
                20 * np.log10(np.abs(audio[mask_knee]))
                - params["threshold"]
                + params["knee_width"] / 2
            )
            ** 2
            / (2 * params["knee_width"])
        )
        gain_reduction[mask_knee] = knee_curve

        # Apply attack/release envelope
        attack_coef = np.exp(-1 / (params["attack"] * self.target_sr))
        release_coef = np.exp(-1 / (params["release"] * self.target_sr))

        gain_reduction_smoothed = np.zeros_like(gain_reduction)
        for index in range(1, len(gain_reduction)):
            if gain_reduction[index] <= gain_reduction_smoothed[index - 1]:
                # Attack phase
                gain_reduction_smoothed[index] = (
                    attack_coef * gain_reduction_smoothed[index - 1]
                    + (1 - attack_coef) * gain_reduction[index]
                )
            else:
                # Release phase
                gain_reduction_smoothed[index] = (
                    release_coef * gain_reduction_smoothed[index - 1]
                    + (1 - release_coef) * gain_reduction[index]
                )

        # Convert gain reduction to linear domain and apply
        gain_reduction_linear = 10 ** (gain_reduction_smoothed / 20)
        return audio * gain_reduction_linear

    def _deess(self, audio: np.ndarray, profile: Literal["low", "high"]) -> np.ndarray:
        """Enhanced de-essing with two profiles."""
        params = self.dess_profiles[profile]

        # Create bandpass filter for sibilance detection
        nyquist = self.target_sr / 2
        low_freq = params["freq_range"][0] / nyquist
        high_freq = params["freq_range"][1] / nyquist
        b, a = signal.butter(4, [low_freq, high_freq], btype="band")

        # Extract and process sibilance
        sibilants = signal.filtfilt(b, a, audio)

        # Calculate adaptive threshold
        rms = np.sqrt(np.mean(sibilants**2))
        adaptive_threshold = params["threshold"] * rms

        # Apply compression to sibilants
        mask = np.abs(sibilants) > adaptive_threshold
        sibilants[mask] = (
            adaptive_threshold + (np.abs(sibilants[mask]) - adaptive_threshold) / params["ratio"]
        ) * np.sign(sibilants[mask])

        # Apply makeup gain
        sibilants *= params["makeup_gain"]

        # Smooth transitions
        env_b, env_a = signal.butter(2, 150 / nyquist, btype="low")
        smoothed_sibilants = signal.filtfilt(env_b, env_a, sibilants)

        # Mix processed sibilants back with original
        return audio - smoothed_sibilants * params["reduction_factor"]
