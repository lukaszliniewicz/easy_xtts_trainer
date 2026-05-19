from __future__ import annotations


def is_abbreviation(text: str, pos: int) -> bool:
    """Check if punctuation is part of a known abbreviation."""
    common_abbrev = [
        "Mr.",
        "Mrs.",
        "Ms.",
        "Dr.",
        "Prof.",
        "Sr.",
        "Jr.",
        "et.",
        "al.",
        "etc.",
        "e.g.",
        "i.e.",
        "vs.",
        "Ph.D.",
        "M.D.",
        "B.A.",
        "M.A.",
        "a.m.",
        "p.m.",
        "U.S.",
        "U.K.",
        "St.",
    ]

    start = max(0, pos - 5)
    text_slice = text[start : pos + 1].lower()
    return any(abbr.lower() in text_slice for abbr in common_abbrev)


def find_real_sentence_end(text: str, pos: int) -> int:
    """Determine if a punctuation mark is a real sentence ending."""
    if pos >= len(text):
        return pos

    stack: list[str] = []
    i = pos + 1

    while i < len(text):
        if text[i] in '"\'(':
            stack.append(text[i])
        elif text[i] in '"\')':
            if stack and (
                (stack[-1] == '"' and text[i] == '"')
                or (stack[-1] == "'" and text[i] == "'")
                or (stack[-1] == "(" and text[i] == ")")
            ):
                stack.pop()
        elif text[i] not in " \t\n" and not stack:
            break
        i += 1

    return i - 1
