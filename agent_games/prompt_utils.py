"""
Shared Prompt Utilities

Common functions for handling agent system prompts, including:
- Protected section detection and filtering
- Line number translation (visible <-> actual)
- Safety filtering for prompt modifications
- Prompt validation and cleanup

Used by: Tribal Council, Self-Reflection, and other prompt-editing systems.
"""

import re
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# Protected Section Handling
# ============================================================================

def get_protected_line_ranges(prompt: str) -> List[Tuple[int, int]]:
    """
    Find all protected sections in a prompt and return their line ranges.
    Protected sections are marked with:
        === PROTECTED: ... ===
        ... content ...
        === END PROTECTED ===

    Returns list of (start_line, end_line) tuples (1-indexed, inclusive).
    """
    lines = prompt.split('\n')
    ranges = []

    start_pattern = re.compile(r'^===\s*PROTECTED:', re.IGNORECASE)
    end_pattern = re.compile(r'^===\s*END\s*PROTECTED\s*===', re.IGNORECASE)

    current_start = None
    for i, line in enumerate(lines):
        line_num = i + 1  # 1-indexed
        if start_pattern.match(line.strip()):
            current_start = line_num
        elif end_pattern.match(line.strip()) and current_start is not None:
            ranges.append((current_start, line_num))
            current_start = None

    # Handle unclosed protected section (protect to end)
    if current_start is not None:
        ranges.append((current_start, len(lines)))

    return ranges


def get_visible_prompt(prompt: str) -> str:
    """
    Return prompt with protected sections replaced by a placeholder.
    The placeholder takes one line to maintain rough structure awareness.
    """
    ranges = get_protected_line_ranges(prompt)
    if not ranges:
        return prompt

    lines = prompt.split('\n')
    result_lines = []

    # Build set of protected line numbers
    protected_lines = set()
    for start, end in ranges:
        for line_num in range(start, end + 1):
            protected_lines.add(line_num)

    in_protected = False
    for i, line in enumerate(lines):
        line_num = i + 1
        if line_num in protected_lines:
            if not in_protected:
                result_lines.append("[PROTECTED ABILITY - HIDDEN]")
                in_protected = True
            # Skip protected lines
        else:
            in_protected = False
            result_lines.append(line)

    return '\n'.join(result_lines)


def translate_visible_to_actual_line(prompt: str, visible_line_num: int) -> Optional[int]:
    """
    Translate a line number from the visible (filtered) prompt to the actual prompt.
    Returns None if the visible line maps to a protected placeholder.
    """
    ranges = get_protected_line_ranges(prompt)
    if not ranges:
        return visible_line_num

    lines = prompt.split('\n')

    # Build set of protected line numbers
    protected_lines = set()
    for start, end in ranges:
        for line_num in range(start, end + 1):
            protected_lines.add(line_num)

    # Walk through actual lines, counting visible lines
    visible_count = 0
    in_protected = False

    for i, line in enumerate(lines):
        line_num = i + 1

        if line_num in protected_lines:
            if not in_protected:
                # This is the placeholder line
                visible_count += 1
                if visible_count == visible_line_num:
                    # User is targeting the protected placeholder - reject
                    return None
                in_protected = True
        else:
            in_protected = False
            visible_count += 1
            if visible_count == visible_line_num:
                return line_num

    return None  # Line number out of range


def get_numbered_visible_prompt(prompt: str) -> Tuple[str, int]:
    """
    Return a numbered version of the visible prompt for display.

    Returns:
        Tuple of (numbered_prompt_string, line_count)
    """
    visible = get_visible_prompt(prompt)
    lines = visible.split('\n')
    numbered = '\n'.join([f"{i+1}: {line}" for i, line in enumerate(lines)])
    return numbered, len(lines)


# ============================================================================
# Safety Filtering
# ============================================================================

# Blocked terms that indicate non-consensual/predatory content
BLOCKED_PATTERNS = [
    "without consent", "non-consensual", "nonconsensual",
    "force yourself", "force them", "force her", "force him",
    "ignore consent", "don't need consent", "doesn't need consent",
    "take what you want", "whether they want", "whether she wants", "whether he wants",
    "unconscious", "passed out", "asleep",  # in sexual context
    "drugged", "roofie", "spike",
    "rape", "molest", "assault",
    "predator", "predatory",
    "child", "minor", "underage", "kid",
    "coerce", "manipulate into sex",
]


def is_content_safe(content: str) -> bool:
    """
    Check if proposed content contains harmful patterns.
    Returns True if safe, False if blocked.
    """
    if not content:
        return True  # Empty content is safe (deletions)

    content_lower = content.lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern in content_lower:
            return False

    return True


# ============================================================================
# Prompt Modification
# ============================================================================

@dataclass
class PromptEdit:
    """Represents a proposed edit to a prompt."""
    action: str  # "add", "delete", "change"
    line_number: Optional[int] = None  # For delete/change
    new_content: Optional[str] = None  # For add/change
    reason: str = ""


def apply_prompt_edit(prompt: str, edit: PromptEdit) -> Optional[str]:
    """
    Apply an edit to a prompt.

    Args:
        prompt: The original prompt
        edit: The edit to apply

    Returns:
        New prompt string, or None if edit was rejected
    """
    # Safety check
    if edit.new_content and not is_content_safe(edit.new_content):
        logger.warning(f"[PromptUtils] Rejected unsafe content in edit")
        return None

    lines = prompt.split('\n')

    try:
        if edit.action == "add":
            if edit.new_content:
                lines.append(edit.new_content)

        elif edit.action == "delete":
            if edit.line_number:
                # Translate from visible line number to actual line number
                actual_line = translate_visible_to_actual_line(prompt, edit.line_number)
                if actual_line is None:
                    logger.warning(f"[PromptUtils] Rejected delete of protected/invalid line {edit.line_number}")
                    return None
                if 0 < actual_line <= len(lines):
                    del lines[actual_line - 1]

        elif edit.action == "change":
            if edit.line_number and edit.new_content:
                # Translate from visible line number to actual line number
                actual_line = translate_visible_to_actual_line(prompt, edit.line_number)
                if actual_line is None:
                    logger.warning(f"[PromptUtils] Rejected change of protected/invalid line {edit.line_number}")
                    return None
                if 0 < actual_line <= len(lines):
                    lines[actual_line - 1] = edit.new_content

        return '\n'.join(lines)

    except Exception as e:
        logger.error(f"[PromptUtils] Error applying edit: {e}")
        return None


# ============================================================================
# Prompt Validation & Cleanup
# ============================================================================

def validate_and_cleanup_prompt(prompt: str) -> Tuple[str, List[str]]:
    """
    Validate prompt formatting and clean up common issues.

    Returns:
        Tuple of (cleaned_prompt, list_of_changes_made)
    """
    changes = []
    cleaned = prompt

    # 1. Remove excessive blank lines (more than 2 consecutive)
    original_len = len(cleaned)
    cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)
    if len(cleaned) != original_len:
        changes.append("Reduced excessive blank lines")

    # 2. Remove trailing whitespace on lines
    lines = cleaned.split('\n')
    stripped_lines = [line.rstrip() for line in lines]
    if lines != stripped_lines:
        changes.append("Removed trailing whitespace")
        cleaned = '\n'.join(stripped_lines)

    # 3. Ensure prompt doesn't start or end with excessive whitespace
    original = cleaned
    cleaned = cleaned.strip()
    if cleaned != original:
        changes.append("Trimmed leading/trailing whitespace")

    # 4. Check for and remove any null bytes or control characters (except newlines/tabs)
    original = cleaned
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
    if cleaned != original:
        changes.append("Removed control characters")

    # 5. Normalize line endings (CRLF -> LF)
    original = cleaned
    cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
    if cleaned != original:
        changes.append("Normalized line endings")

    # 6. Check for broken protected section markers and warn (don't fix)
    start_count = len(re.findall(r'===\s*PROTECTED:', cleaned, re.IGNORECASE))
    end_count = len(re.findall(r'===\s*END\s*PROTECTED\s*===', cleaned, re.IGNORECASE))
    if start_count != end_count:
        changes.append(f"WARNING: Mismatched protected sections ({start_count} starts, {end_count} ends)")

    return cleaned, changes
