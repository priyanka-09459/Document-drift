"""
updater.py
----------
Locates and updates stale documentation sections in Markdown files with LLM-generated fixes.
"""

import os
import re
from typing import List, Tuple


def _find_section_bounds(lines: List[str], heading: str) -> Tuple[int, int]:
    """
    Find start and end line index for a section by heading name.
    Returns (start_content_index, end_section_index).
    """
    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")
    target_level = None
    start_idx = -1
    end_idx = len(lines)

    for i, line in enumerate(lines):
        m = heading_pattern.match(line)
        if m:
            level = len(m.group(1))
            h_text = m.group(2).strip()
            if start_idx == -1:
                if h_text.lower() == heading.lower() or heading.lower() in h_text.lower():
                    target_level = level
                    start_idx = i + 1  # Content starts after heading line
            else:
                if level <= target_level:
                    end_idx = i
                    break

    return start_idx, end_idx


def update_doc_file(
    file_path: str,
    heading: str,
    new_doc_text: str,
    interactive: bool = False,
) -> bool:
    """
    Updates a specific section in a Markdown file under the given heading.
    """
    if not os.path.exists(file_path):
        print(f"[docdrift] Warning: File {file_path} not found.")
        return False

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    start_idx, end_idx = _find_section_bounds(lines, heading)
    if start_idx == -1:
        print(f"[docdrift] Warning: Could not find heading '{heading}' in {file_path}")
        return False

    formatted_new_text = new_doc_text.strip() + "\n\n"

    if interactive:
        print(f"\nProposed update for {file_path} [Section: {heading}]:")
        print("-" * 50)
        print(formatted_new_text)
        print("-" * 50)
        resp = input("Apply this update? [y/N]: ").strip().lower()
        if resp != "y":
            print("Skipped.")
            return False

    new_lines = lines[:start_idx] + [formatted_new_text] + lines[end_idx:]

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"[docdrift] Updated {file_path} -> section '{heading}'")
    return True


def apply_doc_updates(
    confirmed_items: List[Tuple],
    repo_root: str,
    interactive: bool = False,
) -> int:
    """
    Applies LLM verdicts to documentation files.
    confirmed_items: List of (DriftFlag, LLMVerdict)
    """
    applied_count = 0
    for flag, verdict in confirmed_items:
        if not verdict.suggested_doc_text:
            continue

        mapped = flag.mapped_doc
        if "#" not in mapped:
            continue

        rel_path, heading = mapped.split("#", 1)
        rel_path = rel_path.lstrip("\\/ ")
        full_path = os.path.join(repo_root, rel_path)

        if update_doc_file(full_path, heading, verdict.suggested_doc_text, interactive=interactive):
            applied_count += 1

    return applied_count
