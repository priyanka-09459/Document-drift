"""
drift_detector.py
------------------
Compares two snapshots of a repo (e.g. base branch vs PR branch, or two
commits) and flags documentation that likely needs updating.

Logic:
  1. Parse symbols + doc chunks at both snapshots.
  2. Diff symbols by qualified_name: find which changed (signature or body_hash).
  3. For each changed symbol, find its mapped doc chunk(s) (old snapshot).
  4. If none of those doc chunks changed text between snapshots -> flag as DRIFT.
  5. If a mapped doc chunk WAS edited -> considered handled, not flagged.
  6. Symbols with no confident doc mapping are reported separately (coverage gap),
     not as drift (nothing to compare against).

Supports:
  - Git ref comparison (HEAD~1..HEAD, origin/main..feature-branch, etc.)
  - Directory comparison (two local directories)
  - Scoped scanning: only parses files that actually changed between refs
  - Broad doc discovery: README.md, CONTRIBUTING.md, CHANGELOG.md, docs/, doc/, etc.
"""

import subprocess
import tempfile
import os
import shutil
import difflib
import tarfile
from dataclasses import dataclass
from typing import Optional

from .ast_parser import parse_repo, parse_file, Symbol
from .mapper import DocCodeMapper, collect_doc_chunks, DocChunk


@dataclass
class DriftFlag:
    symbol_name: str
    file_path: str
    reason: str
    old_signature: str
    new_signature: str
    mapped_doc: str  # "path#heading" or "(none)"
    confidence: float
    doc_text: str = ""
    code_diff: str = ""


def _resolve_repo_path(repo_path: str) -> str:
    """Resolve to the actual git repo root using `git rev-parse --show-toplevel`."""
    abs_path = os.path.abspath(repo_path)
    try:
        result = subprocess.run(
            ["git", "-C", abs_path, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return abs_path


def _validate_git_ref(repo_path: str, ref: str) -> bool:
    """Check if a git ref exists in the repository."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "--verify", ref],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _get_changed_files(repo_path: str, base_ref: str, head_ref: str) -> list[str]:
    """Get list of files changed between two git refs using `git diff --name-only`."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "diff", "--name-only", f"{base_ref}..{head_ref}"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except Exception:
        pass
    return []


def _get_commit_count(repo_path: str) -> int:
    """Return the number of commits in the repo."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "rev-list", "--count", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0


def _snapshot_from_git(repo_path: str, ref: str, dest_dir: str):
    """Checkout `ref` of `repo_path` into a clean temp dir using git archive and tarfile."""
    os.makedirs(dest_dir, exist_ok=True)
    proc = subprocess.Popen(
        ["git", "-C", repo_path, "archive", "--format=tar", ref],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|*") as tar:
            tar.extractall(dest_dir)
    except tarfile.ReadError as e:
        proc.wait()
        err = proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Failed to extract git archive for ref '{ref}' in '{repo_path}'. "
            f"Make sure the ref exists and has commits. Git error: {err}"
        )
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"git archive failed for ref '{ref}': {err}")


def _normalize_path(path: str) -> str:
    """Normalize path separators to forward slashes for consistent comparison."""
    return path.replace("\\", "/").strip("/")


def _index_symbols(symbols: list[Symbol]) -> dict:
    return {s.qualified_name: s for s in symbols}


def _parse_changed_py_files(repo_dir: str, changed_files: list[str]) -> list[Symbol]:
    """Parse only the Python files that actually changed."""
    symbols = []
    for rel_file in changed_files:
        if not rel_file.endswith(".py"):
            continue
        full_path = os.path.join(repo_dir, rel_file)
        if os.path.isfile(full_path):
            try:
                symbols.extend(parse_file(full_path))
            except Exception as e:
                print(f"[docdrift] Warning: Could not parse {rel_file}: {e}")
    return symbols


def detect_drift_between_dirs(old_dir: str, new_dir: str, similarity_threshold: float = 0.12,
                               changed_files: Optional[list[str]] = None):
    """
    Core comparison logic, operating on two plain directories (already
    checked out). Returns (drift_flags, coverage_gaps).

    If `changed_files` is provided, only parses those files for symbols
    (performance optimization for large repos). Documentation files are
    always fully scanned.
    """
    # Parse symbols: scoped to changed files if available, else full repo scan
    if changed_files:
        py_changed = [f for f in changed_files if f.endswith(".py")]
        if not py_changed:
            return [], []  # No Python files changed, no drift possible
        old_symbols = _parse_changed_py_files(old_dir, py_changed)
        new_symbols = _parse_changed_py_files(new_dir, py_changed)
    else:
        old_symbols = parse_repo(old_dir)
        new_symbols = parse_repo(new_dir)

    # Always scan all docs (they're usually few files)
    old_docs = collect_doc_chunks(old_dir)

    if not old_docs:
        print("[docdrift] Warning: No documentation files found (README.md, docs/*.md, etc.).")
        # Still report coverage gaps for changed symbols
        coverage_gaps = [s.qualified_name for s in new_symbols]
        return [], coverage_gaps

    mapper = DocCodeMapper(similarity_threshold=similarity_threshold)
    mapper.fit(old_symbols, old_docs)

    old_index = _index_symbols(old_symbols)
    new_index = _index_symbols(new_symbols)

    old_doc_text_by_loc = {
        (_normalize_path(c.doc_path.replace(old_dir, "")), c.heading): c.text
        for c in old_docs
    }
    new_docs = collect_doc_chunks(new_dir)
    new_doc_text_by_loc = {
        (_normalize_path(c.doc_path.replace(new_dir, "")), c.heading): c.text
        for c in new_docs
    }

    drift_flags: list[DriftFlag] = []
    coverage_gaps: list[str] = []

    for qname, new_sym in new_index.items():
        old_sym = old_index.get(qname)
        if old_sym is None:
            continue  # new symbol entirely; not a drift case (no prior doc to go stale)

        changed = (old_sym.signature != new_sym.signature) or (old_sym.body_hash != new_sym.body_hash)
        if not changed:
            continue

        reason = []
        if old_sym.signature != new_sym.signature:
            if getattr(old_sym, "kind", "") == "constant":
                reason.append("constant value or type changed")
            else:
                reason.append("signature or type annotations changed")
        if old_sym.body_hash != new_sym.body_hash:
            reason.append("implementation / body logic changed")

        mappings = mapper.matches_for_symbol(old_sym, top_k=1)
        if not mappings:
            coverage_gaps.append(qname)
            continue

        top = mappings[0]
        loc_key = (_normalize_path(top.doc_chunk.doc_path.replace(old_dir, "")), top.doc_chunk.heading)
        old_text = old_doc_text_by_loc.get(loc_key)
        new_text = new_doc_text_by_loc.get(loc_key)

        doc_unchanged = (old_text is not None) and (new_text is not None) and (old_text.strip() == new_text.strip())
        doc_removed = (old_text is not None) and (new_text is None)

        if doc_unchanged or doc_removed:
            symbol_diff = ""
            try:
                if os.path.isfile(old_sym.file_path) and os.path.isfile(new_sym.file_path):
                    with open(old_sym.file_path, "r", encoding="utf-8", errors="replace") as f:
                        old_lines = f.readlines()[old_sym.lineno - 1 : (old_sym.end_lineno or old_sym.lineno)]
                    with open(new_sym.file_path, "r", encoding="utf-8", errors="replace") as f:
                        new_lines = f.readlines()[new_sym.lineno - 1 : (new_sym.end_lineno or new_sym.lineno)]
                    diff = list(difflib.unified_diff(
                        old_lines, new_lines,
                        fromfile=f"old/{old_sym.qualified_name}",
                        tofile=f"new/{new_sym.qualified_name}",
                    ))
                    symbol_diff = "".join(diff)
            except Exception:
                symbol_diff = ""

            if not symbol_diff:
                symbol_diff = f"--- old/{old_sym.qualified_name}\n+++ new/{new_sym.qualified_name}\n- {old_sym.signature}\n+ {new_sym.signature}"

            drift_flags.append(
                DriftFlag(
                    symbol_name=qname,
                    file_path=_normalize_path(new_sym.file_path.replace(new_dir, "")),
                    reason=", ".join(reason) + (" (doc section removed)" if doc_removed else " but doc untouched"),
                    old_signature=old_sym.signature,
                    new_signature=new_sym.signature,
                    mapped_doc=f"{_normalize_path(top.doc_chunk.doc_path.replace(old_dir, ''))}#{top.doc_chunk.heading}",
                    confidence=top.score,
                    doc_text=old_text or "",
                    code_diff=symbol_diff,
                )
            )

    # Check for deleted/removed symbols (zombie documentation)
    for qname, old_sym in old_index.items():
        if qname not in new_index:
            mappings = mapper.matches_for_symbol(old_sym, top_k=1)
            if not mappings:
                continue
            top = mappings[0]
            loc_key = (_normalize_path(top.doc_chunk.doc_path.replace(old_dir, "")), top.doc_chunk.heading)
            new_text = new_doc_text_by_loc.get(loc_key)
            if new_text is not None and new_text.strip():
                # Documentation still exists in new snapshot for a symbol deleted in code
                drift_flags.append(
                    DriftFlag(
                        symbol_name=qname,
                        file_path=_normalize_path(old_sym.file_path.replace(old_dir, "")),
                        reason="symbol deleted from code, but documentation section remains (zombie documentation)",
                        old_signature=old_sym.signature,
                        new_signature="(deleted)",
                        mapped_doc=f"{_normalize_path(top.doc_chunk.doc_path.replace(old_dir, ''))}#{top.doc_chunk.heading}",
                        confidence=top.score,
                        doc_text=new_text,
                        code_diff=f"--- old/{old_sym.qualified_name}\n+++ /dev/null\n- {old_sym.signature}",
                    )
                )

    return drift_flags, coverage_gaps


def detect_drift_git(repo_path: str, base_ref: str = "HEAD~1", head_ref: str = "HEAD",
                      similarity_threshold: float = 0.12):
    """
    Convenience wrapper: diff two git refs of the same repo.

    Handles:
      - Resolves repo path to git root automatically
      - Validates refs exist before proceeding
      - Falls back to initial commit comparison if HEAD~1 doesn't exist
      - Scopes file parsing to only changed files for performance
    """
    # Resolve the actual git repo root
    repo_path = _resolve_repo_path(repo_path)

    # Validate the repo has commits
    commit_count = _get_commit_count(repo_path)
    if commit_count == 0:
        print("[docdrift] Error: Repository has no commits yet. Commit your code first.")
        return [], []

    # Handle HEAD~1 on repos with only 1 commit
    if base_ref == "HEAD~1" and commit_count == 1:
        print("[docdrift] Info: Only one commit exists. Comparing against empty tree.")
        base_ref = _get_empty_tree_sha()

    # Validate refs
    if not _validate_git_ref(repo_path, base_ref) and base_ref != _get_empty_tree_sha():
        print(f"[docdrift] Error: Base ref '{base_ref}' does not exist in the repository.")
        print(f"  Available branches/tags can be listed with: git -C {repo_path} branch -a")
        return [], []

    if not _validate_git_ref(repo_path, head_ref):
        print(f"[docdrift] Error: Head ref '{head_ref}' does not exist in the repository.")
        return [], []

    # Get list of changed files for scoped parsing
    changed_files = _get_changed_files(repo_path, base_ref, head_ref)
    if not changed_files:
        print("[docdrift] No files changed between the two refs.")
        return [], []

    print(f"[docdrift] Scanning {len(changed_files)} changed file(s) between {base_ref}..{head_ref}")
    for f in changed_files:
        print(f"  - {f}")

    with tempfile.TemporaryDirectory() as tmp:
        old_dir = os.path.join(tmp, "old")
        new_dir = os.path.join(tmp, "new")

        # For empty tree (initial commit), create an empty old_dir
        if base_ref == _get_empty_tree_sha():
            os.makedirs(old_dir, exist_ok=True)
        else:
            _snapshot_from_git(repo_path, base_ref, old_dir)

        _snapshot_from_git(repo_path, head_ref, new_dir)
        return detect_drift_between_dirs(old_dir, new_dir, similarity_threshold, changed_files=changed_files)


def _get_empty_tree_sha() -> str:
    """Return git's well-known empty tree SHA."""
    return "4b825dc642cb6eb9a060e54bf899d15363d7d95f"


if __name__ == "__main__":
    import sys
    import json
    from dataclasses import asdict

    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    base = sys.argv[2] if len(sys.argv) > 2 else "HEAD~1"
    head = sys.argv[3] if len(sys.argv) > 3 else "HEAD"

    flags, gaps = detect_drift_git(repo, base, head)
    print(json.dumps({
        "drift_flags": [asdict(f) for f in flags],
        "coverage_gaps": gaps,
    }, indent=2))
