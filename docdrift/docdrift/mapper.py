"""
mapper.py
---------
Maps documentation sections (chunks of Markdown/RST/plain-text docs) to
code symbols extracted by ast_parser.py.

Uses TF-IDF + cosine similarity as a dependency-light stand-in for a real
code embedding model (CodeBERT / UniXcoder / etc). The interface is kept
narrow (`fit`, `match`) so the TF-IDF backend can be swapped for a neural
embedding backend later without touching drift_detector.py.

Doc discovery:
  - README.md, CONTRIBUTING.md, CHANGELOG.md (repo root)
  - docs/, doc/, documentation/ directories (recursive)
  - Nested README.md files in subdirectories (1 level deep)
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ast_parser import Symbol


@dataclass
class DocChunk:
    doc_path: str
    heading: str
    text: str
    start_line: int


@dataclass
class Mapping:
    symbol: Symbol
    doc_chunk: DocChunk
    score: float


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# Common Markdown doc file names to look for at repo root
_ROOT_DOC_FILES = [
    "README.md",
    "readme.md",
    "CONTRIBUTING.md",
    "contributing.md",
    "CHANGELOG.md",
    "changelog.md",
    "HISTORY.md",
    "API.md",
    "api.md",
    "USAGE.md",
    "usage.md",
    "GUIDE.md",
    "guide.md",
]

# Common documentation directories
_DOC_DIRS = [
    "docs",
    "doc",
    "documentation",
    "wiki",
]


def chunk_markdown(file_path: str) -> list[DocChunk]:
    """Split a Markdown file into chunks by heading."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (OSError, IOError) as e:
        print(f"[docdrift] Warning: Could not read {file_path}: {e}")
        return []

    chunks: list[DocChunk] = []
    current_heading = "(intro)"
    current_lines: list[str] = []
    start_line = 1

    def flush():
        if current_lines and "".join(current_lines).strip():
            chunks.append(
                DocChunk(
                    doc_path=file_path,
                    heading=current_heading,
                    text="".join(current_lines).strip(),
                    start_line=start_line,
                )
            )

    for i, line in enumerate(lines, start=1):
        m = _HEADING_RE.match(line)
        if m:
            flush()
            current_heading = m.group(2).strip()
            current_lines = []
            start_line = i
        else:
            current_lines.append(line)
    flush()
    return chunks


def collect_doc_chunks(
    repo_root: str,
    extra_doc_dirs: tuple = (),
    max_depth: int = 3,
) -> list[DocChunk]:
    """
    Discover and parse all Markdown documentation files in the repo.

    Searches:
      1. Common doc files at repo root (README.md, CONTRIBUTING.md, etc.)
      2. Common doc directories (docs/, doc/, documentation/, wiki/)
      3. Any extra directories specified by the caller
      4. Subdirectory README.md files (1 level deep)

    Returns a flat list of DocChunks for all discovered Markdown files.
    """
    chunks: list[DocChunk] = []
    seen_files: set[str] = set()

    def _add_file(path: str):
        """Add a markdown file if not already seen."""
        norm = os.path.normpath(os.path.abspath(path))
        if norm in seen_files:
            return
        if not os.path.isfile(norm):
            return
        seen_files.add(norm)
        chunks.extend(chunk_markdown(norm))

    def _add_dir(dirpath: str, current_depth: int = 0):
        """Recursively add all .md files from a directory."""
        if not os.path.isdir(dirpath):
            return
        if current_depth > max_depth:
            return
        try:
            entries = os.listdir(dirpath)
        except PermissionError:
            return
        for entry in sorted(entries):
            full = os.path.join(dirpath, entry)
            if os.path.isfile(full) and entry.lower().endswith(".md"):
                _add_file(full)
            elif os.path.isdir(full) and entry not in (".git", "__pycache__", "node_modules", ".venv", "venv", ".tox"):
                _add_dir(full, current_depth + 1)

    # 1. Root-level doc files
    for fname in _ROOT_DOC_FILES:
        _add_file(os.path.join(repo_root, fname))

    # 2. Common doc directories
    for dname in _DOC_DIRS:
        _add_dir(os.path.join(repo_root, dname))

    # 3. Extra directories from caller
    for dname in extra_doc_dirs:
        _add_dir(os.path.join(repo_root, dname))

    # 4. Subdirectory README.md files (1 level deep — for monorepos / packages)
    try:
        for entry in os.listdir(repo_root):
            if entry.startswith(".") or entry in ("node_modules", "__pycache__", "venv", ".venv"):
                continue
            sub = os.path.join(repo_root, entry)
            if os.path.isdir(sub):
                readme = os.path.join(sub, "README.md")
                if os.path.isfile(readme):
                    _add_file(readme)
                readme_lower = os.path.join(sub, "readme.md")
                if os.path.isfile(readme_lower):
                    _add_file(readme_lower)
    except PermissionError:
        pass

    return chunks


class DocCodeMapper:
    """
    Fits a TF-IDF vectorizer over (symbol text, doc chunk text) and links
    each doc chunk to its best-matching code symbol(s) above a threshold.
    """

    def __init__(self, similarity_threshold: float = 0.12):
        self.similarity_threshold = similarity_threshold
        self.vectorizer: Optional[TfidfVectorizer] = None
        self._symbols: list[Symbol] = []
        self._doc_chunks: list[DocChunk] = []
        self._doc_vectors = None
        self._symbol_vectors = None

    @staticmethod
    def _symbol_text(sym: Symbol) -> str:
        # Signature + docstring + qualified name give the vectorizer real signal.
        name_tokens = re.sub(r"([a-z])([A-Z])", r"\1 \2", sym.qualified_name).replace("_", " ").replace(".", " ")
        return f"{name_tokens} {sym.signature} {sym.docstring}"

    @staticmethod
    def _doc_text(chunk: DocChunk) -> str:
        return f"{chunk.heading} {chunk.text}"

    def fit(self, symbols: list[Symbol], doc_chunks: list[DocChunk]):
        self._symbols = symbols
        self._doc_chunks = doc_chunks

        corpus = [self._symbol_text(s) for s in symbols] + [self._doc_text(c) for c in doc_chunks]
        if not corpus:
            return self

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        vectors = self.vectorizer.fit_transform(corpus)
        n_sym = len(symbols)
        self._symbol_vectors = vectors[:n_sym]
        self._doc_vectors = vectors[n_sym:]
        return self

    def match_all(self) -> list[Mapping]:
        """Return the best-matching symbol for every doc chunk above threshold."""
        if self._doc_vectors is None or self._symbol_vectors is None or len(self._symbols) == 0:
            return []

        sims = cosine_similarity(self._doc_vectors, self._symbol_vectors)  # (n_docs, n_symbols)
        mappings: list[Mapping] = []
        for i, chunk in enumerate(self._doc_chunks):
            best_j = sims[i].argmax()
            best_score = float(sims[i][best_j])
            if best_score >= self.similarity_threshold:
                mappings.append(Mapping(symbol=self._symbols[best_j], doc_chunk=chunk, score=best_score))
        return mappings

    def matches_for_symbol(self, symbol: Symbol, top_k: int = 3) -> list[Mapping]:
        """Return the top-k doc chunks most relevant to a given symbol (used by drift_detector)."""
        if self.vectorizer is None or self._doc_vectors is None:
            return []
        sym_vec = self.vectorizer.transform([self._symbol_text(symbol)])
        sims = cosine_similarity(sym_vec, self._doc_vectors)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        out = []
        for i in ranked[:top_k]:
            if sims[i] >= self.similarity_threshold:
                out.append(Mapping(symbol=symbol, doc_chunk=self._doc_chunks[i], score=float(sims[i])))
        return out
