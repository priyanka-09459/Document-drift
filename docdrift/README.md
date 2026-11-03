# DocDrift — Automated Documentation Drift Detection

Prototype implementation for *Automated Documentation Drift Detection and
Correction for Collaborative Software Repositories*.

## What's implemented (v0)

| Piece | File | Status |
|---|---|---|
| AST parsing → symbol extraction | `docdrift/ast_parser.py` | ✅ working |
| Doc-to-code mapping (TF-IDF cosine similarity) | `docdrift/mapper.py` | ✅ working, swappable for real embeddings |
| Drift detection (signature/body diff vs. doc diff) | `docdrift/drift_detector.py` | ✅ working, tested on toy repo |
| LLM verification + suggested rewrite | `docdrift/llm_check.py` | ✅ implemented, needs `ANTHROPIC_API_KEY` |
| CLI / report renderer | `docdrift/cli.py` | ✅ working |
| GitHub Action | `.github/workflows/docdrift.yml` | ✅ scaffolded, needs testing on a real PR |

## Quick start

```bash
pip install -r requirements.txt

# 1. Compare two commits/branches in any git repo:
python -m docdrift.cli --repo /path/to/repo --base HEAD~1 --head HEAD

# 2. Compare two local folders directly (no git needed):
python -m docdrift.cli --old-dir /path/to/old_repo --new-dir /path/to/new_repo

# 3. With Local LLM (Ollama - free & runs 100% on your machine):
# Ensure Ollama is running (`ollama run llama3.2` or mistral / qwen2.5-coder / deepseek-r1)
python -m docdrift.cli --repo /path/to/repo --base HEAD~1 --head HEAD --use-llm --provider ollama --model llama3.2

# 4. Automatically update documentation files with LLM fixes:
python -m docdrift.cli --repo /path/to/repo --base HEAD~1 --head HEAD --use-llm --provider ollama --model llama3.2 --apply-fixes

# 5. Interactive mode (prompt before modifying each documentation file):
python -m docdrift.cli --repo /path/to/repo --base HEAD~1 --head HEAD --use-llm --provider ollama --model llama3.2 --interactive

# 6. With Custom Local Endpoint (vLLM, LM Studio, LocalAI):
python -m docdrift.cli --repo /path/to/repo --base HEAD~1 --head HEAD --use-llm --provider vllm --endpoint http://localhost:8000/v1 --model your-model-name

# 7. With Anthropic (Claude):
export ANTHROPIC_API_KEY=sk-...
python -m docdrift.cli --repo /path/to/repo --base origin/main --head HEAD --use-llm --provider anthropic
```

## How it works

1. **`ast_parser.py`** walks all `.py` files and extracts every function,
   method, and class into a `Symbol`: qualified name, signature, docstring,
   and a hash of the body (docstring excluded, so pure docstring edits
   don't count as "code changed").
2. **`mapper.py`** chunks Markdown docs (README + `/docs`) by heading and
   uses TF-IDF + cosine similarity to link each doc chunk to its most
   relevant code symbol. This is the placeholder for the "code embeddings"
   layer in the proposal — swap in CodeBERT/UniXcoder here later.
3. **`drift_detector.py`** takes two snapshots (e.g. base branch vs. PR
   branch), finds symbols whose signature or body hash changed, looks up
   their mapped doc section from the *old* snapshot, and checks whether
   that doc section's text changed too. If code changed and the doc
   didn't → flagged as drift.
4. **`llm_check.py`** (optional pass) sends each flagged case to Claude to
   filter out non-behavioral refactors and draft replacement doc text.
5. **`cli.py`** renders everything into a Markdown report; the GitHub
   Action posts it as a PR comment.

## Validated on a toy example

`examples/toy_repo/` has 3 commits: a baseline `Calculator` class with
docs, a commit that changes `add()`'s signature and makes `divide()` raise
on zero (docs left untouched — simulated drift), and a commit that fixes
the docs for `add()` only. Running the detector across these correctly:
- Flags both `add()` and `divide()` after the undocumented change
- Stops flagging `add()` once its doc section is updated
- Keeps flagging `divide()` since its doc is still stale

Run it yourself:
```bash
python -m docdrift.cli --repo examples/toy_repo --base HEAD~2 --head HEAD
```

## Known limitations / next steps for the eval section of the project

- **TF-IDF mapping is a baseline, not the final method.** It works for
  the toy repo but will need validation against real repos where doc
  wording diverges more from code identifier names — this is exactly the
  "mapping accuracy" metric your abstract mentions evaluating. Plan: build
  a small hand-labeled (doc-section, function) pair dataset from 2–3 real
  OSS repos, measure TF-IDF precision/recall, then compare against a
  CodeBERT/UniXcoder embedding baseline.
- **Language scope:** currently Python-only via the `ast` module.
  Multi-language support would mean swapping to `tree-sitter`.
- **Body-hash granularity:** currently any body change trips a flag; the
  LLM pass is what filters "real" drift from harmless refactors — worth
  measuring false-positive rate pre- vs. post-LLM-filter as a metric.
- **GitHub Action not yet run against a live PR** — needs an
  `ANTHROPIC_API_KEY` secret and a real PR to validate the comment format
  end-to-end.
