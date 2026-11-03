"""
cli.py
------
Command-line entry point tying together drift_detector.py (+ optionally
llm_check.py) into a single Markdown report suitable for posting as a
GitHub PR comment.

Usage:
    python -m docdrift.cli --repo . --base origin/main --head HEAD --output drift_report.md
    python -m docdrift.cli --repo . --base HEAD~1 --head HEAD --use-llm
"""

import argparse
import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .drift_detector import detect_drift_git, detect_drift_between_dirs


def render_report(drift_flags, coverage_gaps, llm_verdicts=None) -> str:
    if not drift_flags and not coverage_gaps:
        return ""  # nothing to say -> no PR comment

    lines = ["## 📄 Documentation Drift Report", ""]

    if drift_flags:
        lines.append(f"Found **{len(drift_flags)}** function(s)/class(es) whose code changed "
                      f"without a matching documentation update:")
        lines.append("")
        for flag in drift_flags:
            lines.append(f"### `{flag.symbol_name}` — {flag.file_path}")
            lines.append(f"- **Why flagged:** {flag.reason}")
            lines.append(f"- **Old:** `{flag.old_signature}`")
            lines.append(f"- **New:** `{flag.new_signature}`")
            lines.append(f"- **Mapped doc section:** `{flag.mapped_doc}` (match confidence {flag.confidence:.2f})")
            if llm_verdicts:
                verdict = llm_verdicts.get(flag.symbol_name)
                if verdict and verdict.suggested_doc_text:
                    lines.append("- **Suggested doc update:**")
                    lines.append("  ```markdown")
                    lines.append("  " + verdict.suggested_doc_text.replace("\n", "\n  "))
                    lines.append("  ```")
            lines.append("")

    if coverage_gaps:
        lines.append("<details><summary>Symbols changed with no confident doc mapping "
                      f"({len(coverage_gaps)}) — consider adding docs</summary>")
        lines.append("")
        for g in coverage_gaps:
            lines.append(f"- `{g}`")
        lines.append("")
        lines.append("</details>")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Detect documentation drift between two git refs or directories.")
    parser.add_argument("--repo", default=".", help="Path to the git repo")
    parser.add_argument("--base", default="HEAD~1", help="Base ref (e.g. origin/main)")
    parser.add_argument("--head", default="HEAD", help="Head ref (e.g. HEAD, PR branch)")
    parser.add_argument("--old-dir", default=None, help="Compare two local directories instead of git refs (baseline dir)")
    parser.add_argument("--new-dir", default=None, help="Target dir to compare against --old-dir")
    parser.add_argument("--threshold", type=float, default=0.12, help="TF-IDF cosine similarity threshold")
    parser.add_argument("--output", default=None, help="Write Markdown report to this path")
    parser.add_argument("--use-llm", action="store_true", help="Run LLM verification pass to filter false positives and suggest doc updates")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai-compatible", "vllm", "anthropic"], help="Verifier LLM provider (default: ollama)")
    parser.add_argument("--model", default=None, help="Verifier model name (default: 'llama3.2' for ollama, 'claude-sonnet-4-6' for anthropic)")
    parser.add_argument("--endpoint", default="http://localhost:11434", help="Endpoint URL for verifier LLM (default: http://localhost:11434)")
    parser.add_argument("--api-key", default=None, help="API key for verifier LLM (optional for local models)")
    parser.add_argument("--writer-provider", default=None, choices=["ollama", "openai-compatible", "vllm", "anthropic"], help="LLM provider for doc writer (defaults to --provider)")
    parser.add_argument("--writer-model", default=None, help="Model name for doc writer (defaults to --model)")
    parser.add_argument("--writer-endpoint", default=None, help="Endpoint URL for doc writer (defaults to --endpoint)")
    parser.add_argument("--writer-api-key", default=None, help="API key for doc writer (defaults to --api-key)")
    parser.add_argument("--apply-fixes", "--apply", action="store_true", help="Automatically apply LLM-generated suggested documentation updates to files")
    parser.add_argument("--interactive", action="store_true", help="Prompt for confirmation before applying each documentation fix")
    args = parser.parse_args()

    repo_target = args.new_dir if args.new_dir else args.repo

    if args.old_dir and args.new_dir:
        drift_flags, coverage_gaps = detect_drift_between_dirs(args.old_dir, args.new_dir, args.threshold)
    else:
        drift_flags, coverage_gaps = detect_drift_git(args.repo, args.base, args.head, args.threshold)

    llm_verdicts = None
    confirmed_pairs = []
    if args.use_llm and drift_flags:
        from .llm_check import verify_drift
        model_name = args.model or ("claude-sonnet-4-6" if args.provider == "anthropic" else "llama3.2")
        writer_provider_name = args.writer_provider or args.provider
        writer_model_name = args.writer_model or (args.model if args.model else ("claude-sonnet-4-6" if writer_provider_name == "anthropic" else "llama3.2"))
        writer_endpoint_name = args.writer_endpoint or args.endpoint
        writer_api_key_name = args.writer_api_key or args.api_key

        llm_verdicts = {}
        for flag in drift_flags:
            verdict = verify_drift(
                flag,
                old_doc_text=flag.doc_text,
                code_diff=flag.code_diff,
                provider=args.provider,
                model=model_name,
                endpoint=args.endpoint,
                api_key=args.api_key,
                writer_provider=writer_provider_name,
                writer_model=writer_model_name,
                writer_endpoint=writer_endpoint_name,
                writer_api_key=writer_api_key_name,
            )
            if verdict.is_drift:
                llm_verdicts[flag.symbol_name] = verdict
                confirmed_pairs.append((flag, verdict))
        # Only keep flags the LLM actually confirmed
        drift_flags = [f for f in drift_flags if f.symbol_name in llm_verdicts]

    report = render_report(drift_flags, coverage_gaps, llm_verdicts)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to {args.output} ({len(drift_flags)} drift flags, {len(coverage_gaps)} coverage gaps)")
    else:
        print(report if report else "No drift detected.")

    if (args.apply_fixes or args.interactive) and confirmed_pairs:
        from .updater import apply_doc_updates
        applied = apply_doc_updates(confirmed_pairs, repo_target, interactive=args.interactive)
        print(f"\n[docdrift] Successfully applied {applied} documentation update(s).")


if __name__ == "__main__":
    main()
