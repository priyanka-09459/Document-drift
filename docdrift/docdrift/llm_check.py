"""
llm_check.py
------------
Takes a DriftFlag (from drift_detector.py) plus the actual code diff and
current doc text, and asks LLMs to:
  1. Confirm whether the doc is genuinely stale (Verifier Pass).
  2. Propose replacement doc text if it is stale (Writer Pass).

Supports using separate LLM providers/models for verification vs doc writing.

Providers supported:
  - Local models via Ollama (default endpoint: http://localhost:11434)
  - Local models via vLLM / OpenAI-compatible servers (e.g. LM Studio, LocalAI)
  - Cloud models via Anthropic (requires ANTHROPIC_API_KEY)

You can easily modify system prompts, provider logic, or add new LLM providers below.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

try:
    import requests
except ImportError:
    requests = None


# =====================================================================
# SYSTEM PROMPTS (Modify these to customize LLM behavior)
# =====================================================================

VERIFIER_SYSTEM_PROMPT = """You are an expert technical documentation reviewer. You will be given:
- A function, class, or configuration whose signature, type annotations, or body implementation changed, or that was deleted.
- The documentation section currently mapped to it (which was NOT updated).
- The code diff or modified source snippet.

Your job is to decide if the documentation is now inaccurate, stale, or incomplete given the code change.
Guidelines:
- Minor internal refactors with NO observable behavioral change, no signature change, and no impact on callers are NOT drift.
- The following ARE drift:
  1. Parameter changes (new, renamed, reordered, removed, or changed default values).
  2. Type annotation and return type changes (e.g. from int to str).
  3. Logic and behavior changes in the function body (new exceptions raised, altered calculations, new return formats).
  4. Changed configuration values or constants.
  5. Deleted symbols where the documentation still describes the removed function (zombie documentation).

Respond with ONLY valid JSON:
{
  "is_drift": true,
  "reason": "<concise one-sentence explanation of why the doc is stale or why it is not drift>"
}
"""

WRITER_SYSTEM_PROMPT = """You are an expert technical documentation writer. You will be given:
- A symbol whose code was updated, causing its documentation to become stale.
- The current (unmodified) documentation section.
- The code diff / implementation changes.
- The reason why the documentation is stale.

Your job is to write updated, accurate, ready-to-use Markdown documentation for this section.
Guidelines:
- Maintain the style, formatting, and tone of the original documentation section.
- Clearly describe any new parameters, changed behaviors, return types, or exceptions.
- If the symbol was deleted, suggest removing or deprecating that section in the documentation.
- Output ONLY the replacement Markdown content for the section. Do NOT wrap in JSON or extra commentary.
"""


@dataclass
class LLMVerdict:
    is_drift: bool
    reason: str
    suggested_doc_text: str = ""


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def _clean_json_response(text: str) -> dict:
    """Extract and parse JSON from model responses, handling markdown and think blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {
        "is_drift": True,
        "reason": "LLM response could not be parsed as JSON; defaulting to flagged.",
    }


def _clean_markdown_response(text: str) -> str:
    """Clean LLM markdown response, removing reasoning tags and surrounding triple backticks if any."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```markdown") and text.endswith("```"):
        text = text[11:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    return text


# =====================================================================
# PROMPT BUILDERS (Modify prompt formatting here)
# =====================================================================

def build_verifier_prompt(flag, old_doc_text: str = "", code_diff: str = "") -> str:
    doc_text = old_doc_text or getattr(flag, "doc_text", "") or "(no existing doc text available)"
    diff_text = code_diff or getattr(flag, "code_diff", "") or "(not provided)"

    return f"""Symbol: {flag.symbol_name}
File: {flag.file_path}

Old signature: {flag.old_signature}
New signature: {flag.new_signature}

Current (unmodified) documentation section [{flag.mapped_doc}]:
---
{doc_text}
---

Code diff / implementation changes:
---
{diff_text}
---
"""


def build_writer_prompt(flag, old_doc_text: str = "", code_diff: str = "", reason: str = "") -> str:
    doc_text = old_doc_text or getattr(flag, "doc_text", "") or "(no existing doc text available)"
    diff_text = code_diff or getattr(flag, "code_diff", "") or "(not provided)"

    return f"""Symbol: {flag.symbol_name}
File: {flag.file_path}

Reason documentation is stale:
{reason or flag.reason}

Old signature: {flag.old_signature}
New signature: {flag.new_signature}

Current documentation section [{flag.mapped_doc}]:
---
{doc_text}
---

Code diff / implementation changes:
---
{diff_text}
---

Write the updated Markdown section content:
"""


# =====================================================================
# LLM PROVIDER DISPATCHER (Extend or modify model providers here)
# =====================================================================

def call_llm_provider(
    system_prompt: str,
    user_prompt: str,
    provider: str = "ollama",
    model: str = "llama3.2",
    endpoint: str = "http://localhost:11434",
    api_key: Optional[str] = None,
    json_mode: bool = False,
) -> str:
    """Generic entry point for dispatching requests to configured LLM provider."""
    provider_clean = (provider or "ollama").lower().strip()

    if provider_clean == "ollama":
        return _call_ollama(
            system_prompt,
            user_prompt,
            model=model or "llama3.2",
            endpoint=endpoint or "http://localhost:11434",
            json_mode=json_mode,
        )
    elif provider_clean in ("openai", "openai-compatible", "vllm", "local"):
        return _call_openai_compatible(
            system_prompt,
            user_prompt,
            model=model or "default",
            endpoint=endpoint or "http://localhost:8000/v1",
            api_key=api_key,
        )
    elif provider_clean == "anthropic":
        return _call_anthropic(
            system_prompt,
            user_prompt,
            model=model or "claude-sonnet-4-6",
            api_key=api_key,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Choose 'ollama', 'vllm'/'openai-compatible', or 'anthropic'."
        )


def _call_ollama(
    system_prompt: str,
    user_prompt: str,
    model: str = "llama3.2",
    endpoint: str = "http://localhost:11434",
    json_mode: bool = False,
) -> str:
    if requests is None:
        raise RuntimeError("'requests' library is required to connect to Ollama. Install it via `pip install requests`.")

    base_url = endpoint.rstrip("/")
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    if json_mode:
        payload["format"] = "json"

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to Ollama at {base_url}. "
            f"Make sure Ollama is running (`ollama serve`) and model '{model}' is pulled (`ollama pull {model}`)."
        )
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")


def _call_openai_compatible(
    system_prompt: str,
    user_prompt: str,
    model: str = "default",
    endpoint: str = "http://localhost:8000/v1",
    api_key: Optional[str] = None,
) -> str:
    if requests is None:
        raise RuntimeError("'requests' library is required. Install it via `pip install requests`.")

    base_url = endpoint.rstrip("/")
    url = f"{base_url}/chat/completions" if not base_url.endswith("/chat/completions") else base_url
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or 'EMPTY'}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""
    except Exception as e:
        raise RuntimeError(f"OpenAI-compatible request to {url} failed: {e}")


def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "Anthropic library is not installed. Install via `pip install anthropic` or use `--provider ollama`."
        )

    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Provide --api-key or set the ANTHROPIC_API_KEY environment variable."
        )

    client = anthropic.Anthropic(api_key=resolved_key)
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# =====================================================================
# MAIN PIPELINE FUNCTIONS (DRIFT VERIFIER & DOC WRITER)
# =====================================================================

def generate_doc_fix(
    flag,
    old_doc_text: str = "",
    code_diff: str = "",
    reason: str = "",
    writer_provider: str = "ollama",
    writer_model: str = "llama3.2",
    writer_endpoint: str = "http://localhost:11434",
    writer_api_key: Optional[str] = None,
) -> str:
    """Generate updated Markdown documentation using the designated writer LLM."""
    prompt = build_writer_prompt(flag, old_doc_text, code_diff, reason)
    raw_response = call_llm_provider(
        system_prompt=WRITER_SYSTEM_PROMPT,
        user_prompt=prompt,
        provider=writer_provider,
        model=writer_model,
        endpoint=writer_endpoint,
        api_key=writer_api_key,
        json_mode=False,
    )
    return _clean_markdown_response(raw_response)


def verify_drift(
    flag,
    old_doc_text: str = "",
    code_diff: str = "",
    provider: str = "ollama",
    model: str = "llama3.2",
    endpoint: str = "http://localhost:11434",
    api_key: Optional[str] = None,
    writer_provider: Optional[str] = None,
    writer_model: Optional[str] = None,
    writer_endpoint: Optional[str] = None,
    writer_api_key: Optional[str] = None,
) -> LLMVerdict:
    """
    Step 1: Check drift using the Verifier LLM.
    Step 2: If drift is confirmed, generate updated doc section using the Writer LLM.
    """
    # 1. Verifier Pass
    verifier_prompt = build_verifier_prompt(flag, old_doc_text, code_diff)
    raw_verifier_resp = call_llm_provider(
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        user_prompt=verifier_prompt,
        provider=provider,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        json_mode=True,
    )
    verifier_data = _clean_json_response(raw_verifier_resp)
    is_drift = bool(verifier_data.get("is_drift", True))
    reason = str(verifier_data.get("reason", ""))

    if not is_drift:
        return LLMVerdict(is_drift=False, reason=reason, suggested_doc_text="")

    # 2. Writer Pass (resolves writer options to verifier defaults if not provided)
    w_provider = writer_provider or provider
    w_model = writer_model or model
    w_endpoint = writer_endpoint or endpoint
    w_api_key = writer_api_key or api_key

    suggested_doc = generate_doc_fix(
        flag,
        old_doc_text=old_doc_text,
        code_diff=code_diff,
        reason=reason,
        writer_provider=w_provider,
        writer_model=w_model,
        writer_endpoint=w_endpoint,
        writer_api_key=w_api_key,
    )

    return LLMVerdict(
        is_drift=True,
        reason=reason,
        suggested_doc_text=suggested_doc,
    )


def verify_batch(
    flags: list,
    doc_texts: Optional[dict] = None,
    code_diffs: Optional[dict] = None,
    provider: str = "ollama",
    model: str = "llama3.2",
    endpoint: str = "http://localhost:11434",
    api_key: Optional[str] = None,
    writer_provider: Optional[str] = None,
    writer_model: Optional[str] = None,
    writer_endpoint: Optional[str] = None,
    writer_api_key: Optional[str] = None,
) -> list[tuple]:
    """
    flags: list of DriftFlag
    doc_texts: optional {flag.mapped_doc: current doc text}
    code_diffs: optional {flag.symbol_name: diff text}
    Returns list of (DriftFlag, LLMVerdict) for flags the LLM confirms as real drift.
    """
    doc_texts = doc_texts or {}
    code_diffs = code_diffs or {}
    confirmed = []
    for flag in flags:
        old_text = doc_texts.get(flag.mapped_doc) or getattr(flag, "doc_text", "")
        diff = code_diffs.get(flag.symbol_name) or getattr(flag, "code_diff", "")
        verdict = verify_drift(
            flag,
            old_doc_text=old_text,
            code_diff=diff,
            provider=provider,
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            writer_provider=writer_provider,
            writer_model=writer_model,
            writer_endpoint=writer_endpoint,
            writer_api_key=writer_api_key,
        )
        if verdict.is_drift:
            confirmed.append((flag, verdict))
    return confirmed
