"""OpenAI-compatible chat backend (self-hosted SGLang/vLLM servers) with a Converse-shaped interface.

The harness keeps messages in Bedrock Converse form (content blocks: text / toolUse / toolResult).
This adapter translates them to OpenAI chat-completions messages for each call, and translates the
response back, so the episode loop is identical for Bedrock and self-hosted models.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        texts = [b["text"] for b in m["content"] if "text" in b]
        if m["role"] == "assistant":
            calls = [{"id": b["toolUse"]["toolUseId"], "type": "function",
                      "function": {"name": b["toolUse"]["name"], "arguments": json.dumps(b["toolUse"].get("input") or {})}}
                     for b in m["content"] if "toolUse" in b]
            msg = {"role": "assistant", "content": "\n".join(texts) or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            results = [b["toolResult"] for b in m["content"] if "toolResult" in b]
            for r in results:
                payload = r["content"][0].get("json", r["content"][0].get("text"))
                out.append({"role": "tool", "tool_call_id": r["toolUseId"],
                            "content": json.dumps(payload) if not isinstance(payload, str) else payload})
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["schema"]}} for t in tools]


def from_openai_response(resp: dict) -> dict:
    """Return a Converse-shaped response: output.message.content blocks, usage, stopReason."""
    choice = resp["choices"][0]
    msg = choice["message"]
    content = []
    if msg.get("reasoning_content"):
        content.append({"reasoningContent": {"reasoningText": {"text": msg["reasoning_content"]}}})
    if msg.get("content"):
        content.append({"text": msg["content"]})
    for call in msg.get("tool_calls") or []:
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {"_unparseable_arguments": call["function"]["arguments"]}
        content.append({"toolUse": {"toolUseId": call["id"], "name": call["function"]["name"], "input": args}})
    if not content:
        content.append({"text": ""})
    u = resp.get("usage") or {}
    details = u.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    return {"output": {"message": {"role": "assistant", "content": content}},
            "stopReason": choice.get("finish_reason"),
            "usage": {"inputTokens": u.get("prompt_tokens", 0) - cached, "outputTokens": u.get("completion_tokens", 0),
                      "cacheReadInputTokens": cached, "cacheWriteInputTokens": 0}}


class OpenAICompatAgent:
    """Drop-in replacement for BedrockAgent.converse() against a local OpenAI-compatible server."""

    def __init__(self, base_url: str, model_name: str, max_tokens: int, timeout_s: float = 900.0,
                 template_kwargs: dict | None = None):
        self.base_url, self.model_id, self.max_tokens, self.timeout_s = base_url.rstrip("/"), model_name, max_tokens, timeout_s
        self.template_kwargs = template_kwargs or {}

    def converse(self, system: str, messages: list, tools: list, cache: bool, force_tool: str | None = None) -> dict:
        # cache: the server's radix cache handles prefix reuse. force_tool: named tool_choice, which SGLang
        # enforces with constrained decoding (guarantees well-formed arguments).
        choice = {"type": "function", "function": {"name": force_tool}} if force_tool else "auto"
        req_body = {"model": self.model_id, "messages": to_openai_messages(system, messages),
                    "tools": to_openai_tools(tools), "tool_choice": choice, "max_tokens": self.max_tokens}
        if self.template_kwargs:  # e.g. thinking switches; SGLang passes these to the chat template
            req_body["chat_template_kwargs"] = self.template_kwargs
        body = json.dumps(req_body).encode()
        for attempt in range(6):
            try:
                req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    return from_openai_response(json.loads(r.read()))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == 5:
                    raise RuntimeError(f"OpenAI-compatible endpoint failed: {exc}") from exc
                time.sleep(min(60, 2 ** attempt * 2))
        raise RuntimeError("unreachable")
