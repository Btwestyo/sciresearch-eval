"""Offline round-trip tests of the Converse <-> OpenAI message translation."""

from sciresearch_eval.agents.openai_backend import from_openai_response, to_openai_messages, to_openai_tools


def test_round_trip_tool_call_and_result():
    messages = [
        {"role": "user", "content": [{"text": "task"}]},
        {"role": "assistant", "content": [{"text": "fit first"},
                                          {"toolUse": {"toolUseId": "c1", "name": "fit_parameters", "input": {"data_ids": ["d0"]}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "c1", "status": "success", "content": [{"json": {"fit_id": "f0"}}]}}]},
    ]
    out = to_openai_messages("sys", messages)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[2]["tool_calls"][0]["function"]["name"] == "fit_parameters"
    assert out[3]["tool_call_id"] == "c1" and '"fit_id"' in out[3]["content"]


def test_response_translation_with_reasoning_and_bad_json():
    resp = {"choices": [{"finish_reason": "tool_calls", "message": {
        "reasoning_content": "think", "content": None,
        "tool_calls": [{"id": "c2", "function": {"name": "submit", "arguments": "{bad json"}}]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 60}}}
    r = from_openai_response(resp)
    blocks = r["output"]["message"]["content"]
    assert "reasoningContent" in blocks[0] and blocks[1]["toolUse"]["input"] == {"_unparseable_arguments": "{bad json"}
    assert r["usage"] == {"inputTokens": 40, "outputTokens": 7, "cacheReadInputTokens": 60, "cacheWriteInputTokens": 0}


def test_tools_translation():
    t = to_openai_tools([{"name": "x", "description": "d", "schema": {"type": "object"}}])
    assert t[0]["function"]["parameters"] == {"type": "object"}
