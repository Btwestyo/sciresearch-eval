"""The final submit-only turn must survive models that reject a reduced tool list (Claude Fable 5.2)."""
from botocore.exceptions import ClientError

from sciresearch_eval.agents import harness


class FakeAgent:
    """Rejects any request whose tool list is shorter than the first one, like Fable 5.2's thinking binding."""

    def __init__(self, *a, **k):
        self.n_tools, self.calls = None, []

    def converse(self, system, messages, tools, cache, force_tool=None):
        self.calls.append(len(tools))
        if self.n_tools is None:
            self.n_tools = len(tools)
        if len(tools) < self.n_tools:
            raise ClientError({"Error": {"Code": "ValidationException", "Message":
                               "Invalid `signature` in `thinking` block. The `tools` list differs"}}, "Converse")
        return {"usage": {"inputTokens": 10, "outputTokens": 5},
                "output": {"message": {"role": "assistant", "content": [
                    {"toolUse": {"toolUseId": f"t{len(self.calls)}", "name": "budget_status", "input": {}}}]}}}


def test_final_turn_keeps_tools_and_refuses_non_submit(tmp_path, monkeypatch):
    agents = []
    monkeypatch.setattr(harness, "BedrockAgent", lambda *a, **k: agents.append(FakeAgent()) or agents[-1])
    r = harness.run_episode("us.anthropic.claude-fable-5-2", "A", "task_bc36c6f990", 0, tmp_path, split="dev",
                            form="open", max_input_tokens=1)
    assert r["stop"] == "limit_turns_or_tokens"  # not "error: ValidationException ..."
    calls = agents[0].calls
    assert calls[0] == calls[-1] and len(calls) == 3  # turn 1, rejected reduced list, retry with the full list
    assert r["usage"]["tool_errors"] == 1  # the final-turn budget_status call was refused
