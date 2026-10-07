"""The system prompt belongs to the call, never to the caller's `conversation=` list.

Compaction hands back copies of the messages, the system prompt among them, and
whatever the list holds is replayed on the next call.
"""

import from jaclang.byllm.lib { MockLLM }
import from jaclang.byllm.mtir { MTRuntime }
import from jaclang.byllm.types { CompletionResult, MockToolCall }
import from jaclang.byllm.tool_protocol { coerce_message_dict }
import from jaclang.byllm._optdeps.loguru { logger }


"""Big enough that the built-in compaction has something worth summarising."""
def read -> str {
    return "result " * 80;
}


def finish_tool(final_output: str) -> str {
    return final_output;
}


"""A MockLLM that records the messages of every request it is sent."""
obj RecordingLLM(MockLLM) {
    has sent: list = [];

    override def dispatch_no_streaming(mt_run: MTRuntime) -> CompletionResult {
        self.sent.append(list(self.make_model_params(mt_run)["messages"]));
        return super.dispatch_no_streaming(mt_run);
    }
}


def call(prompt_tokens: int) -> tuple {
    return (MockToolCall(tool=read, args={}), {"prompt_tokens": prompt_tokens});
}


def finish(output: str) -> MockToolCall {
    return MockToolCall(tool=finish_tool, args={"final_output": output});
}


def roles(llm: RecordingLLM, n: int = 0) -> list[str] {
    return [str(coerce_message_dict(m).get("role")) for m in llm.sent[n]];
}


"""Keeps the system prompt and the task, as the `on_compaction` docs describe."""
def keep_two(msgs: list[any], keep_recent: int) -> list[any] {
    return [msgs[0], msgs[1]];
}


test "a compaction does not store the system prompt, so the next turn sends one" {
    rows = [
        ("a hook that keeps it", keep_two, []),
        ("the built-in compaction", None, ["SUMMARY"])
    ];
    for (label, hook, summary) in rows {
        history: list[dict[str, any]] = [];
        llm = RecordingLLM(
            model_name="mockllm",
            ctx_window=1000,
            config={
                "outputs": [call(100), call(850)] + summary + [
                    finish("first"),
                    finish("second")
                ]
            }
        );

        def ask(q: str) -> str by llm(
            tools=[read],
            conversation=history,
            on_compaction=hook,
            keep_recent_iterations=1
        );

        log_lines: list = [];
        sink_id = logger.add(log_lines.append, level="DEBUG");
        try {
            ask("one");
        } finally {
            logger.remove(sink_id);
        }
        assert "reduced message list" in "".join(log_lines) , f"{label}: did not compact";
        ask("two");

        stored = [m["role"] for m in history];
        assert "system" not in stored , f"{label}: stored {stored}";
        last = roles(llm, len(llm.sent) - 1);
        assert last.count("system") == 1 , f"{label}: the next turn sent {last}";
    }
}


test "a system message already stored in the list is not replayed or kept" {
    history: list[dict[str, any]] = [
        {"role": "system", "content": "STORED_SYSTEM_PROMPT"},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"}
    ];
    llm = RecordingLLM(model_name="mockllm", config={"outputs": ["ok"]});

    def ask(q: str) -> str by llm(conversation=history);

    ask("next question");

    assert roles(llm) == ["system", "user", "assistant", "user"] , f"sent: {roles(
        llm
    )}";
    assert "STORED_SYSTEM_PROMPT" not in str(llm.sent[0]);
    assert [m["role"] for m in history] == ["user", "assistant", "user"] , f"stored: {history}";
}
