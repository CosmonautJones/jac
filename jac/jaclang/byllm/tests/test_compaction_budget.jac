"""Compaction sizes what it keeps in tokens, not in messages.

The trigger is a token ratio, so what compaction keeps and what it asks the summariser to
shrink are budgeted in tokens too. These tests pin that with the default settings, which
keep three assistant rounds.
"""

import litellm;
import re;

import jaclang.byllm.llm as _byllm_llm_mod;
import from jaclang.byllm.lib { MockLLM, MockError }
import from jaclang.byllm.mtir { MTRuntime }
import from jaclang.byllm.types { CompletionResult, MockToolCall }
import from jaclang.byllm.tool_protocol { coerce_message_dict }
import from jaclang.byllm._optdeps.loguru { logger }
import from jaclang.byllm._optdeps.litellm { LiteLLMMessage }


"""A tool result of `size` characters: 4 characters is about one token."""
def chunk(size: int) -> str {
    return "x" * size;
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


def call(size: int, prompt_tokens: int = 0) -> tuple {
    return (
        MockToolCall(tool=chunk, args={"size": size}),
        {"prompt_tokens": prompt_tokens}
    );
}


def finish(output: str) -> MockToolCall {
    return MockToolCall(tool=finish_tool, args={"final_output": output});
}


def roles(llm: RecordingLLM, n: int) -> list[str] {
    return [str(coerce_message_dict(m).get("role")) for m in llm.sent[n]];
}


"""One assistant round and its results: an int for one call, a list for parallel calls."""
def make_round(idx: int, result_chars: any) -> list[any] {
    sizes = result_chars if isinstance(result_chars, list) else [result_chars];
    ids = [f"call_{idx}_{n}" for n in range(len(sizes))];
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": i, "function": {"name": "search", "arguments": "{}"}}
                for i in ids
            ]
        }
    ] + [
        {"role": "tool", "content": "r" * chars, "tool_call_id": i, "name": "search"}
        for (i, chars) in zip(ids, sizes)
    ];
}


"""A system and user preamble, then `round_count` rounds of the same size."""
def history(round_count: int, result_chars: any) -> list[any] {
    messages: list[any] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Do the thing."}
    ];
    for idx in range(1, round_count + 1) {
        messages += make_round(idx, result_chars);
    }
    return messages;
}


"""What compaction returns in a 4000-token window, whose tail budget is 1000 tokens."""
def compact(
    messages: list[any], keep_recent: int = 3, summariser: any = None
) -> list[any] {
    return list(
        _byllm_llm_mod._default_compact(
            messages,
            keep_recent,
            summariser or MockLLM(model_name="mockllm", config={"outputs": ["S"]}),
            ctx_window=4000
        )
    );
}


"""The `key` of every message with this role; "id" means the first tool call's."""
def field(messages: list[any], role: str, key: str) -> list[any] {
    return [
        m["tool_calls"][0]["id"] if key == "id" else m[key]
        for m in messages
        if m["role"] == role
    ];
}


"""One turn in a 1000-token window. Returns the model and the log it wrote."""
def one_turn(
    outputs: list[any], tools: list[any], on_compaction: any = None
) -> tuple[RecordingLLM, str] {
    llm = RecordingLLM(
        model_name="mockllm", ctx_window=1000, config={"outputs": outputs}
    );

    def ask(q: str) -> str by llm(tools=tools, on_compaction=on_compaction);

    log_lines: list = [];
    sink_id = logger.add(log_lines.append, level="DEBUG");
    try {
        assert ask("go") == "ok";
    } finally {
        logger.remove(sink_id);
    }
    return (llm, "".join(log_lines));
}


test "the default settings compact when the threshold is crossed after two rounds" {
    (llm, _) = one_turn(
        [call(1200, 100), call(1200, 900), "SUMMARY_TEXT", finish("ok")], [chunk]
    );
    # The third request is the summariser's: it saw only the first round.
    assert [len(m) for m in llm.sent] == [2, 4, 2, 5];
    assert roles(llm, 3) == ["system", "user", "user", "assistant", "tool"];
    assert "SUMMARY_TEXT" in llm.sent[3][2]["content"];
}


test "a reply held as a LiteLLM message is sized and summarised like its dict" {
    for reply in [
        make_round(1, 8)[0],
        dict(make_round(1, 8)[0], content="LET_ME_CHECK")
    ] {
        held = LiteLLMMessage(**reply);
        line = _byllm_llm_mod._compaction_line(held);
        assert line == _byllm_llm_mod._compaction_line(reply) , line;
        assert _byllm_llm_mod._estimate_tokens([held]) == _byllm_llm_mod._estimate_tokens(
            [reply]
        );
    }
}


test "the kept tail is the newest rounds that fit the token budget" {
    cases = [
        ("every requested round fits", 1000, ["call_2_0", "call_3_0", "call_4_0"]),
        ("the oldest round is dropped to fit", 1500, ["call_3_0", "call_4_0"]),
        ("the newest round is kept however large", 6000, ["call_4_0"])
    ];
    for (label, result_chars, expected) in cases {
        got = field(compact(history(4, result_chars)), "assistant", "id");
        assert got == expected , f"{label}: kept {got}";
    }
}


test "the kept tail's results are clipped on a copy only when they exceed the budget" {
    cases = [
        ("within the budget", history(3, 1000), [1000, 1000, 1000], False),
        (
            "over it, the two results of one round share it",
            history(1, 1000) + make_round(2, [8000, 8000]),
            [2000, 2000],
            True
        )
    ];
    for (label, messages, want_lens, clipped) in cases {
        before = field(messages, "tool", "content");
        got = field(compact(messages), "tool", "content");
        assert len(got) == len(want_lens) , f"{label}: {len(got)} results kept";
        for (content, want) in zip(got, want_lens) {
            assert want <= len(content) < want + 100 , (
                f"{label}: {len(content)} characters"
            );
            assert ("[truncated" in content) == clipped , label;
        }
        assert field(messages, "tool", "content") == before , f"{label}: the input was mutated";
    }
}


test "a slice too small to be worth a summariser call is left alone" {
    # A 4000-token window skips slices under 200 tokens (800 characters).
    cases = [("a few small results", 100, 0), ("results past the minimum", 1000, 1)];
    for (label, result_chars, summariser_calls) in cases {
        messages = history(3, result_chars);
        summariser = MockLLM(model_name="mockllm", config={"outputs": ["S"]});
        compacted = compact(messages, 1, summariser);
        assert len(summariser.seen_prompts) == summariser_calls , (
            f"{label}: {len(summariser.seen_prompts)} summariser calls"
        );
        assert (compacted == messages) == (summariser_calls == 0) , label;
    }
}


test "a context overflow caused by one huge tool result is recovered by truncating it" {
    (llm, _) = one_turn(
        [
            call(40000),
            MockError(
                error=litellm.exceptions.ContextWindowExceededError(
                    message="ctx exceeded", model="mockllm", llm_provider="openai"
                )
            ),
            finish("ok")
        ],
        [chunk]
    );
    # The overflowing request, then the retry; only the retry carries the clipped result.
    assert len(llm.sent[1][-1]["content"]) == 40000;
    retried = llm.sent[2][-1]["content"];
    assert len(retried) < 2000;
    assert retried.startswith("x" * 100);
    assert "[truncated" in retried;
    assert llm.config["outputs"] == [];
}


test "a threshold crossing with nothing worth summarising neither fails nor summarises" {
    (llm, logs) = one_turn([call(8, 100), call(8, 900), finish("ok")], [chunk]);
    assert [len(m) for m in llm.sent] == [2, 4, 6];
    assert "Compacting message history" in logs;
    assert "could not reduce the message history" in logs;
}


test "a history still over the threshold after compaction lets the call continue" {
    (llm, logs) = one_turn(
        [
            call(1200, 100),
            call(1200, 900),
            "SUMMARY_TEXT",
            call(1200, 850),
            finish("ok")
        ],
        [chunk]
    );
    assert llm.config["outputs"] == [];
    assert "still at or above threshold" in logs;
}


test "a hook that shrinks a tool result without removing a message counts as a reduction" {
    def stub_results(msgs: list[any], keep_recent: int) -> list[any] {
        return [
            dict(m, content="[stubbed]")
                if isinstance(m, dict) and m["role"] == "tool"
                else m for m in msgs
        ];
    }

    (llm, logs) = one_turn(
        [call(1200, 100), call(1200, 900), finish("ok")],
        [chunk],
        on_compaction=stub_results
    );
    # Same message count, far fewer tokens: the next request carries the stub.
    assert len(llm.sent[2]) == 6;
    assert llm.sent[2][3]["content"] == "[stubbed]";
    assert "could not reduce the message history" not in logs;
    assert re.search(r"reduced message list from (\d+) to \1 messages", logs) , logs;
}
