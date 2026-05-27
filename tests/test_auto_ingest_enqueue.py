from nesy_reasoning_mcp.auto_ingest.enqueue import should_enqueue


def test_should_enqueue_nesy_facts_priority_and_skip_extraction() -> None:
    decision = should_enqueue('final\nNESY_FACTS:\n[{"source":"A","target":"B"}]')

    assert decision.enqueue is True
    assert decision.priority == 1
    assert decision.skip_extraction is True
    assert decision.reason == "nesy_facts"


def test_should_enqueue_skips_short_messages() -> None:
    decision = should_enqueue("A requires B.")

    assert decision.enqueue is False
    assert decision.reason == "too_short"


def test_should_enqueue_skips_code_heavy_messages() -> None:
    code = "```python\n" + ("print('x')\n" * 80) + "```\nrequires relation outside code"

    decision = should_enqueue(code)

    assert decision.enqueue is False
    assert decision.reason == "code_heavy"


def test_should_enqueue_matches_english_structural_keyword() -> None:
    message = (
        "This proposal explains why the new Stop hook requires a durable queue before "
        "the later extraction worker can run safely. " * 3
    )

    decision = should_enqueue(message)

    assert decision.enqueue is True
    assert decision.priority == 0
    assert decision.skip_extraction is False
    assert decision.reason == "structural_keyword"


def test_should_enqueue_matches_chinese_structural_keyword() -> None:
    message = (
        "这个计划说明为什么停止钩子需要先记录会话和 transcript 路径，"
        "因此后续 worker 才能在后台恢复上下文并继续处理。" * 4
    )

    decision = should_enqueue(message)

    assert decision.enqueue is True
    assert decision.reason == "structural_keyword"


def test_should_enqueue_classifier_flag_only_adds_diagnostic() -> None:
    message = (
        "This message requires durable enqueueing because the later worker depends on "
        "the transcript path and session id being available after restart. " * 3
    )

    decision = should_enqueue(message, env={"NESY_ENQUEUE_CLASSIFIER": "true"})

    assert decision.enqueue is True
    assert decision.reason == "structural_keyword"
    assert decision.diagnostics == [
        "NESY_ENQUEUE_CLASSIFIER is set, but PR1 uses deterministic heuristics only."
    ]
