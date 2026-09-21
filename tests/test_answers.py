from metacog.answers import extract_answer


def test_extract_boxed():
    assert extract_answer("working it out… \\boxed{385}") == "385"


def test_extract_answer_line():
    assert extract_answer("some reasoning\nAnswer: D") == "D"


def test_extract_final_answer_bold():
    assert extract_answer("reasoning… The final answer is **42**.") == "42"


def test_extract_none_when_absent():
    assert extract_answer("no answer here") is None


def test_boxed_wins_over_answer_line():
    text = "Answer: 7 early on\nmore work… \\boxed{42}"
    assert extract_answer(text) == "42"
