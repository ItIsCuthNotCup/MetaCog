from metacog.eval.humaneval import build_problem, extract_code, run_test

PROMPT = 'def add(a, b):\n    """Add two numbers."""\n'
ENTRY = "add"


def test_extract_code_prefers_last_python_block():
    text = "Reasoning...\n```python\ndef add(a, b):\n    return a + b\n```\ntrailing"
    code = extract_code(text, PROMPT, ENTRY)
    assert "return a + b" in code


def test_extract_code_raw_text_fallback_prepends_prompt():
    text = "    return a + b"  # body only, no def
    code = extract_code(text, PROMPT, ENTRY)
    assert code.startswith(PROMPT.rstrip())
    assert "def add" in code


def test_extract_code_full_def_not_duplicated():
    text = "```python\n" + PROMPT + "    return a + b\n```"
    code = extract_code(text, PROMPT, ENTRY)
    assert code.count("def add") == 1


def test_run_test_passes_correct_solution():
    code = PROMPT + "    return a + b\n"
    test = "def check(f):\n    assert f(1, 2) == 3\n    assert f(-1, 1) == 0\n"
    assert run_test(code, test, ENTRY)


def test_run_test_fails_wrong_solution():
    code = PROMPT + "    return a - b\n"
    test = "def check(f):\n    assert f(1, 2) == 3\n"
    assert not run_test(code, test, ENTRY)


def test_run_test_times_out_infinite_loop():
    code = PROMPT + "    while True:\n        pass\n"
    test = "def check(f):\n    f(1, 2)\n"
    assert not run_test(code, test, ENTRY, timeout=1)


def test_build_problem_includes_instruction_and_prompt():
    p = build_problem(PROMPT)
    assert "python code block" in p
    assert PROMPT in p
