from metacog.controller import Config
from metacog.eval.humaneval import _parser, build_config, build_problem, extract_code, run_test

PROMPT = 'def add(a, b):\n    """Add two numbers."""\n'
ENTRY = "add"


def _args(*extra):
    args = _parser().parse_args(
        ["humaneval", "--thinker-url", "http://x", "--thinker-model", "m", *extra]
    )
    args.answer_prior = None if args.answer_prior == "none" else float(args.answer_prior)
    return args


def test_build_config_defaults_match_config():
    cfg, ref = build_config(_args()), Config()
    assert cfg.mode == "adaptive"
    for field in (
        "stop_confidence",
        "n_min",
        "n_max",
        "sketch_tokens",
        "expand_max",
        "max_rounds",
        "triage",
        "answer_prior",
        "greedy_anchor",
        "cascade_confidence",
    ):
        assert getattr(cfg, field) == getattr(ref, field), field


def test_build_config_passes_flags_and_answer_prior_none():
    cfg = build_config(
        _args(
            "--mode",
            "stepwise",
            "--stop-confidence",
            "0.9",
            "--n-min",
            "3",
            "--sketch-tokens",
            "200",
            "--triage",
            "0.5",
            "--answer-prior",
            "none",
            "--greedy-anchor",
            "--cascade-confidence",
            "0.8",
        )
    )
    assert cfg.mode == "stepwise"
    assert cfg.stop_confidence == 0.9
    assert cfg.n_min == 3
    assert cfg.sketch_tokens == 200
    assert cfg.triage == 0.5
    assert cfg.answer_prior is None
    assert cfg.greedy_anchor is True
    assert cfg.cascade_confidence == 0.8


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
