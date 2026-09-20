"""Stepwise metacog against a local vLLM thinker and a local reflex judge.

vllm serve openbmb/MiniCPM4-8B --port 8000
reflex-serve --model Qwen/Qwen3-4B --port 8008
python examples/quickstart.py
"""

from metacog import Config, MetaCog, OpenAICompatThinker, SystemOneJudge

thinker = OpenAICompatThinker(
    base_url="http://localhost:8000",
    model="openbmb/MiniCPM4-8B",
    api="completions",  # use "chat" if your server supports assistant prefill
)
judge = SystemOneJudge.reflex()  # http://localhost:8008, permutations=2

mc = MetaCog(
    thinker,
    judge,
    Config(mode="stepwise", n_paths=4, step_tokens=256, max_steps=6, temperature=0.8),
)

result = mc.run(
    "A train travels 120 km in 1.5 hours, then 80 km in 45 minutes. "
    "What is its average speed for the whole journey, in km/h?"
)

for rnd in result.trace.rounds:
    probs = ", ".join(f"{p:.2f}" for p in rnd.verdict.probabilities)
    print(f"step {rnd.step}: {len(rnd.candidates)} candidates, P = [{probs}]")
print("\nANSWER:\n" + result.answer)
