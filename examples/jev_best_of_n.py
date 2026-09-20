"""Best-of-N metacog: any OpenAI-compatible thinker + the TypeSafe Jev judge.

export TYPESAFE_API_KEY=...
python examples/jev_best_of_n.py
"""

from metacog import Config, MetaCog, OpenAICompatThinker, SystemOneJudge

thinker = OpenAICompatThinker(
    base_url="https://api.openai.com",  # or vLLM / llama.cpp / Ollama
    model="gpt-4o-mini",
    api="chat",
)
judge = SystemOneJudge.jev()  # https://api.typesafe.ai, key from TYPESAFE_API_KEY

mc = MetaCog(thinker, judge, Config(mode="best_of_n", n_paths=4, temperature=0.9))

result = mc.run("Write a one-line Python function that checks whether a string is a palindrome.")
print(result.answer)
print(f"\nconfidence: {result.trace.rounds[0].verdict.confidence:.2f}")
