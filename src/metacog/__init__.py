"""metacog: a model-agnostic metacognition loop (thinker + System-One judge)."""

from .controller import Config, MetaCog
from .judge import (
    DEFAULT_CHOOSE_INSTRUCTIONS,
    DEFAULT_SCORE_INSTRUCTIONS,
    FINISHED_QUESTIONS,
    Judge,
    JudgeError,
    SystemOneJudge,
)
from .split import split_paths
from .thinker import OpenAICompatThinker, Thinker, ThinkerError, TransformersThinker
from .types import Candidate, Generation, Result, Round, Trace, Verdict

__version__ = "0.1.0"

__all__ = [
    "MetaCog",
    "Config",
    "Result",
    "Trace",
    "Round",
    "Candidate",
    "Generation",
    "Verdict",
    "Thinker",
    "Judge",
    "OpenAICompatThinker",
    "TransformersThinker",
    "SystemOneJudge",
    "JudgeError",
    "ThinkerError",
    "split_paths",
    "DEFAULT_CHOOSE_INSTRUCTIONS",
    "DEFAULT_SCORE_INSTRUCTIONS",
    "FINISHED_QUESTIONS",
    "__version__",
]
