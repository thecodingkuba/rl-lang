from typing import List

from src.env.solvers.base import BaseSolver
from src.env.solvers.vocab_drill import VocabDrill
from src.env.solvers.grammar_explanation import GrammarExplanation
from src.env.solvers.mixed_quiz import MixedQuiz
from src.env.solvers.spaced_repetition import SpacedRepetition
from src.env.solvers.free_form import FreeForm


def build_solver_list() -> List[BaseSolver]:
    """Returns the fixed ordered list of solvers (action index = list position)."""
    return [
        VocabDrill(),           # action 0
        GrammarExplanation(),   # action 1
        MixedQuiz(),            # action 2
        SpacedRepetition(),     # action 3
        FreeForm(),             # action 4
    ]
