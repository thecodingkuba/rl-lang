from typing import List

from src.env.solvers.base import BaseSolver
from src.env.solvers.vocab_drill import VocabDrill
from src.env.solvers.grammar_explanation import GrammarExplanation
from src.env.solvers.mixed_quiz import MixedQuiz
from src.env.solvers.spaced_repetition import SpacedRepetition
from src.env.solvers.free_form import FreeForm


def build_solver_list() -> List[BaseSolver]:
    return [
        VocabDrill(),
        GrammarExplanation(),
        MixedQuiz(),
        SpacedRepetition(),
        FreeForm(),
    ]
