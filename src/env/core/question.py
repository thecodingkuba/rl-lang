from dataclasses import dataclass

@dataclass
class Question():
    """Represents concrete instance of a question."""
    difficulty: int
    learn_rate: float
    slip_rate: float
    guess_rate: float
    skills_tested = np
    difficulty: float # (0, 1) because 0 needs know knowledge and 1 needs perfect mastery
    fn_rate: float
    fp_rate: float
    
    @classmethod
    def random()