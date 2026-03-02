from dataclasses import dataclass

@dataclass
class EnvConfig:
    """
    Stores fixed experiment parameters for the tutoring environment.

    This configuration defines the skill set and hyperparameters that control
    learner initialization, learning dynamics, and episode limits. It is created
    once and passed to the environment and learner as read-only settings.

    Note: does not store any learner state or changing data (e.g., mastery or history).
    """
    
    # list of 10 skills, each to be instantiated with a mastery
    skills: list[str] = [
        skills = [
    # grammar
    "grammar:present",
    "grammar:past",
    "grammar:future",
    "grammar:articles",
    "grammar:prepositions",
    # vocab (words or topics)
    "vocab:1",
    "vocab:2",
    "vocab:3",
    "vocab:4",
    "vocab:5",
]
    
    # starting mastery level for each skill where mastery [0, 1]
    init_mastery: float = 0.25

    # strength update size after observation
    learning_rate: float = 0.10
    
    # forgetting decay rate
    decay_rate: float = 0.1
    
    # min and max mastery
    min_mastery: float = 0
    max_mastery: float = 1
    
    # interaction cycles in one tutoring session
    episode_length: int = 200 