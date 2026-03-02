from dataclasses import dataclass

@dataclass
class SkillMap():
    """
    Gives mapping between human readable skills and indices used by RL environment.
    """
    # immmutable bidirectional mapping
    
    def __init__(self, skills: list[str]):
        # error checks
        if len(skills) != len(set(skills)):
            raise ValueError("Duplicate skills detected.")
        # index -> skill name map
        index_to_skill = tuple(skills)
        # skill name -> index map
        skill_to_index = {}