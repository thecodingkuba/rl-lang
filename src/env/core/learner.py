import numpy as np

from config.env_config import EnvConfig


class Learner():
    """Represents a single language learner."""
    
    def __init__(self, config: EnvConfig):
        # store configs for methods
        self.config = config
        
        # num of skills
        self.K: int = len(self.config.skills)
        
        
        # HIDDEN TRUE STATE: mastery vector of learner
        self.mastery = np.full(self.wK, self.config.init_mastery) # knowledge (factors ability and difficulty)
        se.f # uncertainty/exploration tendency
        
    def step(self,):
        """Performs an entire interaction in environment:
        Generate observation -> update learner mastery -> returns resulting reward
        """
        plug other functions ez
        
    def _observe(self, question_type: "Question"):
        """Generates whether a learner got a question right.
        mastery and question type information -> correct/incorrect
        """"
        basically just use a sgigmoid
        
    def transition(self,:):
        """Updates learner internal mastery, applying learning and forgetting.
        
        """
        