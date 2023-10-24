from httomo.utils import Pattern


import os


class Loader:
    """Interface to a loader object (placeholder for now)"""
    
    def __init__(self, in_file: os.PathLike):
        self.pattern: Pattern = Pattern.all
        self.reslice: bool = False

    def load(self, pad: int = 0):
        pass