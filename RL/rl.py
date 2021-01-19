from typing import Union, Callable, List
from ModelStructure import modelStructure as ms


class Rl(ms.ModelStructure):
    def __init__(self, df_init, cost_fun: Union[Callable, List[Callable]], triples=False):
        super().__init__(df_init, cost_fun)
