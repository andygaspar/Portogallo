import string
from typing import Union, Callable, List

import pandas as pd

from GlobalFuns.globalFuns import HiddenPrints
from ModelStructure.modelStructure import ModelStructure
from UDPP.LocalOptimised.udppLocalOpt import UDPPlocalOpt
from ModelStructure.Solution import solution
from UDPP.AirlineAndFlightAndSlot.udppAirline import UDPPairline
from UDPP.AirlineAndFlightAndSlot.udppFlight import UDPPflight
from ModelStructure.Slot.slot import Slot
import time

import ModelStructure.modelStructure as ms


class UDPPmodel(ModelStructure):

    def __init__(self, df_init: pd.DataFrame, costFun: Union[Callable, List[Callable]], xp_problem=None):

        super().__init__(df_init=df_init, costFun=costFun, airline_ctor=UDPPairline)
        self.xp_problem = xp_problem

    def run(self):
        airline: UDPPairline
        start = time.time()
        for airline in self.airlines:
            with HiddenPrints():
                UDPPlocalOpt(airline, self.slots, self.xp_problem)
        # print(time.time() - start)
        solution.make_solution(self)

    def get_new_df(self):
        self.df: pd.DataFrame
        new_df = self.solution.copy(deep=True)
        new_df.reset_index(drop=True, inplace=True)
        new_df["slot"] = new_df["new slot"]
        new_df["fpfs"] = new_df["new arrival"]
        return new_df

