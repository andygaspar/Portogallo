from typing import Union, Callable, List
from Istop import istop
from UDPP import udppModel
from Istop.AirlineAndFlight import istopAirline as air
import checkOffers


class Rl(istop.Istop):
    def __init__(self, df_init, cost_fun: Union[Callable, List[Callable]], triples=False):
        # internal optimisation step
        udpp_model_xp = udppModel.UDPPmodel(df_init, cost_fun)
        udpp_model_xp.run()

        super().__init__(udpp_model_xp.get_new_df(), cost_fun)

    def check_couple_in_pairs(self, couple, parallel=False):
        return checkOffers.check_couple_in_pairs(self.scheduleMatrix, couple, self.airlines_pairs)

    def check_couple_in_triples(self, couple, parallel=False):
        return checkOffers.check_couple_in_triples(self.scheduleMatrix, couple, self.airlines_triples)

    def get_couple_matches(self):
        return checkOffers.run_couples_check(self.scheduleMatrix, self.airlines_pairs)

    def get_triple_matches(self):
        return checkOffers.run_triples_check(self.scheduleMatrix, self.airlines_triples)
