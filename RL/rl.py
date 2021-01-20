from typing import Union, Callable, List
from Istop import istop
from Istop.AirlineAndFlight import istopAirline as air
import checkOffers


class Rl(istop.Istop):
    def __init__(self, df_init, cost_fun: Union[Callable, List[Callable]], triples=False):
        super().__init__(df_init, cost_fun)

    def check_couple_in_pairs(self, couple, parallel=False):
        return checkOffers.check_couple_in_pairs(self.scheduleMatrix, couple, self.airlines_pairs)

    def check_couple_in_triples(self, couple, parallel=False):
        return checkOffers.check_couple_in_triples(self.scheduleMatrix, couple, self.airlines_triples)

    def get_couple_matches(self):
        checkOffers.run_couples_check(self.scheduleMatrix, self.airlines_pairs)

    def get_triple_matches(self):
        checkOffers.run_triples_check(self.scheduleMatrix, self.airlines_triples)