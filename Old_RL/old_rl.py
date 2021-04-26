
from typing import Union, Callable, List
from Istop import istop
from Istop.AirlineAndFlight.istopAirline import IstopAirline
from UDPP import udppModel
from OfferChecker import checkOffer
from ModelStructure.Costs.costFunctionDict import CostFuns


class Rl(istop.Istop):
    def __init__(self, df_init, cost_fun: Union[Callable, List[Callable]], triples=True, parallel=False, private=False):
        # internal optimisation step
        udpp_model_xp = udppModel.UDPPmodel(df_init, cost_fun)
        udpp_model_xp.run()
        self.udppDf = udpp_model_xp.get_new_df()
        super().__init__(udpp_model_xp.get_new_df(), cost_fun, triples=triples)
        types = df_init["type"].unique()
        self.flightTypeDict = CostFuns().flightTypeDict
        self.offerChecker = checkOffer.OfferChecker(self.scheduleMatrix, parallel, private)

    def check_couple_in_pairs(self, couple):
        return self.offerChecker.check_couple_in_pairs(couple, self.airlines_pairs)

    def check_couple_in_triples(self, couple):
        return self.offerChecker.check_couple_in_triples(couple, self.airlines_triples)

    def all_couples_matches(self):
        return self.offerChecker.all_couples_check(self.airlines_pairs)

    def all_triples_matches(self):
        return self.offerChecker.all_triples_check(self.airlines_triples)

    def all_triples_matches_fast(self):
        return self.offerChecker.all_triples_check_fast(self.airlines_triples)

    def get_filtered_schedule(self, airline: IstopAirline):
        return [flight for flight in self.flights if flight.airline != airline]

