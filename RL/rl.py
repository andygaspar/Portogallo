from typing import Union, Callable, List
from Istop import istop
from Istop.AirlineAndFlight.istopAirline import IstopAirline
from UDPP import udppModel
from OfferChecker import checkOffers


class Rl(istop.Istop):
    def __init__(self, df_init, cost_fun: Union[Callable, List[Callable]], triples=False):

        # internal optimisation step
        udpp_model_xp = udppModel.UDPPmodel(df_init, cost_fun)
        udpp_model_xp.run()

        super().__init__(udpp_model_xp.get_new_df(), cost_fun, triples)
        types = df_init["type"].unique()
        self.flightTypeDict = dict(zip(types, range(len(types))))

    def check_couple_in_pairs(self, couple):
        return checkOffers.check_couple_in_pairs(self.scheduleMatrix, couple, self.airlines_pairs)

    def check_couple_in_triples(self, couple):
        return checkOffers.check_couple_in_triples(self.scheduleMatrix, couple, self.airlines_triples)

    def all_couples_matches(self):
        return checkOffers.all_couples_check(self.scheduleMatrix, self.airlines_pairs)

    def all_triples_matches(self):
        return checkOffers.all_triples_check(self.scheduleMatrix, self.airlines_triples)

    def get_filtered_schedule(self, airline: IstopAirline):
        return [flight for flight in self.flights if flight.airline != airline]


