from typing import Union, Callable, List
from Istop import istop
from Istop.AirlineAndFlight.istopAirline import IstopAirline
from ModelStructure.Costs.costFunctionDict import CostFuns
from ModelStructure.ScheduleMaker import scheduleMaker
from UDPP import udppModel
from OfferChecker import checkOffer
import torch


class Instance(istop.Istop):
    def __init__(self,  num_flights, num_airlines, reduction_factor=100):

        scheduleTypes = scheduleMaker.schedule_types(show=True)
        # init variables, chedule and cost function
        schedule_df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=scheduleTypes[0])

        self.reductionFactor = reduction_factor
        self.costFun = CostFuns().costFun["realistic"]
        self.flightDict =  CostFuns().flightTypeDict

        # internal optimisation step
        udpp_model_xp = udppModel.UDPPmodel(schedule_df, self.costFun)
        udpp_model_xp.run()

        super().__init__(udpp_model_xp.get_new_df(), self.costFun, triples=True)
        self.offerChecker = checkOffer.OfferChecker(self.scheduleMatrix)

    def check_couple_in_pairs(self, couple):
        return self.offerChecker.check_couple_in_pairs(couple, self.airlines_pairs)

    def check_couple_in_triples(self, couple):
        return self.offerChecker.check_couple_in_triples(couple, self.airlines_triples)

    def all_couples_matches(self):
        return self.offerChecker.all_couples_check(self.airlines_pairs)

    def all_triples_matches(self):
        return self.offerChecker.all_triples_check(self.airlines_triples)

    def get_filtered_schedule(self, airline: IstopAirline):
        return [flight for flight in self.flights if flight.airline != airline]

    def get_schedule_tensor(self) -> torch.tensor:
        mat = torch.zeros((self.numFlights, self.numAirlines + len(self.flightDict.keys()) + 1))
        for i in range(self.numFlights):
            mat[i, self.airDict[self.flights[i].airline.name]] = 1
            mat[i, self.numAirlines + self.flightDict[self.flights[i].type]] = 1
            mat[i, -1] = (self.flights[i].slot.time - self.flights[i].eta)/self.reductionFactor

        return mat
