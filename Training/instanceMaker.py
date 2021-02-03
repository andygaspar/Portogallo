from typing import Union, Callable, List
from Istop import istop
from Istop.AirlineAndFlight.istopAirline import IstopAirline
from ModelStructure.Costs.costFunctionDict import CostFuns
from ModelStructure.ScheduleMaker import scheduleMaker
from UDPP import udppModel
from OfferChecker import checkOffer
import torch


class Instance(istop.Istop):
    def __init__(self, num_flights=50, num_airlines=5, triples=True,
                 reduction_factor=100, custom_schedule=None, df=None):

        scheduleTypes = scheduleMaker.schedule_types(show=True)
        # init variables, schedule and cost function
        if custom_schedule is None and df is None:
            schedule_df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=scheduleTypes[0])
        else:
            if df is None:
                schedule_df = scheduleMaker.df_maker(custom=custom_schedule)
            else:
                schedule_df = df

        self.reductionFactor = reduction_factor
        self.costFun = CostFuns().costFun["realistic"]
        self.flightTypeDict = CostFuns().flightTypeDict
        self.reverseAirDict = dict(zip(list(self.airDict.keys()), list(self.airDict.values())))

        # internal optimisation step
        udpp_model_xp = udppModel.UDPPmodel(schedule_df, self.costFun)
        udpp_model_xp.run()

        super().__init__(udpp_model_xp.get_new_df(), self.costFun, triples=triples)
        self.offerChecker = checkOffer.OfferChecker(self.scheduleMatrix)

    def get_matches(self, matches=None):
        if matches is None:
            super().get_matches()
        else:
            print("pippimo")

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
        schedule_tensor = torch.zeros((self.numFlights, 2 + self.numAirlines + len(self.flightTypeDict.keys())))
        for i in range(self.numFlights):
            schedule_tensor[i, self.airDict[self.flights[i].airline.name]] = 1
            schedule_tensor[i, self.numAirlines + self.flightTypeDict[self.flights[i].type]] = 1
            schedule_tensor[i, -2] = self.flights[i].slot.time / self.reductionFactor
            schedule_tensor[i, -1] = self.flights[i].eta / self.reductionFactor
        return schedule_tensor.flatten()
