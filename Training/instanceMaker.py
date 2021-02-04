from typing import Union, Callable, List
import xpress as xp
from GlobalFuns.globalFuns import HiddenPrints
from Istop import istop
from Istop.AirlineAndFlight.istopAirline import IstopAirline
from ModelStructure.Costs.costFunctionDict import CostFuns
from ModelStructure.ScheduleMaker import scheduleMaker
from UDPP import udppModel
from OfferChecker import checkOffer
import torch


class Instance(istop.Istop):
    def __init__(self, num_flights=50, num_airlines=5, triples=True,
                 reduction_factor=100, custom_schedule=None, df=None, xp_problem=None):
        scheduleTypes = scheduleMaker.schedule_types(show=False)
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


        if xp_problem is None:
            self.xp_problem = xp.problem()
        else:
            self.xp_problem = xp_problem
        with HiddenPrints():
            self.xp_problem.reset()


        # internal optimisation step
        udpp_model_xp = udppModel.UDPPmodel(schedule_df, self.costFun, self.xp_problem)
        udpp_model_xp.run()

        with HiddenPrints():
            self.xp_problem.reset()

        super().__init__(udpp_model_xp.get_new_df(), self.costFun, triples=triples, xp_problem=self.xp_problem)
        self.offerChecker = checkOffer.OfferChecker(self.scheduleMatrix)
        self.reverseAirDict = dict(zip(list(self.airDict.keys()), list(self.airDict.values())))



    def set_matches(self, matches: torch.tensor, num_trades, single_trade_len):
        for i in range(num_trades):
            start = i*single_trade_len
            airline_1 = self.airlines[torch.argmax(matches[start:start+4]).item()]
            couple_1 = airline_1.flight_pairs[torch.argmax(matches[start+4 :start+14])]
            airline_2 = self.airlines[torch.argmax(matches[start+ 14:start + 18])]
            couple_2 = airline_2.flight_pairs[torch.argmax(matches[start + 18:start + 28])]
            self.matches.append([couple_1, couple_2])
        self.preprocessed = True
        return

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
