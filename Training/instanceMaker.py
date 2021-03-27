from typing import Union, Callable, List
import xpress as xp
from GlobalFuns.globalFuns import HiddenPrints
from Istop import istop
from Istop.AirlineAndFlight.istopAirline import IstopAirline
from Istop.AirlineAndFlight.istopFlight import IstopFlight
from ModelStructure.Costs.costFunctionDict import CostFuns
from ModelStructure.ScheduleMaker import scheduleMaker
from UDPP import udppModel
from OfferChecker import checkOffer
import torch
import numpy as np


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
        flights = [0 for i in range(self.numFlights)]

        for flight in self.flights:
            flights[flight.slot.index] = flight
        self.flights = flights


        self.offerChecker = checkOffer.OfferChecker(self.scheduleMatrix)
        _, self.matches_vect = self.offerChecker.all_couples_check(self.airlines_pairs)
        self.reverseAirDict = dict(zip(list(self.airDict.keys()), list(self.airDict.values())))

    def set_matches(self, matches, num_trades, triples=False):
        self.matches = []
        size = 4 if not triples else 6
        matches = np.array(matches).reshape(num_trades, size)
        for i in range(num_trades):
            if not triples:
                flights = [self.flights[j] for j in matches[i]]
                couple_1 = [flight for flight in flights if flight.airline == flights[0].airline]
                couple_2 = [flight for flight in flights if flight.airline != couple_1[0].airline]
                self.matches.append([couple_1, couple_2])
            else:
                flights = [self.flights[j] for j in matches[i]]
                couple_1 = [flight for flight in flights if flight.airline == flights[0].airline]

                flights = [flight for flight in flights if flight.airline != couple_1[0].airline]
                couple_2 = [flight for flight in flights if flight.airline == flights[0].airline]

                couple_3 = [flight for flight in flights if flight.airline != couple_2[0].airline]
                self.matches.append([couple_1, couple_2, couple_3])

        for match in self.matches:
            for couple in match:
                if not self.is_in(couple, self.couples):
                    self.couples.append(couple)
                    if not self.f_in_matched(couple[0]):
                        self.flights_in_matches.append(couple[0])
                    if not self.f_in_matched(couple[1]):
                        self.flights_in_matches.append(couple[1])

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
        flights: List[IstopFlight]
        flights = self.flights
        schedule_tensor = torch.zeros((self.numFlights, self.numAirlines + self.numFlights))
        for i in range(self.numFlights):
            schedule_tensor[i, self.flights[i].airline.index] = 1
            schedule_tensor[i, -self.numFlights:] = torch.tensor(flights[i].costVect)
        return schedule_tensor.flatten()
