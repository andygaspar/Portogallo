import torch
import numpy as np
from Istop.AirlineAndFlight.istopAirline import IstopAirline


class Masker:

    def __init__(self, instance):
        self.instance = instance
        self.maskDict = self.instance.offerChecker.all_couples_check(self.instance.airlines_pairs, return_info=True)
        # print(self.maskDict)

        # for key in self.maskDict.keys():
        #     for i in range(len(self.maskDict[key][0])):
        #         print(instance.airlines[])

        self.mask = None
        self.numFlights = self.instance.numFlights
        self.actions = None
        self.airlines = None
        self.trades = None

    def set_initial_mask(self):
        self.actions = []
        self.airlines = []
        self.mask = torch.zeros(self.numFlights)
        for fl in self.maskDict.keys():
            self.mask[fl] = 1
        return self.mask

    def set_action(self, action):
        self.actions.append(action)
        flight = self.instance.flights[action]
        if len(self.actions) == 1:
            self.trades = self.maskDict[flight.slot.index]
        self.airlines.append(flight.airline)
        self.mask = torch.zeros(self.numFlights)
        trades = []
        idxs = []
        found_trade = False
        if len(self.actions) % 2 == 1:
            for trade in self.trades:
                for couple in trade:
                    for fl in couple:
                        if fl.slot.index not in self.actions and fl.airline == self.airlines[-1]:
                            idxs.append(fl.slot.index)
                            found_trade = True
                if found_trade:
                    trades.append(trade)
                    found_trade = False
        else:
            for trade in self.trades:
                for couple in trade:
                    for fl in couple:
                        if fl.slot.index not in self.actions and fl.airline not in self.airlines:
                            idxs.append(fl.slot.index)
                            found_trade = True
                if found_trade:
                    trades.append(trade)
                    found_trade = False

        self.trades = trades
        idxs = np.unique(idxs)
        self.mask[idxs] = 1
        return self.mask


class NoneMasker(Masker):

    def __init__(self, instance):
        super().__init__(instance)

    def set_action(self, action):
        self.mask = torch.ones(self.numFlights)
        self.mask[action] = 0
        return self.mask
