import torch
import numpy as np
from Istop.AirlineAndFlight.istopAirline import IstopAirline


class Masker:

    def __init__(self, instance, triples):
        self.instance = instance
        if not triples:
            self.maskDict, self.tradeDict = \
                self.instance.offerChecker.all_couples_check(self.instance.airlines_pairs, return_info=True)
        else:
            self.maskDict, self.tradeDict = \
                self.instance.offerChecker.all_triples_check(self.instance.airlines_triples, return_info=True)
        # print(self.maskDict)

        # for key in self.maskDict.keys():
        #     for i in range(len(self.maskDict[key][0])):
        #         print(instance.airlines[])

        self.mask = None
        self.numFlights = self.instance.numFlights
        self.actions = None
        self.airlines = None
        self.trades = None
        self.selectedTrade = None

    def set_initial_mask(self):
        # print("\n",self.selectedTrade)
        # <>
        # trade_indexes = self.tradeDict[str(self.selectedTrade)]
        # # print(trade_indexes)
        # for i in range(len(trade_indexes[0])):
        #     # print("ciao ", self.maskDict[trade_indexes[0][i]][trade_indexes[1][i]])
        #     self.maskDict[trade_indexes[0][i]][trade_indexes[1][i]] = []
        #     non_empty = sum([1 if len(trade) > 0 else 0 for trade in self.maskDict[trade_indexes[0][i]]])
        #     if non_empty == 0:
        #         del self.maskDict[trade_indexes[0][i]]

        keys_to_delete = []
        if self.actions is not None:
            for fl_index in self.actions:
                if fl_index in self.maskDict.keys():
                    del self.maskDict[fl_index]

            for fl_index in self.actions:
                for key in self.maskDict.keys():
                    new_trades = []
                    for trade in self.maskDict[key]:
                        found = False
                        for couple in trade:
                            for fl in couple:
                                if fl_index == fl.slot.index:
                                    found = True
                        if not found:
                            new_trades.append(trade)

                    self.maskDict[key] = new_trades
                    if len(new_trades) == 0:
                        keys_to_delete.append(key)

            for key in np.unique(keys_to_delete):
                del self.maskDict[key]

        if len(self.maskDict) == 0:
            self.mask = None
            return

        self.actions = []
        self.airlines = []
        self.mask = torch.zeros(self.numFlights)
        # for key in self.maskDict.keys():
        #     print(self.instance.flights[key], self.maskDict[key])

        for fl in self.maskDict.keys():
            self.mask[fl] = 1


    def set_action(self, action):
        self.actions.append(action)
        flight = self.instance.flights[action]
        if len(self.actions) == 1:
            self.trades = self.maskDict[flight.slot.index]
        self.airlines.append(flight.airline)
        self.mask = torch.zeros(self.numFlights)
        trades = []
        idxs = []

        if len(self.actions) % 2 == 1:
            for trade in self.trades:
                if len(trade) > 0:
                    found_trade, new_idxs = self.check_intra_trade(trade)
                    if found_trade:
                        trades.append(trade)
                        idxs += new_idxs
        else:
            for trade in self.trades:
                if len(trade) > 0:
                    found_trade, new_idxs = self.check_inter_trade(trade)
                    if found_trade:
                        trades.append(trade)
                        idxs += new_idxs

        if len(trades) == 0:
            self.selectedTrade = self.trades[0]
        self.trades = trades
        idxs = np.unique(idxs)
        self.mask[idxs] = 1


    def check_intra_trade(self, trade):
        idxs = []
        found_trade = False
        valid_trade = False
        for couple in trade:
            for fl in couple:
                if fl.slot.index not in self.actions and fl.airline == self.airlines[-1]:
                    idxs.append(fl.slot.index)
                    found_trade = True
                if fl.slot.index == self.actions[-1]:
                    valid_trade = True
        if found_trade and valid_trade:
            return True, idxs
        else:
            return False, None

    def check_inter_trade(self, trade):
        idxs = []
        found_trade = False
        valid_trade = False
        for couple in trade:
            for fl in couple:
                if fl.slot.index not in self.actions and fl.airline not in self.airlines:
                    idxs.append(fl.slot.index)
                    found_trade = True
                if fl.slot.index == self.actions[-1]:
                    valid_trade = True
        if found_trade and valid_trade:
            return True, idxs
        else:
            return False, None


class NoneMasker(Masker):

    def __init__(self, instance):
        super().__init__(instance)

    def set_action(self, action):
        self.mask = torch.ones(self.numFlights)
        self.mask[action] = 0
        return self.mask
