import torch
import numpy as np
from Istop.AirlineAndFlight.istopAirline import IstopAirline


class Masker:

    def __init__(self, instance):
        self.instance = instance
        self.maskDict = self.instance.offerChecker.all_couples_check(self.instance.airlines_pairs, return_info=True)
        self.airMask = self.initial_mask()
        self.flMask = None
        self.airAction = []
        self.flAction = None

    def initial_mask(self):
        mask = torch.zeros(self.instance.numFlights)
        for airline in self.maskDict.keys():
            if len(self.maskDict[airline][0]) > 0:
                mask[airline.index] = 1
        return mask

    def air_action(self, air_idx):
        if len(self.airAction) == 0:
            airline_action: IstopAirline
            self.airAction.append(self.instance.airlines[air_idx])
            fl_idxs = self.maskDict[self.airAction[-1]][1]
        else:
            fl_idxs = [self.maskDict[self.airAction[-1]][2][i] for i in range(len(self.maskDict[self.airAction[-1]][0]))
                       if self.maskDict[self.airAction[-1]][0][i] == air_idx and
                       self.maskDict[self.airAction[-1]][1][i] == self.flAction]
        self.flMask = torch.zeros(len(self.airAction[-1].flight_pairs))
        for i in np.unique(fl_idxs):
            self.flMask[i] = 1

    def fl_action(self, fl_idx):
        if self.flAction is None:
            airlines_idxs = [self.maskDict[self.airAction[-1]][0][i] for i in range(len(self.maskDict[self.airAction[-1]][0]))
                             if self.maskDict[self.airAction[-1]][1][i] == fl_idx]
            self.airMask = torch.zeros(self.instance.numAirlines)
            for i in np.unique(airlines_idxs):
                self.airMask[i] = 1
            self.flAction = fl_idx

        else:
            pass # to implement in case of triples

    def reset(self):
        self.airMask = self.initial_air_mask()
        self.airAction = []
        self.flAction = None
