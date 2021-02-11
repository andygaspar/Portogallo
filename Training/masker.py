import torch
import numpy as np
from Istop.AirlineAndFlight.istopAirline import IstopAirline


class Masker:

    def __init__(self, instance):
        self.instance = instance
        self.maskDict = self.instance.offerChecker.all_couples_check(self.instance.airlines_pairs, return_info=True)
        self.airMask = self.initial_air_mask()
        self.flMask = None
        self.airAction = None
        self.flAction = None

    def initial_air_mask(self):
        mask = torch.zeros(self.instance.numAirlines)
        for airline in self.maskDict.keys():
            if len(self.maskDict[airline][0]) > 0:
                mask[airline.index] = 1
        return mask

    def air_action(self, air_idx):
        if self.airAction is None:
            airline_action: IstopAirline
            self.airAction = self.instance.airlines[air_idx]
            self.flMask = torch.zeros(len(self.airAction.flight_pairs))
            for i in np.unique(self.maskDict[self.airAction][1]):
                self.flMask[i] = 1
        else:
            fl_idx = [self.maskDict[self.airAction][2][i] for i in range(len(self.maskDict[self.airAction][0]))
                      if self.maskDict[self.airAction][0][i] == air_idx and
                      self.maskDict[self.airAction][1][i] == self.flAction]

            self.flMask = torch.zeros(len(self.airAction.flight_pairs))
            for i in np.unique(fl_idx):
                self.flMask[i] = 1

    def fl_action(self, fl_idx):
        if self.flAction is None:
            airlines_idx = [self.maskDict[self.airAction][0][i] for i in range(len(self.maskDict[self.airAction][0]))
                            if self.maskDict[self.airAction][1][i] == fl_idx]
            self.airMask = torch.zeros(self.instance.numAirlines)
            for i in np.unique(airlines_idx):
                self.airMask[i] = 1
            self.flAction = fl_idx

    def reset(self):
        self.airMask = self.initial_air_mask()
        self.airAction = None
        self.flAction = None
