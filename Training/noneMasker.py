import torch
import numpy as np
from Istop.AirlineAndFlight.istopAirline import IstopAirline
from Training.masker import Masker


class NoneMasker(Masker):

    def __init__(self, instance):
        super().__init__(instance)


    def initial_air_mask(self):
        return torch.ones(self.instance.numAirlines)


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
