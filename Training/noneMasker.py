import torch
import numpy as np
from Istop.AirlineAndFlight.istopAirline import IstopAirline
from Training.masker import Masker


class NoneMasker(Masker):

    def __init__(self, instance):
        super().__init__(instance)
        self.airMask = self.initial_air_mask()

    def initial_air_mask(self):
        return torch.ones(self.instance.numAirlines)

    def air_action(self, air_idx):

        self.airAction = self.instance.airlines[air_idx]
        self.flMask = torch.ones(len(self.airAction.flight_pairs))

    def fl_action(self, fl_idx):
        pass

    def reset(self):
        self.airMask = self.initial_air_mask()
        self.airAction = []
        self.flAction = None
