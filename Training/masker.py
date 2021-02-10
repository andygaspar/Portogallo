import torch

class Masker:

    def __init__(self, instance):
        self.instance = instance
        self.airlines = self.get_airlines_from_matches(instance.matches)
        self.airMask = self.mask_airline()
        print(self.airlines)

    def get_airlines_from_matches(self, matches):
        all_airlines = [fl.airline for trade in matches for couple in trade for fl in couple]
        airlines_in_matches = []
        # self.airlines = [airline for airline in all_airlines if airline not in self.airlines]
        for airline in all_airlines:
            if airline not in all_airlines:
                airlines_in_matches.append(airline)
        return airlines_in_matches

    def mask_airline(self):

        mask = torch.zeros(self.instance.numAirlines)
        for airline in self.airlines:
            mask[airline.index] = 1
        return mask

    def airAction(self, airline_action):
        self.airlines.remove(airline_action)

