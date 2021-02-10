

class Masker:

    def __init__(self, instance):
        all_airlines = [fl.airline for trade in instance.matches for couple in trade for fl in couple]
        self.airlines = []
        #self.airlines = [airline for airline in all_airlines if airline not in self.airlines]
        for airline in all_airlines:
            if airline not in self.airlines:
                self.airlines.append(airline)
        print(self.airlines)

    def mask_airline(self):
        pass
