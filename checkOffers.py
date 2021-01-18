import numpy as np
import copy
import concurrent.futures


def recursive_calls(flight, flights, free, initial_costs,  airlines, comb):
    for other_flight in free:
        if other_flight != flight:
            if flight.eta <= other_flight.slot.time:
                fl = copy.copy(flights)
                fl.remove(flight)
                new_free = copy.copy(free)
                new_free.remove(other_flight)
                air = copy.copy(airlines)
                c = copy.deepcopy(comb)
                c[flight.airline.name].append([flight, other_flight.slot])
                if flight.airline.name != other_flight.airline.name:
                    air.append(flight.airline.name)

                still_convenient = True

                for airline in c.keys():
                    cost = sum([c[airline][i][0].costFun(c[airline][i][0], c[airline][i][1])
                                for i in range(len(c[airline]))])

                    if cost >= initial_costs[airline]:
                        still_convenient = False
                        break

                if still_convenient:
                    if len(fl) > 0:
                        if recursive_calls(fl[0], fl, new_free, initial_costs, air, c):
                            return True

                    elif len(np.unique(air)) == len(initial_costs.keys()):
                        return True
    return False


def get_combinations(flights, initial_costs):
    combs = dict(zip(np.unique([flight.airline.name for flight in flights]).tolist(),
                     [[] for i in range(int(len(flights ) /2))]))
    free = copy.copy(flights)
    return recursive_calls(flights[0], flights, free, initial_costs, [], combs)


def condition(pairs_list):

    airlines = [pair[0].airline.name for pair in pairs_list]
    initial_costs = dict(zip(airlines,
                             np.array(
                                 [pair[0].costFun(pair[0], pair[0].slot) + pair[1].costFun(pair[1], pair[1].slot)
                                  for
                                  pair in pairs_list])))

    flights = [flight for pair in pairs_list for flight in pair]
    return get_combinations(flights, initial_costs)

def check_couples(airl_pair, matches):
    fl_pair_a = airl_pair[0].flight_pairs
    fl_pair_b = airl_pair[1].flight_pairs
    for pairA in fl_pair_a:
        for pairB in fl_pair_b:
            if condition([pairA, pairB]):
                #matches.append([pairA, pairB])
                print("trovata")

def run_check(airlines_pairs, matches, parallel=False):
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as exe:
            exe.map(check_couples, airlines_pairs)
    else:
        pass
