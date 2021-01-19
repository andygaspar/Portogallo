import numpy as np
import copy
import concurrent.futures
from multiprocessing import Pool, RawArray, Array
from itertools import combinations
import time


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
                     [[] for i in range(int(len(flights)/2))]))
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

def check_couples(airl_pair):
    matches = []
    fl_pair_a = airl_pair[0].flight_pairs
    fl_pair_b = airl_pair[1].flight_pairs
    for pairA in fl_pair_a:
        for pairB in fl_pair_b:
            if condition([pairA, pairB]):
                matches.append([pairA, pairB])
    return matches

def comb(airl_pair):
    matches = []
    fl_pair_a = airl_pair[0].flight_pairs
    fl_pair_b = airl_pair[1].flight_pairs
    for pairA in fl_pair_a:
        for pairB in fl_pair_b:

            if condition([pairA, pairB]):
                matches.append([pairA, pairB])
    return matches


def run_check(flights, airlines_pairs, air_dict, parallel=False):
    arr =  []
    for flight in flights:
        arr.append([flight.slot.time]+[flight.eta]+[air_dict[flight.airline.name]]+flight.costs)
    arr = np.array(arr)
    arr = [l for sublist in arr for l in sublist]
    data = Array('d', len(arr))
    mat = np.frombuffer(data)

    mat.reshape(arr.shape)
    print(mat)
    # d = copy.deepcopy(airlines_pairs[0][0])
    # if parallel:
    #     with concurrent.futures.ProcessPoolExecutor() as exe:
    #         results = [exe.submit(check_couples, air_pair) for air_pair in airlines_pairs]
    #         print(len([match for f in concurrent.futures.as_completed(results)
    #                    for match in f.result() if len(f.result()) > 0]))
    # else:
    #     pass
    t = time.perf_counter()
    print([comb(i) for i in range(20)])
    print("seq ", time.perf_counter() -t )

    t = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as exe:
        results = [exe.submit(comb, i) for i in range(20)]

        print([f.result() for f in concurrent.futures.as_completed(results)])
    print("par", time.perf_counter()-t)


