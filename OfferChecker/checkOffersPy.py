import numpy as np
import copy
import concurrent.futures
from multiprocessing import Pool, Array
from itertools import combinations, permutations
import time

# precomputations

couples = list(permutations([0, 1, 2, 3]))
couples_copy = copy.copy(couples)
for c in couples_copy:
    if (c[0] == 0 and c[1] == 1) or (c[0] == 1 and c[1] == 0):
        couples.remove(c)
couples = np.array(couples)

triples = list(permutations([0, 1, 2, 3, 4, 5]))
triples_copy = copy.copy(triples)
for t in triples_copy:
    if ((t[0] == 0 and t[1] == 1) or (t[0] == 1 and t[1] == 0)) or \
            ((t[2] == 2 and t[3] == 3) or (t[2] == 3 and t[3] == 2)) or \
            ((t[4] == 4 and t[5] == 5) or (t[4] == 5 and t[5] == 4)):
        triples.remove(t)
triples = np.array(triples)


def check_couple_condition(mat, flights):
    for c in couples:
        # first airline eta check
        if mat[flights[0], 1] <= mat[flights[c[0]], 0]:
            if mat[flights[1], 1] <= mat[flights[c[1]], 0]:

                # check first airline's convenience
                if mat[flights[0], 2 + flights[0]] + mat[flights[1], 2 + flights[1]] > \
                        mat[flights[0], 2 + flights[c[0]]] + mat[flights[1], 2 + flights[c[1]]]:

                    # second airline eta check
                    if mat[flights[2], 1] <= mat[flights[c[2]], 0]:
                        if mat[flights[3], 1] <= mat[flights[c[3]], 0]:
                            if mat[flights[2], 2 + flights[2]] + mat[flights[3], 2 + flights[3]] > \
                                    mat[flights[2], 2 + flights[c[2]]] + \
                                    mat[flights[3], 2 + flights[c[3]]]:
                                return True
    return False


def air_couple_check(mat, airl_pair):
    matches = []
    fl_pair_a = airl_pair[0].flight_pairs
    fl_pair_b = airl_pair[1].flight_pairs
    for pairA in fl_pair_a:
        for pairB in fl_pair_b:
            if check_couple_condition(mat, [fl.slot.index for fl in pairA] + [fl.slot.index for fl in pairB]):
                matches.append([pairA, pairB])
    return matches


def all_couples_check(mat, airlines_pairs):
    matches = []
    for air_pair in airlines_pairs:
        match = air_couple_check(mat, air_pair)
        if len(match) > 0:
            matches += match
    return matches


def check_triple_condition(mat, flights):
    matches = []
    for t in triples:

        # first airline eta check
        if mat[flights[0], 1] <= mat[flights[t[0]], 0]:
            if mat[flights[1], 1] <= mat[flights[t[1]], 0]:

                # check first airline's convenience
                if mat[flights[0], 2 + flights[0]] + mat[flights[1], 2 + flights[1]] > \
                        mat[flights[0], 2 + flights[t[0]]] + mat[flights[1], 2 + flights[t[1]]]:

                    # second airline eta check
                    if mat[flights[2], 1] <= mat[flights[t[2]], 0]:
                        if mat[flights[3], 1] <= mat[flights[t[3]], 0]:

                            # second convenience check
                            if mat[flights[2], 2 + flights[2]] + mat[flights[3], 2 + flights[3]] > \
                                    mat[flights[2], 2 + flights[t[2]]] + \
                                    mat[flights[3], 2 + flights[t[3]]]:

                                # third airline eta check
                                if mat[flights[4], 1] <= mat[flights[t[4]], 0]:
                                    if mat[flights[5], 1] <= mat[flights[t[5]], 0]:

                                        # third convenience check
                                        if mat[flights[4], 2 + flights[4]] + mat[flights[5], 2 + flights[5]] > \
                                                mat[flights[4], 2 + flights[t[4]]] + \
                                                mat[flights[5], 2 + flights[t[5]]]:
                                            # print(flights)
                                            return True
    return False


def air_triple_check(mat, airl_pair):
    matches = []
    fl_pair_a = airl_pair[0].flight_pairs
    fl_pair_b = airl_pair[1].flight_pairs
    fl_pair_c = airl_pair[2].flight_pairs
    for pairA in fl_pair_a:
        for pairB in fl_pair_b:
            for pairC in fl_pair_c:

                if check_triple_condition(mat, [fl.slot.index for fl in pairA] + [fl.slot.index for fl in pairB] +
                                               [fl.slot.index for fl in pairC]):
                    matches.append([pairA, pairB, pairC])
    return matches


def all_triples_check(mat, airlines_triples):
    matches = []

    for air_triple in airlines_triples:
        match = air_triple_check(mat, air_triple)
        if len(match) > 0:
            matches += match
    return matches


def check_couple_in_pairs(mat, couple, airlines_pairs):
    matches = []
    other_airline = None

    for air_pair in airlines_pairs:
        if couple[0].airline.name == air_pair[0].name:
            other_airline = air_pair[1]
        elif couple[0].airline.name == air_pair[1].name:
            other_airline = air_pair[0]

        if other_airline is not None:
            for pair in other_airline.flight_pairs:
                if check_couple_condition(mat, [fl.slot.index for fl in couple] + [fl.slot.index for fl in pair]):
                    matches.append([couple, pair])

    return matches


def check_couple_in_triples(mat, couple, airlines_triples):
    matches = []
    other_airline_A = None
    other_airline_B = None

    for air_pair in airlines_triples:
        if couple[0].airline.name == air_pair[0].name:
            other_airline_A = air_pair[1]
            other_airline_B = air_pair[2]
        elif couple[0].airline.name == air_pair[1].name:
            other_airline_A = air_pair[0]
            other_airline_B = air_pair[2]
        elif couple[0].airline.name == air_pair[2].name:
            other_airline_A = air_pair[0]
            other_airline_B = air_pair[1]

        if other_airline_A is not None:
            for pairB in other_airline_A.flight_pairs:
                for pairC in other_airline_B.flight_pairs:

                    if check_couple_condition(mat, [fl.slot.index for fl in couple] + [fl.slot.index for fl in pairB] +
                                                   [fl.slot.index for fl in pairC]):
                        matches.append([couple, pairB, pairC])
    return matches
