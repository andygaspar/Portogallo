# from mip import *
from typing import List
import numpy as np
from UDPP.AirlineAndFlightAndSlot import udppAirline as air
from UDPP.AirlineAndFlightAndSlot import udppFlight as fl
from ModelStructure.Slot import slot as sl
import xpress as xp
xp.controls.outputlog = 0

import ModelStructure.modelStructure as ms


def slot_range(k: int, AUslots: List[sl.Slot]):
    return range(AUslots[k].index + 1, AUslots[k + 1].index)


def eta_limit_slot(flight: fl.UDPPflight, AUslots: List[sl.Slot]):
    i = 0
    for slot in AUslots:
        if slot >= flight.etaSlot:
            return i
        i += 1


def UDPPlocalOpt(airline: air.UDPPairline, slots: List[sl.Slot]):

    m = xp.problem()

    x = np.array([[xp.var(vartype=xp.binary) for j in slots] for i in airline.flights])

    m.addVariable(x)

    flight: fl.UDPPflight

    for k in range(airline.numFlights):
        #one x max for slot
        m.addConstraint(
            xp.Sum(x[flight.localNum, k] for flight in airline.flights) == 1
        )

    for flight in airline.flights:
        # flight assignment
        m.addConstraint(
            xp.Sum(x[flight.localNum, k] for k in
                  range(eta_limit_slot(flight, airline.AUslots), airline.numFlights)) == 1
        )


    m.setObjective(
            xp.Sum(x[flight.localNum][k] * flight.costFun(flight, airline.AUslots[k])
             for flight in airline.flights for k in range(airline.numFlights))
    )

    m.solve()
    # print("airline ",airline)
    for flight in airline.flights:

        for k in range(airline.numFlights):
            if m.getSolution(x[flight.localNum, k]) > 0.5:
                flight.newSlot = airline.flights[k].slot
                flight.priorityNumber = k
                flight.priorityValue = "N"
                # print(flight.slot, flight.newSlot)

    print(sum([flight.costFun(flight, flight.slot) for flight in airline.flights]),
          sum([flight.costFun(flight, flight.newSlot) for flight in airline.flights]))
