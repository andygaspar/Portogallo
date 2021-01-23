from Istop import istop
from NoNegativeBound import nnBound
from ModelStructure.ScheduleMaker import scheduleMaker
from ModelStructure.Costs.costFunctionDict import CostFuns
from UDPP import udppModel
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np

"""
Here all models are tested:
NNbound = No Negative Bound (unfortunately NN can be confused with Neural Network but it has nothing to do with it)
    This model optimise the total cost constraining all airline to don't have negative impact
    
UDPP = full udpp internal (not exactly) optimisation

ISTOP = offer provider

"""

np.random.seed(0)
scheduleType = scheduleMaker.schedule_types(show=True)

num_flights = 25
num_airlines = 5

for i in range(0, 1):
    df = scheduleMaker.df_maker(num_flights, num_airlines, distribution=scheduleType[3])


    df_NNB = df.copy(deep=True)
    df_UDPP = df_NNB.copy(deep=True)
    costFun = CostFuns().costFun["realistic"]

    print("max from FPFS")
    max_model = nnBound.NNBoundModel(df_NNB, costFun)
    max_model.run()
    max_model.print_performance()


    print("UDPP Opt from FPFS")
    udpp_model_xp = udppModel.UDPPmodel(df_UDPP, costFun)
    udpp_model_xp.run()
    udpp_model_xp.print_performance()

    print("istop from UDPP opt")
    xpModel = istop.Istop(udpp_model_xp.get_new_df(), costFun)
    xpModel.run(True)
    xpModel.print_performance()
