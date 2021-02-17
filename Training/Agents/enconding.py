import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size,lr,  weight_decay, num_flight_types, num_airlines,
                 num_flights, num_trades, num_combs, output_size):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")
        self.to(self.device)

        self.outputSize = output_size
        self.numFlightTypes = num_flight_types
        self.numCombs = num_combs
        self.numAirlines = num_airlines
        self.numFlights = num_flights
        self.inputSize = input_size
        self.numTrades = num_trades
        self.loss = 0

        self.flightShrinkSize = 16
        self.tradeShrinkSize = 16

        ETA_info_size = 1
        time_info_size = 1

        self.flightConvSize = self.numAirlines + self.numFlightTypes + time_info_size + ETA_info_size
        self.singleTradeSize = (self.numAirlines + self.numCombs) * 2

        self.airConv = nn.Linear(self.numAirlines, 4).to(self.device)
        self.flightTypeConv = nn.Linear(self.numFlightTypes, 4).to(self.device)
        self.etaConv = nn.Linear(1, 4).to(self.device)
        self.timeConv = nn.Linear(1, 4).to(self.device)

        self.firstConvFlights = nn.Linear(self.flightShrinkSize, self.flightShrinkSize * 4).to(self.device)
        self.secondConvFlights = nn.Linear(self.flightShrinkSize * 4, self.flightShrinkSize * 8).to(self.device)
        self.thirdConvFlights = nn.Linear(self.flightShrinkSize * 8, self.flightShrinkSize).to(self.device)
        # self.fourthConvFlights = nn.Linear(self.flightConvSize * 2, self.flightShrinkSize * 2).to(self.device)
        # self.fifthConvFlights = nn.Linear(self.flightShrinkSize * 2, self.flightShrinkSize).to(self.device)
        # self.sixthConvFlights = nn.Linear(self.flightShrinkSize, self.flightShrinkSize).to(self.device)

        self.firstConvTrades = nn.Linear(self.singleTradeSize, self.singleTradeSize * 4).to(self.device)
        self.secondConvTrades = nn.Linear(self.singleTradeSize * 4, self.singleTradeSize * 8).to(self.device)
        self.thirdConvTrades = nn.Linear(self.singleTradeSize * 8, self.singleTradeSize * 2).to(self.device)
        self.fourthConvTrades = nn.Linear(self.singleTradeSize * 2, self.tradeShrinkSize).to(self.device)

        self.firstConvCurrentTrade = nn.Linear(self.singleTradeSize, self.singleTradeSize * 4).to(self.device)
        self.secondConvCurrentTrade = nn.Linear(self.singleTradeSize * 4, self.singleTradeSize * 8).to(self.device)
        self.thirdConvCurrentTrade = nn.Linear(self.singleTradeSize * 8, self.singleTradeSize * 2).to(self.device)
        self.fourthConvCurrentTrade = nn.Linear(self.singleTradeSize * 2, self.tradeShrinkSize).to(self.device)

        # deconvolution

        self.firstDeConvFlights = nn.Linear(self.flightShrinkSize, self.flightShrinkSize*8).to(self.device)
        self.secondDeConvFlights = nn.Linear(self.flightShrinkSize*8, self.flightShrinkSize *4).to(self.device)
        self.thirdDeConvFlights = nn.Linear(self.flightShrinkSize * 4, self.flightShrinkSize).to(self.device)
        # self.fourthDeConvFlights = nn.Linear(self.flightConvSize * 2, self.flightConvSize*8).to(self.device)
        # self.fifthDeConvFlights = nn.Linear(self.flightConvSize * 8, self.flightConvSize*4).to(self.device)
        # self.sixthDeConvFlights = nn.Linear(self.flightConvSize * 4, self.flightConvSize).to(self.device)

        self.airDeConv = nn.Linear(4, self.numAirlines).to(self.device)
        self.flightTypeDeConv = nn.Linear(4, self.numFlightTypes).to(self.device)
        self.etaDeConv = nn.Linear(4, 1).to(self.device)
        self.timeDeConv = nn.Linear(4, 1).to(self.device)

        self.firstDeConvTrades = nn.Linear(self.tradeShrinkSize, self.singleTradeSize * 2).to(self.device)
        self.secondDeConvTrades = nn.Linear(self.singleTradeSize * 2, self.singleTradeSize*8).to(self.device)
        self.thirdDeConvTrades = nn.Linear(self.singleTradeSize * 8, self.singleTradeSize*4).to(self.device)
        self.fourthDeConvTrades = nn.Linear(self.singleTradeSize * 4, self.singleTradeSize).to(self.device)

        self.firstDeConvCurrentTrade = nn.Linear(self.tradeShrinkSize, self.singleTradeSize * 2).to(self.device)
        self.secondDeConvCurrentTrade = nn.Linear(self.singleTradeSize * 2, self.singleTradeSize*8).to(self.device)
        self.thirdDeConvCurrentTrade = nn.Linear(self.singleTradeSize * 8, self.singleTradeSize*4).to(self.device)
        self.fourthDeConvCurrentTrade = nn.Linear(self.singleTradeSize * 4, self.singleTradeSize).to(self.device)

        # self.jointInputSize = self.numFlights * 4 + (self.numTrades + 1) * 8
        # self.l1 = nn.Linear(self.jointInputSize, self.jointInputSize * 2).to(self.device)
        # self.l2 = nn.Linear(self.jointInputSize * 2, self.jointInputSize).to(self.device)
        # self.l3 = nn.Linear(self.jointInputSize, int(self.jointInputSize / 2)).to(self.device)
        # self.l4 = nn.Linear(int(self.jointInputSize / 2), self.outputSize).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, state):
        flights, trades, current_trade = torch.split(state, [self.flightConvSize * self.numFlights,
                                                             self.singleTradeSize * self.numTrades,
                                                             self.singleTradeSize], dim=-1)

        flights = torch.split(flights, [self.flightConvSize for _ in range(self.numFlights)], dim=-1)
        flight_decomposition = [torch.split(f, [self.numAirlines,
                                                self.numFlightTypes,
                                                1, 1], dim=-1) for f in flights]
        air = [self.airConv(a[0]) for a in flight_decomposition]
        f_type = [self.flightTypeConv(f[1]) for f in flight_decomposition]
        time = [self.timeConv(t[2]) for t in flight_decomposition]
        eta = [self.etaConv(e[3]) for e in flight_decomposition]

        flights = [torch.cat([air[i], f_type[i], time[i], eta[i]], dim=-1) for i in range(self.numFlights)]

        trades = torch.split(trades, [self.singleTradeSize for _ in range(self.numTrades)], dim=-1)

        flights = [self.firstConvFlights(flight) for flight in flights]
        flights = [self.secondConvFlights(flight) for flight in flights]
        flights = [self.thirdConvFlights(flight) for flight in flights]
        # flights = [self.fourthConvFlights(flight) for flight in flights]
        # flights = [self.fifthConvFlights(flight) for flight in flights]
        # flights = [self.sixthConvFlights(flight) for flight in flights]
        flights = torch.cat(flights, dim=-1)

        trades = [self.firstConvTrades(trade) for trade in trades]
        trades = [self.secondConvTrades(trade) for trade in trades]
        trades = [self.thirdConvTrades(trade) for trade in trades]
        trades = [self.fourthConvTrades(trade) for trade in trades]
        trades = torch.cat(trades, dim=-1)

        current_trade = self.firstConvCurrentTrade(current_trade)
        current_trade = self.secondConvCurrentTrade(current_trade)
        current_trade = self.thirdConvCurrentTrade(current_trade)
        current_trade = self.fourthConvCurrentTrade(current_trade)

        # joint = torch.cat([flights, trades, current_trade], dim=-1)
        #
        # x = F.relu(self.l1(joint))
        # x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        # return self.l4(x)

        flights = torch.split(flights, [self.flightShrinkSize for _ in range(self.numFlights)], dim=-1)
        trades = torch.split(trades, [self.tradeShrinkSize for _ in range(self.numTrades)], dim=-1)

        flights = [self.firstDeConvFlights(flight) for flight in flights]
        flights = [self.secondDeConvFlights(flight) for flight in flights]
        flights = [self.thirdDeConvFlights(flight) for flight in flights]

        flight_decomposition = [torch.split(f, [4, 4, 4, 4], dim=-1) for f in flights]
        air = [self.airDeConv(a[0]) for a in flight_decomposition]
        f_type = [self.flightTypeDeConv(f[1]) for f in flight_decomposition]
        time = [self.timeDeConv(t[2]) for t in flight_decomposition]
        eta = [self.etaDeConv(e[3]) for e in flight_decomposition]
        # flights = [self.fourthDeConvFlights(flight) for flight in flights]
        # flights = [self.fifthDeConvFlights(flight) for flight in flights]
        # flights = [self.sixthDeConvFlights(flight) for flight in flights]
        flights = [torch.cat([air[i], f_type[i], time[i], eta[i]], dim=-1) for i in range(self.numFlights)]
        flights = torch.cat(flights, dim=-1)

        trades = [self.firstDeConvTrades(trade) for trade in trades]
        trades = [self.secondDeConvTrades(trade) for trade in trades]
        trades = [self.thirdDeConvTrades(trade) for trade in trades]
        trades = [self.fourthDeConvTrades(trade) for trade in trades]
        trades = torch.cat(trades, dim=-1)

        current_trade = self.firstDeConvCurrentTrade(current_trade)
        current_trade = self.secondDeConvCurrentTrade(current_trade)
        current_trade = self.thirdDeConvCurrentTrade(current_trade)
        current_trade = self.fourthDeConvCurrentTrade(current_trade)
        x = torch.cat([flights, trades, current_trade], dim=-1)

        return flights



    def pick_action(self, state):
        with torch.no_grad():
            scores = self.forward(state.to(self.device))
        action = torch.zeros_like(scores)
        action[torch.argmax(scores)] = 1

        return action

    def update_weights(self, X: torch.tensor, gamma: float = 0.9):
        criterion = torch.nn.MSELoss()

        Y_train = X.clone()
        flights, trades, current_trade = torch.split(Y_train, [self.flightConvSize * self.numFlights,
                                                             self.singleTradeSize * self.numTrades,
                                                             self.singleTradeSize], dim=-1)
        for i in range(1):
            self.zero_grad()
            Y = self.forward(X)

            loss = criterion(flights, Y)  # .detach()
            self.loss = loss.item()
            self.optimizer.zero_grad()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.optimizer.step()
    #
    #   super().__init__()
    #   self.convSchedule = nn.Linear(21, 4)
    #   self.
    #   self.convSchedule = nn.Linear(5, 1)
    #
    #   self.tradesConv1 = nn.Linear()
    #
    #   self.fc1 = nn.Linear(20, 64)
    #   self.fc2 = nn.Linear(64, 4)
    #
    # def forward(self, air, others, trades):
    #
    #   first_conv_out = torch.split(air, [17 for _ in range(5)], dim=-1)
    #   second_conv_out = [F.relu(self.convAir1(t)) for t in first_conv_out]
    #   to_merge = torch.cat([F.relu(self.convAir2(t)) for t in second_conv_out], dim=-1)
    #
    #   first_conv_out = torch.split(others, [21 for _ in range(15)], dim=-1)
    #   second_conv_out = [F.relu(self.convOthers1(t)) for t in first_conv_out]
    #   to_merge1 = torch.cat([F.relu(self.convOthers2(t)) for t in second_conv_out], dim=-1)
    #
    #   merged_tensor = torch.cat((to_merge, to_merge1), dim=-1)
    #
    #   out = F.relu(self.fc1(merged_tensor))
    #
    #   # return self.fc2(out)
