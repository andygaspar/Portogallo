
def run_episode(model, air_agent, fl_agent, length_episode):

    current_trade_list = []
    current_trade_list.append(air_agent.pick_action(current_trade_list))
    current_trade_list.append(fl_agent.pick_action(current_trade_list))
    current_trade_list.append(air_agent.pick_action(current_trade_list))
    current_trade_list.append(fl_agent.pick_action(current_trade_list))