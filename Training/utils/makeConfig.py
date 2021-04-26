import pickle

config_dir = './Training/configurations/'
filename = 'config_firstOverfit.pickle'
config_dict = {}

config_dict['discretization_size'] = 50
config_dict['weight_decay'] = 1e-2
config_dict['learning_rate'] = 1e-6
config_dict['batch_size'] = 1024
config_dict['memory_size'] = 200
config_dict['hidden_dim'] = 32
config_dict['n_heads'] = 4
config_dict['n_attention_layers'] = 1


with open(config_dir + filename, 'wb') as handle:
    pickle.dump(config_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
