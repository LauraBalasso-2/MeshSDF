from collections import OrderedDict

import torch
import sys

filename = sys.argv[1]

data = torch.load(filename)

new_data = data.copy()

new_data['model_state_dict'] = OrderedDict()

for k in data.get('model_state_dict').keys():
    k_new = k.replace('module.module', 'module')
    new_data['model_state_dict'][k_new] = data.get('model_state_dict').get(k)

torch.save(new_data, filename)

