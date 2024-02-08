import json
import numpy as np
import os

data_dir = '/eagle/MDClimSim/mjp5595/data/'
files = os.listdir(data_dir)
files.sort()

for i,f_name in enumerate(files):
    f = open(os.path.join(data_dir,f_name))
    data_j = json.load(f)

    if i == 0:
        print('data_j.keys() :',data_j.keys())
        # dict_keys(['x_norm', 'x_raw', 'pred_norm', 'pred_raw', 'y_norm', 'y_raw', 'out_variables'])
        variables = data_j['out_variables']
        print('variables :',variables)

    x_raw = np.array(data_j['x_raw'][0][0])
    # x is (82,128,256) -> (vars,lat,lon)

    pred_raw = np.array(data_j['pred_raw'][-1][0])
    # pred_raw is (82,128,256) -> (vars,lat,lon)
    # there are 6 preds, but we use the last one for 36 hours

    y_raw = np.array(data_j['y_raw']['36'][0])
    # y is (82,128,256) -> (vars,lat,lon)