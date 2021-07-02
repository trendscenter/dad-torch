#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import glob

paths = []
if len(sys.argv) > 1:
    paths = sys.argv[1:]
else:
    raise Exception('Must pass log directories as runtime args.')


def get_log(pth):
    _exp_pth = pth + os.sep + os.listdir(pth)[0]
    _jsn_pth = glob.glob(f'{_exp_pth}{os.sep}*_log.json')[0]
    return json.load(open(_jsn_pth))


keys = [k for k in get_log(paths[0]).keys() if '_duration' in k]
header = [p.split(os.sep)[-1].split('_')[-1] for p in paths]

skip = 5
for k in keys:
    data_cumu = []
    for p in paths:
        data_cumu.append(np.cumsum(get_log(p).get(k, [])[skip:]))

    data_cumu = np.array(data_cumu).T
    df = pd.DataFrame(data=data_cumu, columns=header)[skip:]
    df.plot()
    plt.ylabel('Cumulative Millis')
    plt.xlabel('Iteration')
    plt.savefig(f'{k}_cumulative.png')
    plt.close('all')
