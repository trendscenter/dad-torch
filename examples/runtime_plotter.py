#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import seaborn as sns

sns.set_style("darkgrid")

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-paths', '--paths', nargs='*', type=str, help='Root path to Logs.')
ap.add_argument('-keys', '--keys', nargs='*', type=str, default=[], help='Keys to plot.')
args = vars(ap.parse_args())


def get_log(pth):
    _exp_pth = pth + os.sep + os.listdir(pth)[0]
    _jsn_pth = glob.glob(f'{_exp_pth}{os.sep}*_log.json')[0]
    return json.load(open(_jsn_pth))


keys = args['keys']
if len(keys) == 0:
    _keys = [k for k in get_log(args['paths'][0]).keys() if '_duration' in k]
    keys = [k for k in _keys if not k.startswith('l')]  # SKIP Layer
header = [p.split(os.sep)[-1].split('_')[-1] for p in args['paths']]

skip = 5
for k in keys:
    print(f"Working on key {k}...")
    for p in args['paths']:
        _data = np.cumsum(get_log(p)[k][skip:])
        sns.lineplot(x=range(len(_data)), y=_data)

    plt.yscale("log")
    plt.ylabel("Cumulative Runtime (log)")
    plt.xlabel("Batch Iteration")
    plt.legend(header)
    plt.savefig(f'{k}_cumulative.png', bbox_inches="tight")
    print('Done.')
