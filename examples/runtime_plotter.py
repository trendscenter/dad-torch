#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import seaborn as sns

plt.rcParams["figure.figsize"] = (20, 9)

sns.set_style("darkgrid")

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-paths', '--paths', nargs='*', type=str, help='Root path to Logs.')
ap.add_argument('-keys', '--keys', nargs='*', type=str, default=[], help='Keys to plot.')
ap.add_argument('-name', '--name', type=str, default='', help='Name of plot.')
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

    fig, axs = plt.subplots(1, 2)
    for p in args['paths']:
        _data = np.cumsum(get_log(p)[k][skip:])
        sns.lineplot(x=range(len(_data)), y=_data, ax=axs[0])
        sns.lineplot(x=range(len(_data)), y=_data, ax=axs[1])

    axs[1].set_yscale("log")
    axs[0].set_ylabel("Cumulative Runtime")
    for ax in axs:
        ax.set_xlabel("Batch Iteration")

    plt.legend(header)
    plt.savefig(f"{k}_cumulative{args['name']}.png", bbox_inches="tight")
    plt.close('all')
    print('Done.')
