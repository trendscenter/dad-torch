#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

p1 = '../net_logs/experiment/experiment_log.json'
p2 = '../net_logs_DDP/experiment/experiment_log.json'
p3 = '../net_logs_DAD/experiment/experiment_log.json'
p4 = '../net_logs_DAD_NO_BK/experiment/experiment_log.json'

paths = [p1, p2, p3, p4]
if len(sys.argv) > 1:
    paths = sys.argv[1:]

skip = 5
data = []
data_cumu = []
for p in paths:
    if not p.endswith('.json'):
        p = p + os.sep + 'experiment' + os.sep + 'experiment_log.json'
        
    j = json.load(open(p))
    data.append(j['batch_run_time'][skip:])
    data_cumu.append(np.cumsum(j['batch_run_time'][skip:]))

data = np.array(data).T
data_cumu = np.array(data_cumu).T

colors = ['green', 'magenta', 'red', 'blue', 'cyan', 'yellow']
colors = colors[:len(paths)]
header = ['Single GPU', 'DDP', 'DAD', 'DAD_NO_BK']
df = pd.DataFrame(data=data, columns=header)[skip:]
df.plot(color=colors)
plt.ylabel('Millis')
plt.xlabel('Iteration')
plt.savefig('runtime-compare.png')
plt.close('all')

df = pd.DataFrame(data=data_cumu, columns=header)[skip:]
df.plot(color=colors)
plt.ylabel('Cumulative Millis')
plt.xlabel('Iteration')
plt.savefig('runtime-compare_cumulative.png')
plt.close('all')
