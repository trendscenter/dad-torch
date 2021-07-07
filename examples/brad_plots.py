import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import json
import pandas as pd
import os
ROOT = '/data/users2/bbaker/projects/dad-torch'

DSGD_FOLDER = 'examples/net_logs/batch-128/dsgd'
DAD_FOLDER = 'examples/net_logs/batch-128/dad'
RANKDAD_FOLDER_1 = 'examples/net_logs/batch-128/rankdad-rank4-numiter1'
RANKDAD_FOLDER_10 = 'examples/net_logs/batch-128/rankdad-rank4-numiter10'

JSON_NAME = 'experiment_log.json'

DSGD_DICT = json.load(open(os.path.join(ROOT,DSGD_FOLDER, JSON_NAME), 'r'))
DAD_DICT = json.load(open(os.path.join(ROOT,DAD_FOLDER, JSON_NAME)))
RANKDAD_1_DICT = json.load(open(os.path.join(ROOT,RANKDAD_FOLDER_1, JSON_NAME)))
RANKDAD_10_DICT = json.load(open(os.path.join(ROOT,RANKDAD_FOLDER_10, JSON_NAME)))

DSGD_DURATION = np.cumsum(DSGD_DICT["batch_duration"])
DAD_DURATION = np.cumsum(DAD_DICT["batch_duration"])
RANKDAD_1_DURATION = np.cumsum(RANKDAD_1_DICT["batch_duration"])
RANKDAD_10_DURATION = np.cumsum(RANKDAD_10_DICT["batch_duration"])
sb.set()
fig = plt.figure()

sb.lineplot(range(len(DSGD_DURATION)), DSGD_DURATION)
sb.lineplot(range(len(DSGD_DURATION)), DAD_DURATION)
sb.lineplot(range(len(DSGD_DURATION)), RANKDAD_1_DURATION)
sb.lineplot(range(len(DSGD_DURATION)), RANKDAD_10_DURATION)
plt.yscale("log")
plt.ylabel("Cumulative Runtime (log)")
plt.xlabel("Batch Iteration")
plt.legend(["dSGD", "dAD", "rank-dAD - rank 4 - 1 Iteration", "rank-dAD - rank 4 - 10 Iterations"])
plt.savefig("brad_rankdad_compare.png", bbox_inches="tight")
print("ok")