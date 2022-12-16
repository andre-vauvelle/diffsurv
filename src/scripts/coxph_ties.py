import itertools
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from tqdm import tqdm

from definitions import DATA_DIR

path = os.path.join(DATA_DIR, "synthetic", "linear_exp_synthetic.pt")


trails = range(100)
ties_groups = (4, 5, 10, 50, 100, 200, 500, 1000, 2500, 5000, 10000)

experiments = list(itertools.product(trails, ties_groups))
path_store = []
dict_store = []
for trail, ties in tqdm(experiments, total=len(experiments)):
    path = (
        f"/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties{str(ties)}"
        f"_{trail}.pt"
    )
    path_store.append(path)
    data = torch.load(os.path.join(path))
    x_covar, y_times, censored_events, risk, y_times_uncensored = (
        data["x_covar"],
        data["y_times"],
        data["censored_events"],
        data["risk"],
        data["y_times_uncensored"],
    )

    future_label_multihot = 1 - censored_events
    future_label_times = y_times

    # Create dataframe from tensors
    data = torch.cat((x_covar[:, :], future_label_multihot, future_label_times), dim=1)
    df = pd.DataFrame.from_records(
        data.numpy(), columns=list(range(x_covar.shape[1])) + ["event", "time"]
    )

    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="event", show_progress=True)
    dict_store.append(
        {
            "tr": trail,
            "ties": ties,
            "param1": cph.summary.coef[0],
            "param2": cph.summary.coef[1],
        }
    )

df_results = pd.DataFrame.from_records(dict_store)

df_results.loc[:, "ratio"] = df_results.param2 / df_results.param1
df_results.boxplot(by="ties", column="ratio")

df_results.hist()

df_results.plot()
plt.show()
