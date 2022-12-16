import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from pysurvival.models.semi_parametric import CoxPHModel

from definitions import DATA_DIR

path = os.path.join(DATA_DIR, "synthetic", "linear_exp_synthetic.pt")

# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties100.pt"
# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties10.pt"
# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties5.pt"
# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties3.pt"
# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties9000_nocensoring.pt"
# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties_10000_nocensoring_unif.pt"
# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties_10000_0.3_unif.pt"
path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties3_10000_0.3_unif.pt"


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
cph.print_summary(decimals=5)
print(f"Ratio of coef's:   {cph.summary.coef[1]/cph.summary.coef[0]}")
# cph_pysurv = CoxPHModel()
# cph_pysurv.fit(x_covar, y_times, 1-censored_events)

# wei = WeibullAFTFitter()
# wei.fit(df, duration_col="time", event_col="event", show_progress=True)
# wei.print_summary()
#
# # Plot the survival curves
# kmf = KaplanMeierFitter()
# kmf.fit(df["time"], df["event"], label="CoxPH")
# kmf.survival_function_
# kmf.cumulative_density_
# kmf.plot_survival_function()
#
# kmf.plot_cumulative_density()
# plt.show()
