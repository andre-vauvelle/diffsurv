import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter

from definitions import DATA_DIR

path = os.path.join(DATA_DIR, "synthetic", "linear_exp_synthetic.pt")
(x_covar, y_times, censored_events) = torch.load(os.path.join(path))

future_label_multihot = 1 - censored_events
future_label_times = y_times

# Create dataframe from tensors
data = torch.cat((x_covar[:, :], future_label_multihot, future_label_times), dim=1)
df = pd.DataFrame.from_records(
    data.numpy(), columns=list(range(x_covar.shape[1])) + ["event", "time"]
)

cph = CoxPHFitter()
cph.fit(df, duration_col="time", event_col="event", show_progress=True)
cph.print_summary()

wei = WeibullAFTFitter()
wei.fit(df, duration_col="time", event_col="event", show_progress=True)
wei.print_summary()

# Plot the survival curves
kmf = KaplanMeierFitter()
kmf.fit(df["time"], df["event"], label="CoxPH")
kmf.survival_function_
kmf.cumulative_density_
kmf.plot_survival_function()

kmf.plot_cumulative_density()
plt.show()
