import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from pysurvival.models.semi_parametric import CoxPHModel
from pysurvival.models.survival_forest import RandomSurvivalForestModel
from pysurvival.utils.metrics import concordance_index

from definitions import DATA_DIR

path = os.path.join(DATA_DIR, "synthetic", "linear_exp_synthetic.pt")

# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties100.pt"
# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties10.pt"
# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties5.pt"
# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties3.pt"
# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties9000_nocensoring.pt"
# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties_10000_nocensoring_unif.pt"
# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties_10000_0.3_unif.pt"
# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_linear_exp_independent_ties3_10000_0.3_unif.pt"
# path = '/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_beta_exp.pt'
# path = "/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_beta5_exp.pt"
# path = '/Users/andre/Documents/UCL/diffsurv/data/synthetic/pysurv_beta10_exp_nocen.pt'
path = "/Users/andre/Documents/UCL/diffsurv/data/realworld/flchain.pt"
# path = '/Users/andre/Documents/UCL/diffsurv/data/realworld/nwtco.pt'
# path = '/Users/andre/Documents/UCL/diffsurv/data/realworld/support.pt'
# path = '/Users/andre/Documents/UCL/diffsurv/data/realworld/metabric.pt'

data = torch.load(os.path.join(path))
if len(data) == 5:
    x_covar, y_times, censored_events, risk, y_times_uncensored = (
        data["x_covar"],
        data["y_times"],
        data["censored_events"],
        data["risk"],
        data["y_times_uncensored"],
    )
elif len(data) == 3:
    x_covar, y_times, censored_events = (
        data["x_covar"],
        data["y_times"],
        data["censored_events"],
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
print(f"Ratio of coef's:   {cph.summary.coef[1] / cph.summary.coef[0]}")
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


# Survival Trees
n_patients = x_covar.shape[0]
val_split = 0.2
n_training_patients = int(n_patients * (1 - val_split)) if val_split else n_patients
X = x_covar.numpy()
T = future_label_times.numpy().flatten()
E = future_label_multihot.numpy().flatten()

X_train = X[:n_training_patients]
T_train = T[:n_training_patients]
E_train = E[:n_training_patients]

X_test = X[n_training_patients:]
T_test = T[n_training_patients:]
E_test = E[n_training_patients:]

trees = RandomSurvivalForestModel(num_trees=200)
trees.fit(X=X_train, T=T_train, E=E_train, max_features="sqrt", max_depth=5, min_node_size=20)

#### 5 - Cross Validation / Model Performances
c_index = concordance_index(trees, X_test, T_test, E_test)
print(f"C-index: {c_index:.3f}")
