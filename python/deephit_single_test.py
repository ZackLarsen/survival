
"""
https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/deephit.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch # For building the networks
import torchtuples as tt # Some useful functions

from pycox.datasets import metabric
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

np.random.seed(1234)
_ = torch.manual_seed(123)

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

df_train.head()



cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

num_durations = 10
labtrans = DeepHitSingle.label_transform(num_durations)
get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))

train = (x_train, y_train)
val = (x_val, y_val)

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)

type(labtrans)

# Making Neural Network with torch:
in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.1

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)

batch_size = 256
lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=3)
_ = lr_finder.plot()

lr_finder.get_best_lr()

model.optimizer.set_lr(0.01)

# Training with best learning rate:
epochs = 100
callbacks = [tt.callbacks.EarlyStopping()]
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)

_ = log.plot()

# Prediction:
surv = model.predict_surv_df(x_test)

surv.iloc[:, :5].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

# Interpolating the survival estimates because the survival estimates so far are
# only defined at the 10 times in the discretization grid and the survival
# estimates are therefore a step function rather than a continuous one
surv = model.interpolate(10).predict_surv_df(x_test)

surv.iloc[:, :5].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

# The EvalSurv class contains some useful evaluation criteria for time-to-event prediction.
# We set censor_surv = 'km' to state that we want to use Kaplan-Meier for estimating the
# censoring distribution.

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')

ev.concordance_td('antolini')


# Brier Score
# We can plot the the IPCW Brier score for a given set of times.
# Here we just use 100 time-points between the min and max duration in the test set.
# Note that the score becomes unstable for the highest times.
# It is therefore common to disregard the rightmost part of the graph.

time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
ev.brier_score(time_grid).plot()
plt.ylabel('Brier score')
_ = plt.xlabel('Time')


# Negative binomial log-likelihood
# In a similar manner, we can plot the the IPCW negative binomial log-likelihood.

ev.nbll(time_grid).plot()
plt.ylabel('NBLL')
_ = plt.xlabel('Time')


# Integrated scores
# The two time-dependent scores above can be integrated over time to produce a single score
# (Graf et al. 1999). In practice this is done by numerical integration over a defined time_grid.

ev.integrated_brier_score(time_grid)

ev.integrated_nbll(time_grid)

