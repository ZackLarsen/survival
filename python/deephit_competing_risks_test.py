
"""
https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/deephit_competing_risks.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchtuples as tt

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models import DeepHit
from pycox.evaluation import EvalSurv

np.random.seed(1234)
_ = torch.manual_seed(1234)

url = 'https://raw.githubusercontent.com/chl8856/DeepHit/master/sample%20data/SYNTHETIC/synthetic_comprisk.csv'
df_train = pd.read_csv(url)
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

df_train.head()

get_x = lambda df: (df
                    .drop(columns=['time', 'label', 'true_time', 'true_label'])
                    .values.astype('float32'))

x_train = get_x(df_train)
x_val = get_x(df_val)
x_test = get_x(df_test)

class LabTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')

num_durations = 10
labtrans = LabTransform(num_durations)
get_target = lambda df: (df['time'].values, df['label'].values)

y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))
durations_test, events_test = get_target(df_test)
val = (x_val, y_val)

y_train[0][:6], y_train[1][:6]

labtrans.cuts


class SimpleMLP(torch.nn.Module):
    """Simple network structure for competing risks.
    """

    def __init__(self, in_features, num_nodes, num_risks, out_features, batch_norm=True,
                 dropout=None):
        super().__init__()
        self.num_risks = num_risks
        self.mlp = tt.practical.MLPVanilla(
            in_features, num_nodes, num_risks * out_features,
            batch_norm, dropout,
        )

    def forward(self, input):
        out = self.mlp(input)
        return out.view(out.size(0), self.num_risks, -1)


class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                 out_features, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input):
        out = self.shared_net(input)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out


in_features = x_train.shape[1]
num_nodes_shared = [64, 64]
num_nodes_indiv = [32]
num_risks = y_train[1].max()
out_features = len(labtrans.cuts)
batch_norm = True
dropout = 0.1

# net = SimpleMLP(in_features, num_nodes_shared, num_risks, out_features)
net = CauseSpecificNet(in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                       out_features, batch_norm, dropout)

# Training
optimizer = tt.optim.AdamWR(lr=0.01, decoupled_weight_decay=0.01,
                            cycle_eta_multiplier=0.8)
model = DeepHit(net, optimizer, alpha=0.2, sigma=0.1,
                duration_index=labtrans.cuts)

epochs = 512
batch_size = 256
callbacks = [tt.callbacks.EarlyStoppingCycle()]
#verbose = False # set to True if you want printout
verbose = True

#%%time # Magic command for Jupyter Notebook only
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val)

_ = log.plot()

# Evaluation
surv = model.predict_surv_df(x_test)
ev = EvalSurv(surv, durations_test, events_test != 0, censor_surv='km')

ev.concordance_td()

ev.integrated_brier_score(np.linspace(0, durations_test.max(), 100))

cif = model.predict_cif(x_test)
cif1 = pd.DataFrame(cif[0], model.duration_index)
cif2 = pd.DataFrame(cif[1], model.duration_index)

ev1 = EvalSurv(1-cif1, durations_test, events_test == 1, censor_surv='km')
ev2 = EvalSurv(1-cif2, durations_test, events_test == 2, censor_surv='km')

ev1.concordance_td()

ev2.concordance_td()

sample = np.random.choice(len(durations_test), 6)
fig, axs = plt.subplots(2, 3, True, True, figsize=(10, 5))
for ax, idx in zip(axs.flat, sample):
    pd.DataFrame(cif.transpose()[idx], index=labtrans.cuts).plot(ax=ax)
    ax.set_ylabel('CIF')
    ax.set_xlabel('Time')
    ax.grid(linestyle='--')

