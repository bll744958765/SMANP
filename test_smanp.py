import torchvision
# -*- coding : utf-8 -*-
import pandas as pd
from dataloader_smanp import DatasetGP_test,data_load
from model_smanp import SpatialNeuralProcess, Criterion

import torch as torch
from train_configs import val_runner
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import time
start = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_tasks = 1
batch_size = 1
x_size = 3
y_size = 1
z_size = 128
lr = 0.001
num_context =982
num_hidden = 128

model = SpatialNeuralProcess(x_size=x_size, y_size=y_size, num_hidden=num_hidden)
model=model.to(device)
dataset = DatasetGP_test(n_tasks=n_tasks, batch_size=batch_size)
testloader = DataLoader(dataset, batch_size=1, shuffle=False)
# _, _, _, _, s1, s2, = data_load()
state_dict = torch.load('./checkpoint_anp/checkpoint_{}.pth.tar'.format(num_context))
model.load_state_dict(state_dict=state_dict['model'])
model.eval()
criterion = Criterion()

val_pred_y, val_var_y, val_target_id, val_target_y, val_loss, valid_r2 = val_runner( model, testloader,criterion)

from math import sqrt

val_target_y = val_target_y.cpu().detach().numpy()
val_pred_y = val_pred_y.cpu().detach().numpy()
val_var_y = val_var_y.cpu().detach().numpy()
val_target_id = val_target_id.cpu().detach().numpy()
valid_mse = (np.sum((val_target_y - val_pred_y) ** 2)) / len(val_target_y)
valid_rmse = sqrt(valid_mse)
valid_mae = (np.sum(np.absolute(val_target_y- val_pred_y))) / len(val_target_y)
corr= np.corrcoef(val_target_y, val_pred_y)
print(corr[0,1])
C=(2*corr[0,1]*np.std(val_pred_y)*np.std(val_target_y))/(np.var(val_target_y)+np.var(val_pred_y)+(val_target_y.mean()-val_pred_y.mean())**2)
# print("val_pred_y",val_pred_y)
print('val_target_y',np.var(val_target_y))
print("validation set effect---------------------------------")
print("valid_MAE:", valid_mae, "valid_MSE:", valid_mse, " valid_RMSE:", valid_rmse,
      " valid_R-square:", valid_r2.cpu().detach().numpy(),"CCC:",C,"average_var:",np.mean(val_var_y))

c = np.linspace(0, len(val_target_y), len(val_target_y))
plt.scatter(c, val_target_y, color="b", marker="x", label="true_value")
plt.scatter(c, val_pred_y, color="r", marker="o", label="predict_value")
plt.fill_between(c, val_pred_y - val_var_y, val_pred_y + val_var_y, alpha=0.2, facecolor="r", interpolate=True)
# plt.xticks(ID)
plt.legend()
plt.xlabel('Well-ID')
plt.ylabel('Reservoir thickness')
plt.savefig('./figure_anp/predict_{}.jpg'.format(num_context), bbox_inches='tight')
plt.grid()
# plt.show()
plt.close()


prediction = pd.DataFrame(
    {"id": np.array(val_target_id ), "true": np.array(val_target_y), "pred": np.array(val_pred_y),
     "cha": np.array(val_target_y) - np.array(val_pred_y), 'var_y': np.array(val_var_y)})
prediction.to_csv('./prediction_val.csv', index=False)