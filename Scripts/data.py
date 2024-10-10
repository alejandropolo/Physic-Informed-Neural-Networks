import sys
import importlib

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

sys.path.append('../Scripts/')
import PINN_moredim

importlib.reload(PINN_moredim)
from PINN_moredim import DiffEquation, min_max_normalize

# from train import train

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)


def load_csv_data(diff_equation: DiffEquation, filename: str , params_to_optimize: list[str] = [], noise: float = 0.0):

    col = 'Voltage_Eq'
    
    df = pd.read_csv(filename)
    df['Time'] = df['Time'].round(2) ## Evita efectos raros en las derivasdas por valores temporales muy proximos
    df.drop_duplicates(subset=['Time'], keep='first', inplace=True)
    df.drop_duplicates(subset=['v_cell'], keep='first', inplace=True)
    df.drop_duplicates(subset=['t_membrane'], keep='first', inplace=True)  

    df = df.loc[df.Time > 200]
    df = df[(df['T'] > 79) & (df['T'] < 81)]
    k_1_mean = df['k_1'].mean()
    k_2_mean = df['k_2'].mean()
    k_3_mean = df['k_3'].mean()


    df['Voltage_Eq'] = k_1_mean+k_2_mean*np.log(df['power_elec']/df['v_cell'])+k_3_mean*df['power_elec']*df['t_membrane']/df['v_cell']
    k_1_mean_mod = (k_1_mean+k_2_mean*np.log(100000.0)-k_2_mean)
    

    X_test_tensor = torch.tensor(df['Time'].values, dtype=torch.float32, device=DEVICE).view(-1,1)
    y_test_tensor = torch.tensor(df[col].values, dtype=torch.float32, device=DEVICE).view(-1,1)

    ## Min Max Scale
    X_test_tensor,tensor_min,tensor_max = min_max_normalize(X_test_tensor)

    ## Select n first points as train
    n_points = 1
    t = np.round(np.linspace(0, len(X_test_tensor)//2, n_points))
    if params_to_optimize:
        n_points = 6
        t = np.round(np.linspace(0, len(X_test_tensor)//2, n_points))

        t = np.concatenate((t,np.array([len(X_test_tensor)-40])))
        t = np.concatenate((t,np.array([len(X_test_tensor)-25])))
        # t = np.concatenate((t,np.array([len(X_test_tensor)-10])))
        # t = np.concatenate((t,np.array([len(X_test_tensor)-1])))


    X_tensor = X_test_tensor[t]
    y_tensor = y_test_tensor[t]

    ## Add noise to y_tensor but not to the first point
    y_tensor[1:] = y_tensor[1:] + noise * torch.randn(len(y_tensor)-1,1)

    ## Scatterplot of the training data and test data
    plt.scatter(X_test_tensor.detach().numpy(), y_test_tensor.detach().numpy(),label='Test data',s=5)
    plt.scatter(X_tensor.detach().numpy(), y_tensor.detach().numpy(),label='Training data',color='orange')
    ## Plot also real voltage
    #plt.plot(X_test_tensor.detach().numpy(), df['Voltage_Eq'].values,label='Real Voltage',color='red')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.title('Training data')
    plt.legend()
    plt.show()

    diff_equation.tensor_max = tensor_max
    diff_equation.tensor_min = tensor_min
    constants = {'k_1_mean_mod': k_1_mean_mod, 'k_1_mean': k_1_mean, 'k_2_mean': k_2_mean, 'k_3_mean': k_3_mean}
    diff_equation.set_constants(constants)

    return (X_tensor, y_tensor, X_test_tensor, y_test_tensor)

