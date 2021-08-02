import torch
import numpy as np


def nse(y_true, y_pred):
    return 1-np.sum((y_pred-y_true)**2)/np.sum((y_true-np.mean(y_true))**2)

def kge(y_true, y_pred):
    kge_r = np.corrcoef(y_true,y_pred)[1][0]
    kge_a = np.std(y_pred)/np.std(y_true)
    kge_b = np.mean(y_pred)/np.mean(y_true)
    return 1-np.sqrt((kge_r-1)**2+(kge_a-1)**2+(kge_b-1)**2)