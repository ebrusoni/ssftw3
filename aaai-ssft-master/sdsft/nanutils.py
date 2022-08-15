from matplotlib.pyplot import axis
import numpy as np
def delete(arr: np.array, del0, del1):
    arr_del0 = np.delete(arr, del0, axis=0)
    ax1 = arr_del0.ndim - 1 
    arr_del1 = np.delete(arr_del0, del1, axis=ax1)
    return arr_del1

def clean_nan(M: np.array, measurements_new: np.array, measurements_previous: np.array):
    #removes rows M_i of M, where M_i a non-zero entry of M_i was multiplied with a NaN entry in x
    #removes columns M_j of M, where x_j is NaN
    valid_entries = np.logical_not(np.isnan(measurements_new))
    valid_idxs = np.where(valid_entries)[0]
    M = M[valid_idxs][:, valid_idxs]
    measurements_new = measurements_new[valid_idxs]
    measurements_previous = measurements_previous[valid_idxs]

    return (M, measurements_new, measurements_previous)