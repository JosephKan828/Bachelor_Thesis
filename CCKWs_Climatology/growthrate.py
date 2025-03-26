# This program is to compute the growth rate of
# different heating source
# Section 1: Import packages
import sys;
import numpy as np;
from joblib import Parallel, delayed;
from netCDF4 import Dataset as ds;
from itertools import product;
from matplotlib import pyplot as plt;
from matplotlib.colors import TwoSlopeNorm;
from numpy.random import permutation;

sys.path.append("/home/b11209013/Package");
import Theory as th; #type: ignore
import Chien_Kim as ck; #type: ignore

# Section 2: Load data
def Load_data() -> dict[str, np.ndarray]:
    # File name
    fname   : str = "/work/b11209013/2024_Research/MPAS/PC/CNTL_PCs.nc";
    
    # Variable list
    var_list: list[str] = ["t", "lw_heating", "sw_heating"];
        
    with ds(fname, "r") as f:
        dims: dict[str, np.ndarray] = dict(
            (var, f.variables[var][:])
            for var in f.dimensions.keys()
        );

        data = dict(
            (var, f.variables[var][:2])
            for var in var_list 
        );
        
    return var_list, dims, data;
# Formatting data
def symm_fmt(data:np.ndarray) -> np.ndarray:
    symm_data: np.ndarray = (data + np.flip(data, axis=2)) / 2;
    return symm_data;

def parallel_shuffle(data, var_list):
    """Shuffles the data and removes the 'pc' layer in the output dictionary."""
    
    def process(key):
        """Shuffles data for a specific key across all `pc` keys and stacks them."""
        shuffled = np.array([np.array([np.random.permutation(data[key][i]) for _ in range(300)]) for i in range (2)])
        return key, shuffled

    # Run parallel shuffle computation
    results = Parallel(n_jobs=-1)(
        delayed(process)(key) for key in var_list
    )

    # Convert results to a dictionary with only var_list keys
    data_shuffle = {key: shuffled_data for key, shuffled_data in results}
    
    return data_shuffle

def main():
    # Load data
    var_list, dims, data = Load_data();    
    print("Finish loading data");
    
    # Processing data
    data_symm: dict[str, dict[str, np.ndarray]] = dict(
        zip(var_list, Parallel(n_jobs=-1)(
            delayed(symm_fmt)(data[var])
            for var in var_list
        ))
    );

    del data;
    
    print("Finish symmetrizing data");
    
    # Random Shuffle data
    # dictionary for shuffle data
    # random object
    data_shuffle = parallel_shuffle(data_symm, var_list)

    print("Finish shuffling data");
    
    # Windowing data
    lsec: int = 120;
    
    hanning: np.ndarray = np.hanning(lsec)[None, :, None, None];
    hanning_shuffled: np.ndarray = np.hanning(lsec)[None, None, :, None, None];
    
    data_window: dict[str, dict[str, np.ndarray]] = dict();
    data_window_shuffled: dict[str, dict[str, np.ndarray]] = dict();

    indices = [slice(i*60, i*60+120) for i in range(5)]

    def compute_windows(key):
        """Compute both regular and shuffled windowed data for a given pc and key."""
        data_win = np.stack([
            data_symm[key][:, idx] * hanning for idx in indices
        ])  # Shape: (5, 120, ...)

        data_win_shuffled = np.stack([
            data_shuffle[key][:, :, idx] * hanning_shuffled for idx in indices
        ])  # Shape: (5, num_shuffles, 120, ...)

        return key, data_win, data_win_shuffled

    results = Parallel(n_jobs=-1)(
        delayed(compute_windows)(key) for key in data_symm.keys())
    
    for key, win, win_shuffled in results:
        data_window[key] = win
        data_window_shuffled[key] = win_shuffled

    print(data_window["t"].shape)
    print(data_window_shuffled["t"].shape)

    print("Finish windowing data");

if __name__ == "__main__":
    # variable list
    main();
    