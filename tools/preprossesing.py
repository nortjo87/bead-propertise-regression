from scipy.signal import savgol_filter
import numpy as np
import pandas as pd

#1. Mean norm
def mean_norm(x):
    if isinstance(x, np.ndarray):
        r, c = np.shape(x)
        mn = np.zeros((r, c))
        for i in range(r):
            mn[i, :] = x[i, :] / np.mean(x[i, :])
        return mn
    elif isinstance(x, pd.DataFrame):
        return x.apply(lambda row: row / row.mean(), axis=1)
    else:
        raise TypeError("Input must be a numpy array or a pandas DataFrame")
#2. Max norm
def max_norm(x):
    if isinstance(x, np.ndarray):
        r, c = np.shape(x)
        mx = np.zeros((r, c))
        for i in range(r):
            mx[i, :] = x[i, :] / np.max(x[i, :])
        return mx
    elif isinstance(x, pd.DataFrame):
        return x.apply(lambda row: row / row.max(), axis=1)
    else:
        raise TypeError("Input must be a numpy array or a pandas DataFrame")

#3. Range norm
def range_norm(x):
    if isinstance(x, np.ndarray):
        r, c = np.shape(x)
        rn = np.zeros((r, c))
        for i in range(r):
            rn[i, :] = x[i, :] / (np.max(x[i, :]) - np.min(x[i, :]))
        return rn
    elif isinstance(x, pd.DataFrame):
        return x.apply(lambda row: row / (row.max() - row.min()), axis=1)
    else:
        raise TypeError("Input must be a numpy array or a pandas DataFrame")

#4. SNV
def snv(x):
    if isinstance(x, np.ndarray):
        _, c = np.shape(x)
        mean = np.mean(x, axis=1)
        sd = np.std(x, axis=1, ddof=0)
        return (x - mean[:, np.newaxis]) / (sd[:, np.newaxis])
    elif isinstance(x, pd.DataFrame):
        mean = x.mean(axis=1)
        sd = x.std(axis=1, ddof=0)
        return (x.sub(mean, axis=0)).div(sd, axis=0)
    else:
        raise TypeError("Input must be a numpy array or a pandas DataFrame")

#5. MSC
def msc(x):
    if isinstance(x, np.ndarray):
        reference = None
        for i in range(x.shape[0]):
            x[i, :] -= x[i, :].mean()
        ref = np.mean(x, axis=0) if reference is None else reference
        pp = np.zeros_like(x)
        for i in range(x.shape[0]):
            fit = np.polyfit(ref, x[i, :], 1, full=True)
            pp[i, :] = (x[i, :] - fit[0][1]) / fit[0][0]
        return pp
    elif isinstance(x, pd.DataFrame):
        reference = None
        x_centered = x.sub(x.mean(axis=1), axis=0)
        ref = x_centered.mean(axis=0) if reference is None else reference
        pp = x_centered.copy()
        for i in range(x_centered.shape[0]):
            fit = np.polyfit(ref, x_centered.iloc[i, :], 1, full=True)
            pp.iloc[i, :] = (x_centered.iloc[i, :] - fit[0][1]) / fit[0][0]
        return pp
    else:
        raise TypeError("Input must be a numpy array or a pandas DataFrame")

#6. SG1
def sg1(x):
    if isinstance(x, np.ndarray):
        return savgol_filter(x, 5, 3, 1)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(savgol_filter(x.values, 5, 3, 1), index=x.index, columns=x.columns)
    else:
        raise TypeError("Input must be a numpy array or a pandas DataFrame")

#7. SG2
def sg2(x):
    if isinstance(x, np.ndarray):
        return savgol_filter(x, 5, 3, 2)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(savgol_filter(x.values, 5, 3, 2), index=x.index, columns=x.columns)
    else:
        raise TypeError("Input must be a numpy array or a pandas DataFrame")

#8. Smoothing mean
def smoothing_mean(x, Msize):
    if isinstance(x, np.ndarray):
        return savgol_filter(x, window_length=Msize, polyorder=2, mode='nearest')
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(savgol_filter(x.values, window_length=Msize, polyorder=2, mode='nearest'), index=x.index, columns=x.columns)
    else:
        raise TypeError("Input must be a numpy array or a pandas DataFrame")
    
#9. First derivative
def fd1(x):
    if isinstance(x, np.ndarray):
        return np.gradient(x, axis=1)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(np.gradient(x.values, axis=1), index=x.index, columns=x.columns)
    else:
        raise TypeError("Input must be a numpy array or a pandas DataFrame")

#10. Second derivative
def fd2(x):
    if isinstance(x, np.ndarray):
        return np.gradient(np.gradient(x, axis=1), axis=1)
    elif isinstance(x, pd.DataFrame):
        return pd.DataFrame(np.gradient(np.gradient(x.values, axis=1), axis=1), index=x.index, columns=x.columns)
    else:
        raise TypeError("Input must be a numpy array or a pandas DataFrame")
    
#11. Savitzky-Golay filter

    
