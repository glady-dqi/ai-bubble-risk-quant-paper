import numpy as np
from scipy.optimize import curve_fit

# Simple LPPL model

def lppl(t, A, B, C, tc, m, w, phi):
    return A + B*(tc - t)**m + C*(tc - t)**m*np.cos(w*np.log(tc - t) + phi)


def fit_lppl(price_series):
    y = np.log(price_series.values)
    t = np.arange(len(y))
    tc_init = len(y) + 30
    p0 = [y[-1], -0.1, 0.1, tc_init, 0.5, 6.28, 0.1]
    bounds = ([ -np.inf, -np.inf, -np.inf, len(y)+1, 0.1, 4.0, -np.pi ],
              [  np.inf,  np.inf,  np.inf, len(y)+200, 0.9, 15.0,  np.pi ])
    popt, _ = curve_fit(lppl, t, y, p0=p0, bounds=bounds, maxfev=20000)
    return popt
