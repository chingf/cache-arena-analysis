import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import gaussian_kde
from scipy import optimize
from math import pi

mat_files = [m for m in os.listdir("data") if m.endswith('mat')]
session_list = pd.read_excel('data/CacheRetrieveSessionList.xlsx', index_col=0)
fps = 20
cmap = cm.get_cmap('viridis')

def estimate_center(x, y):
    method_2 = "leastsq"

    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(x), np.mean(y)
    center_2, ier = optimize.leastsq(f_2, center_estimate)
    return center_2

def get_xy(f, in_bound=True):
    x = np.squeeze(np.array(f['X']))
    y = np.squeeze(np.array(f['Y']))
    x_c, y_c = estimate_center(x, y)
    x -= x_c; y -= y_c
    length = np.sqrt(np.square(x) + np.square(y))
    frames = np.arange(x.size)
    if in_bound:
        oob = np.logical_or(length <= 145, length >= 215)
        x = x[np.logical_not(oob)]
        y = y[np.logical_not(oob)]
        frames = frames[np.logical_not(oob)]
    return x, y, frames

def get_theta(x, y):
    return np.arctan2(y, x)

def get_velocity(f):
    x, y, frames = get_xy(f, in_bound=False)
    delta_x = x[1:] - x[:-1]
    delta_y = y[1:] - y[:-1]
    frames = frames[1:]
    velocity = np.sqrt(np.square(delta_x) + np.square(delta_y)) # pixels/frame
    velocity = velocity*fps # pixels/s
    smoothing_kernel = np.ones(fps)/fps
    velocity = np.convolve(velocity, smoothing_kernel, "valid")
    frames = frames[:velocity.size]
    return velocity, frames

def get_wedges(f):
    x, y, frames = get_xy(f)
    theta = np.mod(get_theta(x, y), 2*pi)
    boundaries = np.linspace(0, 2*pi, 16, endpoint=False)
    boundaries = np.append(boundaries, [2*pi])
    wedges = np.digitize(theta, boundaries)
    return wedges, frames

for mat_file in mat_files:
    f = h5py.File("data/" + mat_file, 'r')
    wedges, wedge_frames = get_wedges(f)
    S = np.array(f['S'])
    num_frames, num_neurons = S.shape
    neurs = np.random.choice(num_neurons, size=20, replace=False)
    for neur in neurs:
        spikes = S[:,neur]
        smoothing_kernel = np.ones(fps)/fps # One sec smoothing
        spikesHz = np.convolve(spikes, smoothing_kernel, "valid")
        spike_frames = np.arange(num_frames)[:spikesHz.size]
        plt.figure(figsize=(10,4))
        plt.plot(spikesHz[1000:1000 + 60*fps])
        plt.title("Neuron %d"%neur)
        plt.show()
    break;
