
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import sys
import os
import torch
from tqdm import tqdm_notebook as tqdm
from config_trAISformer import Config
try:
    sys.path.remove('/sanssauvegarde/homes/vnguye04/Codes/DAPPER')
except:
    pass
sys.path.append("..")
import utils
import pickle
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import time
from io import StringIO

from tqdm import tqdm
import argparse

# In[2]:

cf = Config()
#=====================================================================
LAT_MIN,LAT_MAX,LON_MIN,LON_MAX = cf.lat_min,cf.lat_max,cf.lon_min,cf.lon_max

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots
DURATION_MAX = 24 #h

EPOCH = datetime(1970, 1, 1)
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))

FIG_W = 960
FIG_H = int(960*LAT_RANGE/LON_RANGE) #533 #768
def plotPredit(predics, inputs,init_seqlen,count):
    if not os.path.isdir(cf.haitudir):
        os.makedirs(cf.haitudir)
        print('======= Create directory to store trained models: ' + cf.haitudir)
    else:
        print('======= Directory to store trained models: ' + cf.haitudir)
    utils.new_log(cf.haitudir, "log")
    coastline_filename = "./dma_coastline_polygons.pkl"
    coastline_filename = os.path.join(cf.datadir, coastline_filename)
    dict_list = []
    l_pkl_filenames = [cf.trainset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
    for phase, filename in zip(("train"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            Data = pickle.load(f)
            Vs = Data
            FIG_DPI = 150
            plt.figure(figsize=(FIG_W/FIG_DPI, FIG_H/FIG_DPI), dpi=FIG_DPI)
            cmap = plt.cm.get_cmap('Blues')
            # print(Vs['mmsi'])
            # l_keys = list(Vs['mmsi'].keys())
            N = len(Vs)
            for d_i in range(N):
                # key = l_keys[d_i]
                c = cmap(float(d_i)/(N-1))
                tmp = Vs[d_i]
                v_lat = tmp['traj'][:,0]*LAT_RANGE + LAT_MIN
                v_lon = tmp['traj'][:,1]*LON_RANGE + LON_MIN
            #     plt.plot(v_lon,v_lat,linewidth=0.8)
                plt.plot(v_lon,v_lat,color=c,linewidth=0.8)

    with open(coastline_filename, 'rb') as f:
        l_coastline_poly = pickle.load(f)
        for point in l_coastline_poly:
            poly = np.array(point)
            plt.plot(poly[:,1],poly[:,0],color="k",linewidth=0.8)

    v_ranges = torch.tensor([LAT_RANGE, LON_RANGE, 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([LAT_MIN, LON_MIN, 0, 0]).to(cf.device)
    inputs = inputs*v_ranges + v_roi_min
    predics = predics*v_ranges + v_roi_min
    preds_np = predics.detach().cpu().numpy()
    inputs_np = inputs.detach().cpu().numpy()
    img_path = os.path.join(cf.haitudir, f"{count}_haitu.png")
    for idx in range(1):
        c = cmap(float(idx) / (8))
        # plt.subplot(4,2,idx+1)
        plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], color="r")
        plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], "o", markersize=3, color="r")
        plt.plot(inputs_np[idx][:, 1], inputs_np[idx][:, 0], linestyle="-.", color="r")
        plt.plot(preds_np[idx][init_seqlen:, 1], preds_np[idx][init_seqlen:, 0], "x", markersize=4, color="g")
    plt.xlim([LON_MIN,LON_MAX])
    plt.ylim([LAT_MIN,LAT_MAX])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
