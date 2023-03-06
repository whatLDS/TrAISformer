# coding: utf-8

# MIT License
#
# Copyright (c) 2018 Duong Nguyen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
A script to merge AIS messages into AIS tracks.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# sys.path.append("..")
# import utils
import pickle
import copy
import csv
from datetime import datetime
import time
from io import StringIO
from tqdm import tqdm as tqdm

## PARAMS
# ======================================

## Bretagne dataset
# LAT_MIN = 46.5
# LAT_MAX = 50.5
# LON_MIN = -8.0
# LON_MAX = -3.0

# # Pkl filenames.
# pkl_filename = "bretagne_20170103_track.pkl"
# pkl_filename_train = "bretagne_20170103_10_20_train_track.pkl"
# pkl_filename_valid = "bretagne_20170103_10_20_valid_track.pkl"
# pkl_filename_test  = "bretagne_20170103_10_20_test_track.pkl"

# # Path to csv files.
# dataset_path = "./"
# l_csv_filename =["positions_bretagne_jan_mar_2017.csv"]


# # Training/validation/test/total period.
# t_train_min = time.mktime(time.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
# t_train_max = time.mktime(time.strptime("10/03/2017 23:59:59", "%d/%m/%Y %H:%M:%S"))
# t_valid_min = time.mktime(time.strptime("11/03/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
# t_valid_max = time.mktime(time.strptime("20/03/2017 23:59:59", "%d/%m/%Y %H:%M:%S"))
# t_test_min  = time.mktime(time.strptime("21/03/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
# t_test_max  = time.mktime(time.strptime("31/03/2017 23:59:59", "%d/%m/%Y %H:%M:%S"))
# t_min = time.mktime(time.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
# t_max = time.mktime(time.strptime("31/03/2017 23:59:59", "%d/%m/%Y %H:%M:%S"))

# cargo_tanker_filename = "bretagne_20170103_cargo_tanker.npy"

# ## Aruba
LAT_MIN = 9.0
LAT_MAX = 14.0
LON_MIN = -71.0
LON_MAX = -66.0

D2C_MIN = 2000  # meters

# ===============
"""
dataset_path = "./"
l_csv_filename =["aruba_5x5deg_2017305_2018031.csv",
                 "aruba_5x5deg_2018305_2019031.csv",
                 "aruba_5x5deg_2019305_2020031.csv"]
l_csv_filename =["aruba_5x5deg_2017305_2018031.csv"]
pkl_filename = "aruba_20172020_track.pkl"
pkl_filename_train = "aruba_20172020_train_track.pkl"
pkl_filename_valid = "aruba_20172020_valid_track.pkl"
pkl_filename_test  = "aruba_20172020_test_track.pkl"

cargo_tanker_filename = "aruba_20172020_cargo_tanker.npy"

t_train_min = time.mktime(time.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_train_max = time.mktime(time.strptime("31/01/2019 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_valid_min = time.mktime(time.strptime("01/11/2019 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_valid_max = time.mktime(time.strptime("31/12/2019 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_test_min  = time.mktime(time.strptime("01/01/2020 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_test_max  = time.mktime(time.strptime("31/01/2020 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_min = time.mktime(time.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_max = time.mktime(time.strptime("31/01/2020 23:59:59", "%d/%m/%Y %H:%M:%S"))

"""

# ===============
"""
dataset_path = "./"
l_csv_filename =["aruba_zone1_5x5deg_2017121_2017244.csv",
                 "aruba_5x5deg_2018121_2018244.csv",
                 "aruba_zone1_5x5deg_2019121_2019244.csv"]
#l_csv_filename =["aruba_5x5deg_2018121_2018244.csv"]
pkl_filename = "aruba_20172020_summer_track.pkl"
pkl_filename_train = "aruba_20172020_summer_train_track.pkl"
pkl_filename_valid = "aruba_20172020_summer_valid_track.pkl"
pkl_filename_test  = "aruba_20172020_summer_test_track.pkl"

cargo_tanker_filename = "aruba_20172020_summer_cargo_tanker.npy"

t_train_min = time.mktime(time.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_train_max = time.mktime(time.strptime("31/08/2018 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_valid_min = time.mktime(time.strptime("01/05/2019 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_valid_max = time.mktime(time.strptime("31/07/2019 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_test_min  = time.mktime(time.strptime("01/08/2019 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_test_max  = time.mktime(time.strptime("31/08/2019 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_min = time.mktime(time.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_max = time.mktime(time.strptime("31/01/2020 23:59:59", "%d/%m/%Y %H:%M:%S"))
"""

# ===============
"""
dataset_path = "./"
l_csv_filename =["aruba_zone1_5x5deg_2017121_2017244.csv",
                 "aruba_5x5deg_2017305_2018031.csv",
                 "aruba_5x5deg_2018121_2018244.csv",
                 "Aruba_5x5deg_2018305_2019031.csv",
                 "aruba_zone1_5x5deg_2019121_2019244.csv"]
#l_csv_filename =["Aruba_5x5deg_2018305_2019031.csv"]
pkl_filename = "aruba_20172019_track.pkl"
pkl_filename_train = "aruba_20172019_all_train_track.pkl"
pkl_filename_valid = "aruba_20172019_all_valid_track.pkl"
pkl_filename_test  = "aruba_20172019_all_test_track.pkl"

cargo_tanker_filename = "aruba_20172019_all_cargo_tanker.npy"

t_train_min = time.mktime(time.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_train_max = time.mktime(time.strptime("31/01/2019 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_valid_min = time.mktime(time.strptime("01/05/2019 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_valid_max = time.mktime(time.strptime("31/07/2019 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_test_min  = time.mktime(time.strptime("01/08/2019 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_test_max  = time.mktime(time.strptime("31/08/2019 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_min = time.mktime(time.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_max = time.mktime(time.strptime("31/01/2020 23:59:59", "%d/%m/%Y %H:%M:%S"))
"""

# ===============
# ## Est Aruba
LAT_MIN = 10.0
LAT_MAX = 14.0
LON_MIN = -66.0
LON_MAX = -60.0

dataset_path = "./data"
# l_csv_filename =["Est-aruba_5x5deg_2018001_2018120.csv",
#                  "Est-aruba_5x5deg_2018001_2018180.csv",
#                 "Est-aruba_5x5deg_2019240_2019365_.csv"]
l_csv_filename = ["zhoushan_train.csv"]
pkl_filename = "estaruba_20182019_track.pkl"
pkl_filename_train = "zhoushan_train_track.pkl"
pkl_filename_valid = "estaruba_20182019_valid_track.pkl"
pkl_filename_test = "estaruba_20182019_test_track.pkl"

cargo_tanker_filename = "estaruba_20182019_cargo_tanker.npy"

t_train_min = time.mktime(time.strptime("01/01/2018 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_train_max = time.mktime(time.strptime("30/04/2019 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_valid_min = time.mktime(time.strptime("01/09/2019 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_valid_max = time.mktime(time.strptime("30/11/2019 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_test_min = time.mktime(time.strptime("01/12/2019 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_test_max = time.mktime(time.strptime("31/12/2019 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_min = time.mktime(time.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_max = time.mktime(time.strptime("31/01/2020 23:59:59", "%d/%m/%Y %H:%M:%S"))

# ========================================================================
LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SOG_MAX = 30.0  # the SOG is truncated to 30.0 knots max.

EPOCH = datetime(1970, 1, 1)

LAT, LON, SOG, TIMESTAMP, MMSI = list(range(5))

CARGO_TANKER_ONLY = True
if CARGO_TANKER_ONLY:
    pkl_filename = "ct_" + pkl_filename
    pkl_filename_train = "ct_" + pkl_filename_train
    pkl_filename_valid = "ct_" + pkl_filename_valid
    pkl_filename_test = "ct_" + pkl_filename_test

print(pkl_filename_train)

## LOADING CSV FILES
# ======================================
l_l_msg = []  # list of AIS messages, each row is a message (list of AIS attributes)
n_error = 0
for csv_filename in l_csv_filename:
    data_path = os.path.join(dataset_path, csv_filename)
    with open(data_path, "r") as f:
        print("Reading ", csv_filename, "...")
        csvReader = csv.reader(f)
        next(csvReader)  # skip the legend row
        count = 1
        for row in csvReader:
            #             utc_time = datetime.strptime(row[8], "%Y/%m/%d %H:%M:%S")
            #             timestamp = (utc_time - EPOCH).total_seconds()
            print(count)
            count += 1
            try:
                # l_l_msg.append(row)
                l_l_msg.append([float(row[1]),float(row[3]),
                                float(row[4]),
                               int(float(row[2])),
                               int(row[0])])
            except:
                n_error += 1
                continue

m_msg = np.array(l_l_msg)
# del l_l_msg
print("Total number of AIS messages: ", m_msg.shape[0])
print("Lat min: ", np.min(m_msg[:, LAT]), "Lat max: ", np.max(m_msg[:, LAT]))
print("Lon min: ", np.min(m_msg[:, LON]), "Lon max: ", np.max(m_msg[:, LON]))
print("Ts min: ", np.min(m_msg[:, TIMESTAMP]), "Ts max: ", np.max(m_msg[:, TIMESTAMP]))

if m_msg[0, TIMESTAMP] > 1584720228:
    m_msg[:, TIMESTAMP] = m_msg[:, TIMESTAMP] / 1000  # Convert to suitable timestamp format
print("Time min: ", datetime.utcfromtimestamp(np.min(m_msg[:, TIMESTAMP])).strftime('%Y-%m-%d %H:%M:%SZ'))
print("Time max: ", datetime.utcfromtimestamp(np.max(m_msg[:, TIMESTAMP])).strftime('%Y-%m-%d %H:%M:%SZ'))
print("Selecting vessel type ...")
print("Convert to dicts of vessel's tracks...")

# Training set
Vs_train = dict()
for v_msg in tqdm(m_msg):
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_train.keys())):
        Vs_train[mmsi] = np.empty((0, 5))
    Vs_train[mmsi] = np.concatenate((Vs_train[mmsi], np.expand_dims(v_msg, 0)), axis=0)
## PICKLING
#======================================
for filename, filedict in zip([pkl_filename_train],
                              [Vs_train]
                             ):
    print("Writing to ", os.path.join(dataset_path,filename),"...")
    with open(os.path.join(dataset_path,filename),"wb") as f:
        pickle.dump(filedict,f)
    print("Total number of tracks: ", len(filedict))
# print(Vs_train)
