#!/usr/bin/env python3
# Basic anomaly detection from Marcos
#
# https://github.com/marcopeix/youtube_tutorials/blob/main/YT_02_anomaly_detection_time_series.ipynb
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

def run_all():
    df = pd.read_csv('data/ec2_cpu_utilization.csv') # Downloaded from the link above
    df.head()

    # Labels taken from the link above. We are looking at the labels for ec2_cpu_utilization_24ae8d dataset
    anomalies_timestamp = [
        "2014-02-26 22:05:00",
        "2014-02-27 17:15:00"
    ]

if __name__ == '__main__':
    print('Running')
    plt.rcParams["figure.figsize"] = (9,6)




# End of script

