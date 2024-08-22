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
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings('ignore')

def run_all():
    df = pd.read_csv('data2.csv') # Downloaded from the link above
    df.head()

    # Set the column for all rows to 1 (is good), -1 indicates anomaly
    df['is_anomaly'] = 1

    # Find anom by request time val
    df.loc[df['value'] > 2000.0, 'is_anomaly'] = -1

    # Normalize
    min_val = df['value'].min()
    max_val = df['value'].max()

    # Min-Max Normalization
    normalized_data = (df['value'] - min_val) / (max_val - min_val)
    print(normalized_data)
    print(df)

    # Labels taken from the link above. We are looking at the labels for ec2_cpu_utilization_24ae8d dataset
    # Try removing these or changing the values
    # Test with 0 and 7814
    anomalies_ids = [
        28,
        6203,
        0,
        7814,
        390
    ]

    print('>>>')
    print(df.iloc[0])
    print(df.iloc[7814])

    # Translate the timestamp format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.head()


    # Find the two anomalies
    # df.loc[df['timestamp'] == each, 'is_anomaly'] = -1  
    # You can also use df.iloc[row, column_index] for positional access
    #for row in anomalies_ids:        
    #    df.loc[row, 'is_anomaly'] = -1  
    

    df.head()
    anomaly_df = df.loc[df['is_anomaly'] == -1]
    inlier_df = df.loc[df['is_anomaly'] == 1]

    fig, ax = plt.subplots()

    ax.scatter(inlier_df.index, inlier_df['value'], color='blue', s=3, label='Inlier')
    ax.scatter(anomaly_df.index, anomaly_df['value'], color='red', label='Anomaly')
    ax.set_xlabel('Time')
    ax.set_ylabel('CPU Usage')
    ax.legend(loc=2)

    plt.grid(False)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    # Continue with Baseline - median absolute deviation

    sns.kdeplot(df['value']);
    plt.grid(False)
    plt.axvline(0.134, 0, 1, c='black', ls='--')
    plt.tight_layout()
    plt.show()

    mad = median_abs_deviation(df['value'])
    median = np.median(df['value'])

    print(median)
    print(mad)

    def compute_robust_z_score(x):
        return .6745*(x-median)/mad

    df['z-score'] = df['value'].apply(compute_robust_z_score)
    df.head()

    # Set entire column baseline to 1
    # Do preinitialization with the zscore
    df['baseline'] = 1
    df.loc[df['z-score'] >= 3.5, 'baseline'] = -1
    df.loc[df['z-score'] <=-3.5, 'baseline'] = -1

    l1 = len(df[:6050])
    l2 = len(df[6050:])
    print('Testing Isolation Forest')
    print(' -> Length of DF before to train - ' + str(l1))
    print(' -> Length of DF before to test - ' + str(l2))

    # Build out the training and test data
    train = df[:3550]
    test = df[3550:]
    contamination = 1/len(train)

    # Print final table
    # Final rows: timestamp  value  is_anomaly  z-score  baseline
    print(df)

    d3 = test.loc[test['is_anomaly'] == -1]
    print(d3)

    # Unsupervised learning to isolated anomalies
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    X_train = train['value'].values.reshape(-1,1)
    iso_forest.fit(X_train)

    preds_iso_forest = iso_forest.predict(test['value'].values.reshape(-1,1))
    cm = confusion_matrix(test['is_anomaly'], preds_iso_forest, labels=[1, -1])

    # Show but still failed
    disp_cm = ConfusionMatrixDisplay(cm, display_labels=[1, -1])
    disp_cm.plot();
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # Local density of point to density of neighbors
    # Must be an outlier
    # Show confusion matrix
    lof = LocalOutlierFactor(contamination=contamination, novelty=True)
    lof.fit(X_train)
    preds_lof = lof.predict(test['value'].values.reshape(-1,1))
    cm = confusion_matrix(test['is_anomaly'], preds_lof, labels=[1, -1])
    disp_cm = ConfusionMatrixDisplay(cm, display_labels=[1, -1])
    disp_cm.plot()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print('Running')
    plt.rcParams["figure.figsize"] = (9, 6)
    run_all()

# End of script

