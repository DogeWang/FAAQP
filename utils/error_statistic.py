import pandas as pd
import os
import sys
import csv
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


def error_percentile(target_path):
    file_name = target_path
    csvfile = open(file_name, 'r')
    reader = csv.reader(csvfile, delimiter=',')
    error_list = []
    for row in reader:
        if row[-1].replace('.', '', 1).isdigit():
            error_list.append(float(row[3]))
    for i in range(len(error_list)):
        if error_list[i] == 100:
            error_list[i] = 3
        # error_list.remove(100)
    print(np.percentile(np.array(error_list), 50))
    print(np.percentile(np.array(error_list), 90))
    print(np.percentile(np.array(error_list), 95))
    print(np.percentile(np.array(error_list), 99))
    print(np.percentile(np.array(error_list), 100))

# file_name = './baselines/aqp/results/deepDB/flights_origin_model_based_hc_100_qe.csv'
# # file_name = '/home/qym/zhb/deepdb-public/baselines/aqp/results/deepDB/flights_q_error.csv'
# error_percentile(file_name)