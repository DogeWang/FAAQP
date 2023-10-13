import psycopg2
import os
import sys
import pandas as pd
import numpy as np
import csv
from time import perf_counter, sleep
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from ensemble_compilation.physical_db import DBConnection

def save_csv(csv_rows, target_csv_path):
    os.makedirs(os.path.dirname(target_csv_path), exist_ok=True)
    with open(target_csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, csv_rows[0].keys())
        for i, row in enumerate(csv_rows):
            if i == 0:
                w.writeheader()
            w.writerow(row)

def get_query_result():
    query_dict = {}
    file_name = '/home/qym/zhb/RSPN++/benchmarks/job-light/sql/job_light_true_cardinalities.csv'
    if os.path.isfile(file_name):
        csvfile = open(file_name, 'r')
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if row[-1].isdigit():
                query, result = row[1], float(row[-1])
                # print(query)
                query_dict[query] = result
    return query_dict


table_size = 20000000
table_name = 'cast_info'
sample_name = 'cast_info_001p'
table_name_list = ['aka_name', 'aka_title', 'cast_info', 'char_name', 'movie_companies', 'movie_info', 'movie_info_idx',
                   'movie_keyword', 'name', 'person_info', 'title']
# query_no_list = [877, 688, 87, 47, 377, 806, 557, 394, 575, 974, 164, 413, 108, 875, 344, 145, 403, 976, 254, 749, 500, 334, 891, 247, 706, 744, 869, 401, 769, 742, 455, 636, 173, 941, 783, 847, 89, 251, 712, 733, 750, 969, 585, 617, 409, 281, 972, 404, 762, 234, 784, 331, 849, 13, 246, 419, 988, 870, 252, 794, 871, 6, 823, 431, 603, 39, 632, 848, 508, 851, 478, 735, 736, 357, 828, 287, 40, 406, 693, 23, 896, 719, 185, 650, 793, 619, 302, 926, 606, 504, 951, 590, 821, 103, 370, 965, 765, 51, 914, 727]
# query_no_list = [816, 347, 831, 814, 254, 380, 749, 267, 815, 626, 167, 430, 110, 638, 747, 972, 718, 926, 490, 419, 356, 511, 663, 935, 889, 333, 429, 683, 653, 36, 134, 694, 237, 340, 691, 495, 313, 904, 255, 489, 637, 373, 402, 641, 276, 165, 967, 857, 992, 363, 874, 323, 466, 173, 550, 808, 708, 438, 353, 760, 659, 16, 588, 382, 795, 68, 813, 86, 576, 342, 483, 541, 980, 222, 585, 758, 285, 298, 473, 163, 744, 762, 189, 918, 864, 812, 842, 77, 35, 801, 455, 480, 372, 43, 244, 701, 316, 324, 981, 628]
query_no_list = list(range(0,200))
conn = psycopg2.connect(database="imdb", user="postgres", password="postgres", host="127.0.0.1", port="5432")
cur = conn.cursor()
# file_path = "/home/qym/zhb/airbnb/sql/join_DBEst.sql"
file_path = "/home/qym/zhb/RSPN++/benchmarks/job-light/sql/job_light_queries.sql"
file = open(file_path, 'r')
workload = file.readlines()
query_no_list = list(range(0,len(workload)))
csv_rows = []
query_dict = get_query_result()

s_01, s_001, s_0001, s_00001, s_000001 = 0, 0, 0, 0, 0
s_1_l, s_01_l, s_001_l, s_0001_l, s_00001_l, s_000001_l = [], [], [], [], [], []
re_sum, ql_sum = 0, 0
q_error_list = []
num = 0
for i in query_no_list:
    sql = workload[i]
    # print(sql)
    result = query_dict[sql.replace('\n', '')]
    # sql = sql.replace('AVG', 'COUNT')
    # cur.execute(sql)
    # result = float(cur.fetchall()[0][0])
    # if result / table_size <= 0.00001:
    #     s_000001 += 1
    #     s_000001_l.append(i)
    # elif  result / table_size <=  0.0001:
    #     s_00001 += 1
    #     s_00001_l.append(i)
    # elif  result / table_size <= 0.001:
    #     s_0001 += 1
    #     s_0001_l.append(i)
    # elif  result / table_size <= 0.01:
    #     s_001 += 1
    #     s_001_l.append(i)
    # elif result / table_size <= 0.1:
    #     s_01 += 1
    #     s_01_l.append(i)
    # else:
    #     s_1_l.append(i)
    m = 0
    for table_name in table_name_list:
        if table_name in sql.split(' '):
            m += 1

    if m == 1:
        for table_name in table_name_list:
            if table_name in sql.split(' '):
                sql = sql.replace(table_name, table_name + '_001p')
    elif m == 2:
        for table_name in table_name_list:
            if table_name in sql.split(' '):
                sql = sql.replace(table_name, table_name + '_01p')
    else:
        print(sql)

    start_t = perf_counter()
    cur.execute(sql)
    temp = cur.fetchall()[0][0]
    if temp:
        sample_result = float(temp) * 100
    else:
        sample_result = 0
        num += 1
    end_t = perf_counter()
    print(m, sample_result, result)
    if sample_result:
        q_error = max(result / sample_result, sample_result/result)
        re = abs(result - sample_result) / result
    else:
        q_error = result
        re = 1
    print(q_error)
    q_error_list.append(q_error)
    re_sum += re
    ql_sum += end_t - start_t
    csv_rows.append({'approach': 'US',
                     'query_no': i,
                     'latency': end_t - start_t,
                     'average_relative_error': re * 100,
                     'bin_completeness': 0,
                     'total_bins': 0,
                     'query': sql,
                     'sample_percentage': 100
                     })
# save_csv(csv_rows, '../flights-benchmark/shuf_different_size/results/US_10m_avg.csv')
print(num)
print('Selectivity:', s_01, s_001, s_0001, s_00001, s_000001)
print(s_1_l)
print(s_01_l)
print(s_001_l)
print(s_0001_l)
print(s_00001_l)
print(s_000001_l)
print('RE (%):', re_sum / len(query_no_list) * 100)
print('QL (ms):', ql_sum / len(query_no_list) * 1000)

print(np.percentile(np.array(q_error_list), 50))
print(np.percentile(np.array(q_error_list), 90))
print(np.percentile(np.array(q_error_list), 95))
print(np.percentile(np.array(q_error_list), 99))
print(np.percentile(np.array(q_error_list), 100))
print(np.mean(np.array(q_error_list)))
print(sorted(q_error_list))
print(len(q_error_list))

conn.commit()
conn.close()
