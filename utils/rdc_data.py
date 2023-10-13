import psycopg2
import os
import sys
import pandas as pd
import numpy as np
import csv
from time import perf_counter, sleep
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from spn.algorithms.splitting.RDC import getIndependentRDCGroups_py, rdc_test
from spn.structure.StatisticalTypes import MetaType


def rdc_calculate(local_data, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-1, rand_gen=None):
    rdc_adjacency_matrix = rdc_test(
        local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, n_jobs=n_jobs, rand_gen=rand_gen
    )
    rdc_adjacency_matrix[np.isnan(rdc_adjacency_matrix)] = 0
    print('rdc_adjacency_matrix:', rdc_adjacency_matrix)
    return rdc_adjacency_matrix


def ssb_rdc(sql):
    conn = psycopg2.connect(database="ssb1", user="postgres", password="postgres", host="127.0.0.1", port="5432")
    cur = conn.cursor()
    cur.execute(sql)
    result = cur.fetchall()
    # print(result)
    # print(np.array(result))
    conn.commit()
    conn.close()
    return np.array(result)


# meta_types = [MetaType.REAL, MetaType.REAL]
# domains = [[1,10000], [1,10000]]
# print('meta_types:', type(meta_types), meta_types)
# print('domains:', type(domains), domains)
# sql1 = 'select lo_discount, lo_quantity from lineorder, dwdate where lo_orderdate = d_datekey and d_year = 1993 and lo_discount>=1 and lo_discount<=3 and lo_quantity < 25;'
# sql2 = 'select lo_discount, lo_quantity from lineorder, dwdate where lo_orderdate = d_datekey and d_yearmonthnum = 199401 and lo_discount>=4 and lo_discount<=6 and  lo_quantity>=26 and lo_quantity<=35;'
# sql3 = 'select lo_discount, lo_quantity from lineorder, dwdate where lo_orderdate = d_datekey and d_weeknuminyear = 6 and d_year = 1994 and lo_discount>=5 and lo_discount<=7 and lo_quantity>=26 and lo_quantity<=35;'
# # sql = 'select lo_discount, lo_quantity from lineorder;'
# sql = "select p_size, d_year from dwdate, customer, supplier, part, lineorder where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_partkey = p_partkey and lo_orderdate = d_datekey and c_region = 'AMERICA' and s_nation = 'UNITED STATES' and d_year >= 1997 and d_year <= 1998 and p_category = 'MFGR#14';"
# data = ssb_rdc(sql)
# rdc_calculate(data, meta_types, domains)