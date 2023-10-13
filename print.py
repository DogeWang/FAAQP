import csv
import logging
import pickle
from enum import Enum
from time import perf_counter, sleep
import numpy as np
import pandas as pd
import math
import copy
import os
import sys
import gevent
import threading
import random
import pdb

from ensemble_compilation.graph_representation import Query, QueryType, AggregationType, AggregationOperationType
from ensemble_compilation.physical_db import DBConnection
from ensemble_compilation.spn_ensemble import read_ensemble
from evaluation.utils import parse_query, save_csv
from spn.structure.Base import bfs
from spn.structure.Base import Product
from rspn.structure.base import Sum
from roaringbitmap import RoaringBitmap
from multiprocessing import Pool
from joblib import Parallel, delayed
from gevent.pool import Pool as GPool
from rspn.structure.leaves import Categorical, IdentityNumericLeaf
from utils.toolkit import id_process, id_process2, BitMap

# from multiprocessing import SharedMemoryManger

# a = RoaringBitmap(range(10))
ensemble_location = '../flights-benchmark/spn_ensembles/ensemble_single_flights_origin_5000000_RoaringBitmap_IdentityNumericLeafwithList.pkl'
ensemble_location = '/home/qym/zhb/airbnb/spn_ensembles/ensemble_relationships_airbnb_200000000.pkl'

spn_ensemble = read_ensemble(ensemble_location, build_reverse_dict=True)
count_1, count_2, count_3 = 0, 0, 0
rora_list = []
id_list = []
# _sum, _sum_hy, _sum_bitmap, _sum_ro = 0, 0, 0, 0
# bitmap_size = 5000000
# ro_dict = {}
#
#
def printNode(node):
    global count_1, count_2, count_3
    # if isinstance(node, Sum):
    #     print("bfs - sum node:", node)
    #     print("bfs - sum node.scope:", node.scope)
    #     print("bfs - sum node.weights:", node.weights)
    #     print("bfs - sum node.children:", node.children)
    #     print("bfs - sum node.cluster_centers:", node.cluster_centers)
    #     print("bfs - sum node.cardinality:", node.cardinality)
    # if isinstance(node, Product):
    #     print("bfs - product node:", node)
    #     print("bfs - product node.scope:", node.scope)
    #     print("bfs - product node.children:", node.children)
    # el
    if isinstance(node, IdentityNumericLeaf):
        print("bfs - leaf node:", node)
        print("bfs - leaf node.scope:", node.scope, node.pre_ids_bitmap)
        # exit(1)
        # count += len(node.pre_id_bitmap)
        # if node.ids_histogram_level == 1:
        #     count_1 += 1
        # elif node.ids_histogram_level == 2:
        #     count_2 += 1
        # elif node.ids_histogram_level == 3:
        #     count_3 += 1
        # temp = RoaringBitmap()
        # l = []
        # for key, values in node.unique_vals_ids.items():
        #     print(key, len(values[0]), len(values[1]))
        #     temp |= values
        #     l.append(values)
        # rora_list.append(l)
        # id_list.append(list(temp))
        #     if isinstance(key, tuple):
        #         count_1 += len(key)
                # print("bfs - leaf node.unique_vals_ids:", node.unique_vals_ids)
                # exit(1)
    #     # for key, values in node.unique_vals_ids.items():
    #     #     temp_key = str(node.scope) + str(key)
    #     #     if temp_key in ro_dict.keys():
    #     #         ro_dict[temp_key] |=  RoaringBitmap(values)
    #     #     else:
    #     #         ro_dict[temp_key] =  RoaringBitmap(values)
    #     #     _sum_ro += RoaringBitmap(values).__sizeof__()
    #     #     _sum += len(values) * values[0].__sizeof__()
    #     #     if bitmap_size < len(values) * values[0].__sizeof__():
    #     #         count += 1
    #     #         _sum_hy += bitmap_size
    #     #         _sum_bitmap += bitmap_size
    #     #     else:
    #     #         _sum_hy += len(values) * values[0].__sizeof__()


# def pre_c(a, b):
#     f = open('/home/qym/zhb/imdb-benchmark/gen_single_light/meta_data.pklcd','rb')
#     data = pickle.load(f)
#     pdb.set_trace()
#     temp = 'title.' + a
#     temp = temp[0: -1]
#     #pdb.set_trace()
#     #print(data['calendar']['categorical_columns_dict']['calendar.c_date']['2023-03-06'])
#     return data['title']['categorical_columns_dict'][temp][b]
#     #pdb.set_trace()

# pre_c(1,2)
start_t = perf_counter()
for spn in spn_ensemble.spns:
    print(spn)
    # bfs(spn.mspn, printNode)
# print(count_1)
# end_t = perf_counter()
# print(end_t - start_t)
# print(len(rora_list), len(id_list))
#
# max_array = []
# for i in range(len(id_list)):
#     start_t = perf_counter()
#     st = len(id_list[i])
#     b = BitMap()
#     b.load(id_list[i][:int(st * 0.9)])
#     end_t = perf_counter()
#     time_t = end_t - start_t
#     max_array.append(time_t)
# print('Bitmap max&min :', max(max_array), min(max_array))
#
# max_array = []
# for i in range(len(id_list)):
#     start_t = perf_counter()
#     st = len(id_list[i])
#     RoaringBitmap(id_list[i][:int(st * 0.9)])
#     end_t = perf_counter()
#     time_t = end_t - start_t
#     max_array.append(time_t)
# print('Roaring_Bitmap max&min :', max(max_array), min(max_array))
#
# start_t = perf_counter()
# for i in range(len(id_list)):
#     st = len(id_list[i])
#     RoaringBitmap(id_list[i][:int(st * 0.9)])
# end_t = perf_counter()
# time_t = end_t - start_t
# print('No Copy:', time_t, time_t / len(id_list))
#
# start_t = perf_counter()
# for i in range(len(id_list)):
#     st = len(id_list[i])
#     id_list[i][:int(st * 0.9)]
# end_t = perf_counter()
# time_t = end_t - start_t
# print('Only Get:', time_t, time_t / len(id_list))
#
# max_array = []
# for i in range(len(id_list)):
#     start_t = perf_counter()
#     st = len(id_list[i])
#     id_list[i][:int(st * 0.9)]
#     end_t = perf_counter()
#     time_t = end_t - start_t
#     max_array.append(time_t)
# print('Only Get max&min :', max(max_array), min(max_array))
#
# max_array = []
# for i in range(len(id_list)):
#     start_t = perf_counter()
#     st = len(id_list[i])
#     RoaringBitmap(id_list[i][:int(st * 1)])
#     end_t = perf_counter()
#     time_t = end_t - start_t
#     max_array.append(time_t)
# # print('1:', time_t, time_t / len(id_list))
# print('1 max&min :', max(max_array), min(max_array))
#
# start_t = perf_counter()
# for i in range(len(id_list)):
#     st = len(id_list[i])
#     RoaringBitmap(id_list[i][:int(st * 1)])
# end_t = perf_counter()
# time_t = end_t - start_t
# print('1:', time_t, time_t / len(id_list))
#
# start_t = perf_counter()
# for i in range(len(id_list)):
#     st = len(id_list[i])
#     RoaringBitmap(id_list[i][:int(st * 0.3)])
# end_t = perf_counter()
# time_t = end_t - start_t
# print('0.3:', time_t, time_t / len(id_list))
#
# start_t = perf_counter()
# for i_l in rora_list:
#     temp = RoaringBitmap()
#     for i in range(len(i_l)):
#         temp |= i_l[i]
# end_t = perf_counter()
# time_t = end_t - start_t
# print('Bitmap Union:', time_t, time_t / len(rora_list))
#
# max_array = []
# for i_l in rora_list:
#     start_t = perf_counter()
#     temp = RoaringBitmap()
#     for i in range(len(i_l)):
#         temp |= i_l[i]
#     end_t = perf_counter()
#     time_t = end_t - start_t
#     max_array.append(time_t)
# print('Union max&min :', max(max_array), min(max_array))


# if __name__ == '__main__':
#     start_t = perf_counter()
#     for spn in spn_ensemble.spns:
#         bfs(spn.mspn, printNode)
#     end_t = perf_counter()
#     print(end_t - start_t)
#     print(len(rora_list), len(id_list))
#
#     # start_t = perf_counter()
#     # for i in range(10):
#     #     for i_l in rora_list:
#     #         temp = RoaringBitmap()
#     #         for j in range(len(i_l)):
#     #             temp |= i_l[j]
#     # end_t = perf_counter()
#     # time_t = end_t - start_t
#     # print('Bitmap Union:', time_t, time_t / len(rora_list))
#
#     k = 16
#     long = int(len(rora_list) / k)
#     r_l = []
#     for i in range(k):
#         temp = []
#         for j in rora_list[long * i: long * (i + 1)]:
#             temp.append(j)
#         r_l.append(temp)
#     rora_list = r_l
#     print(len(rora_list))
#
#     pool = Pool(2)
#     start_t = perf_counter()
#     res = pool.map_async(id_process, rora_list).get()
#     # pool.close()
#     # pool.join()
#     end_t = perf_counter()
#     print('Pool 2:', end_t - start_t)
#
#     pool = Pool(4)
#     start_t = perf_counter()
#     res = pool.map_async(id_process, rora_list).get()
#     # pool.close()
#     # pool.join()
#     end_t = perf_counter()
#     print('Pool 4:', end_t - start_t)
#
#     pool = Pool(16)
#     start_t = perf_counter()
#     res = pool.map_async(id_process, rora_list).get()
#     pool.close()
#     # pool.join()
#     end_t = perf_counter()
#     print('Pool 16:', end_t - start_t)
#
#     pool = Pool(32)
#     start_t = perf_counter()
#     res = pool.map_async(id_process, rora_list).get()
#     pool.close()
#     pool.join()
#     end_t = perf_counter()
#     print('Pool 32:', end_t - start_t)
#
#     # pool = Pool(64)
#     # start_t = perf_counter()
#     # pool.map_async(id_process, rora_list)
#     # pool.close()
#     # pool.join()
#     # end_t = perf_counter()
#     # print('Pool 64:', end_t - start_t)



# print(len(temp))
# # print(count_1)
# # print(count_1, count_2, count_3)
# # print(count)b =
# # print(count, _sum / 1024 / 1024, _sum_hy / 1024 / 1024, _sum_bitmap / 1024 / 1024)
# # print(_sum_ro / 1024 / 1024)
# #
# # # for key, values in ro_dict.items():
# # #     ro_dict[key] = np.array(values)
# #
# # with open('/home/qym/zhb/RSPN++/test.pkl', 'wb') as f:
# #     pickle.dump(ro_dict, f, pickle.HIGHEST_PROTOCOL)
# # b = np.array(range(10000))
# # # a = RoaringBitmap(random.sample(list(b), 50000))
# # a = RoaringBitmap(range(1000))
# # c = RoaringBitmap(random.sample(list(b), 200))
# # start_t = perf_counter()
# # for i in range(10000):
# #     a |= c
# # end_t = perf_counter()
# # print(end_t - start_t)
# def id_process(values1):
#     for a in range(100):
#         for b in range(200):
#             for c in range(1000):
#                 values1.append(a*b*c)
#     return values1
#
# pool = Pool(16)
# id_list = [[], [], [], [], [], [], [], []]
#
# start_t = perf_counter()
# pool.map_async(id_process, id_list)
# pool.close()
# pool.join()
# end_t = perf_counter()
# print(end_t - start_t)
# #
# id_list = [[], [], [], [], [], [], [], []]
# start_t = perf_counter()
# for i in range(len(id_list)):
#     id_process(id_list[i])
# # thread = {}
# # for i in range(len(id_list)):
# #     thread[i] = threading.Thread(target=id_process, args=(id_list[i],))
# #     thread[i].run()
# end_t = perf_counter()
# print(end_t - start_t)
# #
# # id_list = [[], [], [], [], [], [], [], []]
# # start_t = perf_counter()
# # for i in id_list:
# #     id_process(i)
# # end_t = perf_counter()
# # print(end_t - start_t)
# # a = np.array(range(20))
# # b = np.array(range(20))
# #
# # start_t = perf_counter()
# # a = list(a)
# # b = list(b)
# # c = 0
# # for i in range(len(a)):
# #     c += a[i] * b[i]
# # end_t = perf_counter()
# # print('pool:', end_t - start_t)
# #
# # pool = Pool(4)
# # start_t = perf_counter()
# # # b = pool.starmap(id_process, zip(list(a.values())[::2], list(a.values())[::-1][::2]))
# # b = pool.starmap(id_process_1, zip(list(a.values())[::2], list(a.values())[::-1][::2]))
# # end_t = perf_counter()
# # pool.close()
# #
# # pool.join()
# #
# # print('pool:', end_t - start_t)
# #


