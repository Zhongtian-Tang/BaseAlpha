#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   corr_concept.py
@Time    :   2022/08/18 11:07:15
@Author  :   Liu Junli
@Version :   1.0
@Contact :   liujunli.xmu@foxmail.com
'''

from typing import Callable
import pandas as pd
from collections import Counter
from pprint import pprint
from basesql import BaseSQL
import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
from scipy.spatial.distance import cosine
from tqdm import trange
import multiprocessing as mp

from utils import func_timer

np.seterr(divide='ignore', invalid='ignore')
idx = pd.IndexSlice


def get_code(row):
    temp = list(row['code'])
    if isinstance(row['CONCEPT'], str):
        for c in row['CONCEPT'].split(';'):
            temp[lab_map[c]] = '1'
        row['code'] = ''.join(temp)
    return row


# 标准化
def normalize(x):
    x = (x - x.mean()) / (x.std())
    return x


# label: 所有的标签
URI = r"D:\Data\stock_concept.csv"
stock_concept = pd.read_csv(URI, encoding='GB18030').rename(columns={'DATETIME': 'tradedate'})
stock_concept['tradedate'] = pd.to_datetime(stock_concept['tradedate']).dt.date
stock_concept.set_index(['tradedate', 'wind_code'], inplace=True)

label = []
temp = [label.extend(sc.split(';')) for sc in stock_concept['CONCEPT'] if isinstance(sc, str)]
del temp

#
lab_map = {v: k for k, v in dict(enumerate([i[0] for i in sorted(dict(Counter(label)).items(), key=lambda x: x[1])])).items()}
stock_concept['code'] = '0' * len(lab_map)

factor = stock_concept.apply(lambda x: get_code(x), axis=1)[['code']]

# ret数据
conn = dict(host='10.224.16.81', user='haquant', passwd='haquant', database='jydb', port=3306, charset='utf8')
bsql = BaseSQL(conn, read_only=True)
bsql.start_conn()
bsql.set_active_table('dailydata')
cp = bsql.read(field_names=['adjclose'], date_range=dict(start='2017-01-01', end='2022-08-15'))
bsql.close_conn()
cp = cp.reindex(factor.index)
cp = cp.groupby('wind_code').fillna(method='ffill')
ret = cp.groupby('wind_code').pct_change()
fac_data = factor.join(ret, how='left').fillna(0)


@func_timer
def _corr_(x: pd.DataFrame) -> pd.DataFrame:
    _dum_mat = np.array([[int(_j) for _j in list(_i)] for _i in x['code'].values])
    # corr_ = np.array([[cosine_similarity(i.reshape(1, -1), j.reshape(1, -1))[0][0] for j in _dum_mat] for i in _dum_mat])
    corr_ = np.array([[1 - cosine(i, j) for j in _dum_mat] for i in _dum_mat])
    corr_[np.eye(corr_.shape[0], dtype=np.bool)] = 0  #对角线赋值为0
    corr_[corr_ < np.quantile(corr_, 0.95, axis=1)] = 0  #小于0.95的赋值为0
    out = corr_.dot(x['adjclose'].fillna(0).values.reshape(1, -1).T)
    return pd.DataFrame(out, index=x.index, columns=['fac'])

    # temp = pd.Series(_dum_mat)

    # for i_n, i in enumerate(_dum_mat):
    #     for j_n, j in enumerate(_dum_mat):
    #         print('(%s, %s): %s' % (i_n, j_n, 1 - cosine(i, j)))

    # _dum_mat.T.corr(method=cosine_similarity)
    # cosine_similarity(_dum_mat[0].reshape(1, -1), _dum_mat[0].reshape(1, -1))[0][0]


def corr(x: pd.DataFrame) -> pd.DataFrame:
    _dum_mat = np.array([[int(_j) for _j in list(_i)] for _i in x['code'].values])
    # corr_ = np.array([[cosine_similarity(i.reshape(1, -1), j.reshape(1, -1))[0][0] for j in _dum_mat] for i in _dum_mat])
    corr_ = np.array([[1 - cosine(_dum_mat[i], j) for j in _dum_mat] for i in trange(_dum_mat.shape[0])])
    # corr_ = np.array([[1 - cosine(i, j) for j in _dum_mat] for i in _dum_mat])
    corr_[np.eye(corr_.shape[0], dtype=np.bool_)] = 0  #对角线赋值为0
    corr_[corr_ < np.quantile(corr_, 0.95, axis=1)] = 0  #小于0.95的赋值为0
    corr_ = corr_ / corr_.sum(axis=1)  #分母可能为0
    corr_[~np.isfinite(corr_)] = 0
    out = corr_.dot(x['adjclose'].fillna(0).values)
    return pd.DataFrame(out, index=x.index, columns=['fac'])


out_ = fac_data.groupby(level=0, group_keys=True).apply(corr)
out_.to_pickle(r'D:\Data\fac_data.pkl')
out_.to_csv(r'D:\Data\fac_data.csv')
# date_ = fac_data.index.levels[0].tolist()
# n = fac_data.index.levels[0].shape[0]
# pool = mp.Pool(processes=int(mp.cpu_count() / 2))
# runs = []
# for i in range(n):
#     runs.append(pool.apply_async(corr, (fac_data.loc[idx[date_[i], :], :], )))
# out_ = pd.concat([r.get() for r in runs], axis=0)
# pool.close()
# pool.join()
# out_.to_pickle(r'D:\Data\fac_data.pkl')

x = fac_data.loc[idx[datetime.date(2022, 3, 11), :], :].droplevel(0)
# a = corr(x)
# x['adjclose'].isnull().sum()
# x['adjclose'].fillna(0).values.reshape(1, -1)

# fac = pd.read_pickle(r'D:\Data\fac_data.pkl')

# fac.unstack().plot()
# fac.index.levels[0]
# x = fac.loc[idx[datetime.date(2022, 7, 28), :], :].droplevel(0)
# plt.hist(x, bins=100)
# plt.show()

# fac.replace(0, np.nan)
# x.plot()
# import matplotlib.pyplot as plt

# plt.show()
# fac.index.levels[0][0]
# plt.hist(np.log(fac.replace(0, np.nan)), bins=1000)
# plt.show()