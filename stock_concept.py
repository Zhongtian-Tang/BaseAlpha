#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   stock_concept.py
@Time    :   2022/08/16 08:21:02
@Author  :   Liu Junli
@Version :   1.0
@Contact :   liujunli.xmu@foxmail.com
'''

from typing import Callable
import pandas as pd
from collections import Counter
from pprint import pprint
import numpy as np
from basesql import BaseSQL
import datetime


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


def similarity(x):
    index = x.index
    x = x.values.squeeze()
    # [np.abs([i - j for j in x]) for i in x]
    # return pd.Series([np.nanmean(np.abs([i - j for j in x])) for i in x], index=index)
    return pd.DataFrame([np.abs([i - j for j in x]) for i in x], index=index, columns=index.levels[1])


URI = r"D:\Data\stock_concept.csv"

stock_concept = pd.read_csv(URI, encoding='GB18030')
label = set()
temp = [label.update(sc.split(';')) for sc in stock_concept['CONCEPT'] if isinstance(sc, str)]
del temp

# label = list(set(label))
lab_map = {v: k for k, v in dict(enumerate(label)).items()}
lab_map = {v: k for k, v in dict(enumerate([i[0] for i in sorted(dict(Counter(label)).items(), key=lambda x: x[1])])).items()}
stock_concept['code'] = '0' * len(lab_map)
fac = pd.DataFrame(0, index=stock_concept.index, columns=lab_map.keys())
a = [[0] * len(lab_map)] * len(stock_concept)

con_code = stock_concept.apply(lambda x: get_code(x), axis=1)

con_code['code'] = [int(i, 2) for i in con_code['code']]
con_code.set_index(['DATETIME', 'wind_code'], inplace=True)

_index = pd.MultiIndex.from_product([con_code.index.levels[0].to_list(), con_code.index.levels[1].to_list()], names=['DATETIME', 'wind_code'])
con_code = con_code.reindex(_index).groupby('wind_code').fillna(method='ffill')

code: pd.DataFrame = con_code[['code']].copy()
code['code'] = np.log2(code['code'].values.astype(float) + 1)

c_ = code.groupby('wind_code').mean()
c_.plot()
import matplotlib.pyplot as plt

plt.hist(c_, bins=200)
plt.hist(code['code'].values.flatten(), bins=1000)
plt.show()

factor = code.groupby('DATETIME').apply(normalize)
factor.index.levels[1].shape

conn = dict(host='10.224.16.81', user='haquant', passwd='haquant', database='jydb', port=3306, charset='utf8')
bsql = BaseSQL(conn, read_only=True)
bsql.start_conn()
bsql.set_active_table('dailydata')
cp = bsql.read(field_names=['adjclose'], date_range=dict(start='2017-01-01', end='2022-08-15'))

# cp.to_pickle('test')
# pd.read_pickle('test')
factor.index.names = cp.index.names

factor.reset_index(inplace=True)
factor['tradedate'] = pd.to_datetime(factor['tradedate']).dt.date
factor.set_index(['tradedate', 'wind_code'], inplace=True)

factor.index.levels[0]

cp = cp.reindex(factor.index)
cp = cp.groupby('wind_code').fillna(method='ffill')
ret = cp.groupby('wind_code').pct_change()

fac_data = factor.join(ret, how='left')


def _func(x: pd.DataFrame):
    index = x.index
    x: pd.DataFrame
    _f = x['code'].values
    _mat = pd.DataFrame([np.abs([i - j for j in _f]) for i in _f])
    _mat = (_mat - _mat.min(axis=1)) / (_mat.max(axis=1) - _mat.min(axis=1))
    _mat = 1 - _mat
    _mat[_mat < _mat.quantile(0.95, axis=1)] = 0
    _mat = _mat / _mat.sum(axis=1)
    _ret = x['adjclose'].fillna(0).values
    _mat = _mat.fillna(0)
    a = _mat.values.dot(_ret)
    b = pd.DataFrame(a).replace(0, np.nan)
    plt.hist(b, bins=100)
    plt.show()

    np.isnan(np.percentile(_mat, 50, axis=1)).sum()

    _mat.dot(x['adjclose'].values)
    np.isnan(x['adjclose'].values).sum()

    _mat.size - np.isnan(_mat).sum()
    _mat.size
    np.isnan(_f).sum()

    _f.size
    (4942 - 1736)**2

    for i in range(len(_t)):
        for j in range(i + 1, len(_t)):
            if _t[i] == _t[j]:
                _t[j] = 0
    # [np.abs([i - j for j in x]) for i in x]
    # return pd.Series([np.nanmean(np.abs([i - j for j in x])) for i in x], index=index)
    return pd.DataFrame([np.abs([i - j for j in x]) for i in x], index=index, columns=['fac'])


def myfunc(x: pd.DataFrame):
    # groupby是注意group_keys应该为False
    _f = x['code'].values
    _mat = pd.DataFrame([np.abs([i - j for j in _f]) for i in _f])
    _mat = 1 - (_mat - _mat.min(axis=1)) / (_mat.max(axis=1) - _mat.min(axis=1))
    _mat[_mat < _mat.quantile(0.95, axis=1)] = 0
    _mat = _mat / _mat.sum(axis=1)
    return pd.DataFrame(_mat.fillna(0).values.dot(x['adjclose'].fillna(0).values), index=x.index, columns=['fac']).replace(0, np.nan)


idx = pd.IndexSlice
x = fac_data.loc[idx[datetime.date(2017, 4, 28), :], :].droplevel(0)
fac_data.index.levels[0]

out = fac_data.groupby('tradedate', group_keys=False).apply(myfunc)

out.columns = ['similarity']
out.to_pickle(r"D:\Data\similarity_0818.pkl", compression='bz2')

plt.hist(out, bins=100)
plt.show()

factor
factor_ = factor.groupby('DATETIME', group_keys=False).apply(similarity)

factor_.where(factor_ < factor_.quantile(0.05), 0)

factor_.index.names = ['tradedate', 'wind_code']
factor_ = factor_.to_frame('similarity')
factor_.to_pickle(r"D:\Data\similarity.pkl", compression='bz2')

_d = pd.read_pickle(r"D:\Data\similarity_0818.pkl", compression='bz2')
_d.unstack()