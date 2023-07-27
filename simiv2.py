from typing import Callable, List
import pandas as pd
from collections import Counter
from pprint import pprint
from basesqlv3 import BaseSQL
import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
from scipy.spatial.distance import cosine
from tqdm import trange
import multiprocessing as mp
import time
from utils import func_timer
from pandarallel import pandarallel
import swifter

pandarallel.initialize(progress_bar=True)

idx = pd.IndexSlice
URI = r"D:\Data\stock_concept.csv"


def corr(x: pd.DataFrame) -> pd.DataFrame:

    _dum_mat = np.array([[int(_j) for _j in list(_i)] for _i in x['code'].values])
    # corr_ = np.array([[cosine_similarity(i.reshape(1, -1), j.reshape(1, -1))[0][0] for j in _dum_mat] for i in _dum_mat])
    corr_ = np.array([[1 - cosine(_dum_mat[i], j) for j in _dum_mat] for i in trange(_dum_mat.shape[0])])
    # corr_ = np.array([[1 - cosine(i, j) for j in _dum_mat] for i in _dum_mat])
    corr_[np.eye(corr_.shape[0], dtype=np.bool_)] = 0  #对角线赋值为0
    corr_[corr_ < np.quantile(corr_, 0.95, axis=1)] = 0  #小于0.95的赋值为0
    corr_ = corr_ / corr_.sum(axis=1)
    out = corr_.dot(x['adjclose'].fillna(0).values)
    return pd.DataFrame(out, index=x.index, columns=['fac'])


def _corr(x: pd.DataFrame) -> pd.DataFrame:
    # x = fac_data.xs(datetime.date(2022, 7, 28))
    import numpy as np
    import pandas as pd
    ix = x['fac'].values
    corr_ = np.array([[len(set(i) & set(j)) for i in ix] for j in ix])
    corr_[np.eye(corr_.shape[0], dtype=np.bool_)] = 0
    corr_ = corr_ / corr_.sum(axis=1)
    corr_[np.eye(corr_.shape[0], dtype=np.bool_)] = -1
    out = corr_.dot(x['adjclose'].fillna(0).values)
    return pd.DataFrame(out, index=x.index, columns=['fac'])


def resample(data: pd.DataFrame, freq: str = 'M'):
    data = cp
    date_idx = pd.DatetimeIndex(data.index.get_level_values('tradedate').unique())
    year = date_idx.year
    if freq == 'D':
        freq = date_idx.dayofyear
    elif freq == 'Y':
        freq = date_idx.year
    elif freq == 'M':
        freq = date_idx.month
    elif freq == 'W':
        year = date_idx.isocalendar().year  #每年的最后一天不一定是当年（有可能是下一年）的星期，最开始一天不一定是当年（有可能是上一年）的星期
        freq = date_idx.isocalendar().week
    elif freq == 'Q':
        freq = date_idx.quarter
    date_idx = date_idx.to_frame().groupby([year, freq], as_index=False).last().set_index('tradedate')

    data = data.reindex(date_idx.index, level='tradedate')
    data.reset_index(inplace=True)
    data['tradedate'] = pd.to_datetime(data['tradedate']).dt.date
    data.set_index(['tradedate', 'wind_code'], inplace=True)
    return data


if __name__ == '__main__':
    # load adjclose
    #****************************************************************************************************
    conn = dict(host='10.224.16.81', user='haquant', passwd='haquant', database='jydb', port=3306, charset='utf8')
    bsql = BaseSQL(conn, read_only=True)
    with bsql.start():
        bsql.set_active_table('dailydata')
        cp = bsql.read(field_names=['adjclose'], date_range=dict(start='2017-01-01', end='2022-08-15'))
    cp = resample(cp, 'D')
    ret = cp.groupby('wind_code').pct_change().fillna(0)
    #****************************************************************************************************

    #****************************************************************************************************
    stock_concept = pd.read_csv(URI, encoding='GB18030').rename(columns={'DATETIME': 'tradedate'})
    stock_concept['tradedate'] = pd.to_datetime(stock_concept['tradedate']).dt.date
    stock_concept.set_index(['tradedate', 'wind_code'], inplace=True)
    fac = stock_concept['CONCEPT'].fillna('').astype(str).apply(lambda x: x.split(';')).to_frame('fac')
    #****************************************************************************************************

    fac_data = pd.concat([fac, ret], axis=1, join='outer')
    fac_data = fac_data.groupby('wind_code').fillna(method='ffill').reindex(ret.index)
    fac_data = fac_data.dropna(how='any', axis=0)
    # fac_data['fac'] = fac_data['fac'].fillna('')

    # isstr = lambda x: isinstance(x, list)

    # fac_data['fac'][~fac_data['fac'].apply(isstr).sum()]

    # fac_data['fac'].apply(isstr).sum()
    factor = fac_data.groupby('tradedate', group_keys=False).parallel_apply(_corr)

    # temp = fac_data.loc[idx[datetime.date(2022, 8, 15), :], :]
    # a = temp.groupby('tradedate', group_keys=False).apply(_corr)
    # temp.isnull().sum()

    # temp=fac_data.loc[]

    # factor = fac_data.groupby('tradedate', group_keys=False).swifter.apply(_corr)
    factor.to_csv('factor0905_d.csv')
    #****************************************************************************************************
    # 草稿纸

    #****************************************************************************************************
    # print('*' * 100)
    # label = []
    # temp = [label.extend(sc.split(';')) for sc in stock_concept['CONCEPT'] if isinstance(sc, str)]
    # del temp
    # lab_map = {v: k for k, v in enumerate(set(label))}
    # # {v: k for k, v in dict(enumerate([i[0] for i in sorted(dict(Counter(label)).items(), key=lambda x: x[1])])).items()}
    # stock_concept['code'] = '0' * len(lab_map)
    # code = stock_concept.apply(lambda x: _get_code(x), axis=1)[['code']]
    # fac = stock_concept['CONCEPT']
    #****************************************************************************************************

    factor = pd.read_csv('factor0905_d.csv')
    factor['tradedate'] = pd.to_datetime(factor['tradedate']).dt.date
    factor.set_index(['tradedate', 'wind_code'], inplace=True)
    # factor = factor['fac'].unstack().iloc[1:-1, :].dropna(how='any', axis=1).stack().to_frame('fac')
