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

idx = pd.IndexSlice
URI = r"D:\Data\stock_concept.csv"


class Similarity:
    def __init__(self, uri: str = URI):
        self.uri = uri
        self.factor = pd.DataFrame()
        self.prepare_data()

    def _get_code(self, row):
        temp = list(row['code'])
        if isinstance(row['CONCEPT'], str):
            for c in row['CONCEPT'].split(';'):
                temp[self.lab_map[c]] = '1'
            row['code'] = ''.join(temp)
        return row

    @func_timer
    def prepare_data(self):
        stock_concept = pd.read_csv(self.uri, encoding='GB18030').rename(columns={'DATETIME': 'tradedate'})
        stock_concept['tradedate'] = pd.to_datetime(stock_concept['tradedate']).dt.date
        stock_concept.set_index(['tradedate', 'wind_code'], inplace=True)
        label = []
        temp = [label.extend(sc.split(';')) for sc in stock_concept['CONCEPT'] if isinstance(sc, str)]
        del temp
        self.lab_map = {v: k for k, v in enumerate(set(label))}
        # {v: k for k, v in dict(enumerate([i[0] for i in sorted(dict(Counter(label)).items(), key=lambda x: x[1])])).items()}
        stock_concept['code'] = '0' * len(self.lab_map)
        code = stock_concept.apply(lambda x: self._get_code(x), axis=1)[['code']]
        conn = dict(host='10.224.16.81', user='haquant', passwd='haquant', database='jydb', port=3306, charset='utf8')
        bsql = BaseSQL(conn, read_only=True)
        with bsql.start():
            bsql.set_active_table('dailydata')
            cp = bsql.read(field_names=['adjclose'], date_range=dict(start='2017-01-01', end='2022-08-15'))
        ret = cp.reindex(code.index).groupby('wind_code').fillna(method='ffill').groupby('wind_code').pct_change()
        self.fac_data = code.join(ret, how='left').fillna(0)

    @staticmethod
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

    @func_timer
    def cal_factor(self):
        self.factor = self.fac_data.groupby(level=0, group_keys=True).apply(self.corr)

    def save(self, path: str):
        self.factor.to_pickle(path)


if __name__ == '__main__':
    fac = Similarity(URI)
    fac.cal_factor()