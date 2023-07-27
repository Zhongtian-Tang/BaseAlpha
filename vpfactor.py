#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vpfactor.py
@Time    :   2022/08/15 14:09:13
@Author  :   Liu Junli
@Version :   1.0
@Contact :   liujunli.xmu@foxmail.com
'''

import time
import pandas as pd
import numpy as np
from WindPy import w
from typing import List, Callable, Union, Optional, Tuple, Dict
from functools import wraps, lru_cache
from pyfinance.ols import PandasRollingOLS
from statsmodels.regression.rolling import RollingWLS as RWLS
from datetime import date, timedelta

from basesql import BaseSQL, WriteSQLV2
from config import Config
from logger import get_module_logger, set_log_with_config
from utils import func_timer

set_log_with_config()
idx = pd.IndexSlice
try:
    w.start()
    time.sleep(5)
except:
    raise Exception('wind连接失败!')
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

default_config = {
    'start': '2020-01-01',  # str(date.today() - timedelta(days=365)),
    'end': str(date.today()),
    'connect': dict(host='10.224.1.70', user='liujl', passwd='CEQZqwer', database='jydb'),
    'write_c': dict(host='10.224.1.70', user='liujl', passwd='CEQZqwer', database='liujl'),
    'write_tab': 'tl_factor'
}

C = Config(default_config)


def fac_label(addsuffix: bool = False):
    def decorate(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            col_names = func.__name__.replace('_', '')
            if addsuffix:
                if 'window' in kwargs.keys():
                    col_names += str(kwargs['window'])
                elif len(args) > 0 and isinstance(args[1], int):
                    col_names += str(args[1])
            out.columns = [col_names]
            return out

        return wrapper

    return decorate


class VPFactor(BaseSQL):
    # data: pd.DataFrame();
    # index: MultiIndex: (tradedate, wind_code)
    # columns: 原始数据字段名, 例如, ['close', 'open', 'high', 'low', 'volume']
    # *计算因子函数的第一个参数最好是 window
    cols_abbr = {
        'openprice': 'open',
        'closeprice': 'close',
        'highprice': 'high',
        'lowprice': 'low',
        'turnovervolume': 'vol',
        'turnovervalue': 'amount',
        'turnoverrate': 'tor',
        'ratioadjustingfactor': 'adj',
        'changepct': 'ret',
        'afloats': 'af',
        'avgprice': 'avg',
    }
    data_cols = list(cols_abbr.keys())
    table = {
        'i': 'incomestate',
        'c': 'cashflowstate',
        'b': 'balancesheet',
        'md': 'maindata',
        'mi': 'mainindex',
        'd': 'dailydata',
        's': 'stockvaluation',
    }

    def __init__(self, read_only: bool = False) -> None:
        super().__init__(C.connect, read_only=read_only)
        self.logger = get_module_logger('file.vpfactor')
        self.data = pd.DataFrame()
        self.start_conn()
        self.data = self._load_data(self.data_cols, 'd')
        self._preprocess()

    def _load_data(self,
                   fields: List[str],
                   table_name: str,
                   date_range: Optional[Union[str, list, Tuple, Dict[str, str]]] = None,
                   other_fields: Optional[Dict[str, Union[str, list, Tuple]]] = None,
                   other_cond: Optional[str] = None,
                   expand: bool = False) -> pd.DataFrame:

        assert table_name and table_name in self.table.keys(), 'table_name must be in {}'.format(self.table.keys())
        if date_range is None:
            date_range = dict(start=C.start, end=C.end)
        out = self.read(self.table[table_name], fields, date_range, other_fields, other_cond)
        out = out.reset_index().rename(columns={
            self.list_date_key(self.table[table_name]): 'tradedate'
        }).set_index(['tradedate', 'wind_code']).sort_index()

        if expand:
            _date = pd.date_range(*self.data.index.levels[0][[0, -1]], freq='D').date
            out = out.unstack().reindex(_date).fillna(method='ffill').stack().reindex(self.data.index)
        return out

    def _preprocess(self) -> None:
        # *字段名的简化: 如, openprice -> open
        # *一些基础的计算: 复权后的价格
        self.data.rename(columns=self.cols_abbr, inplace=True)
        self.data['close_adj'] = self.data['close'] * self.data['adj']

    @func_timer
    def cal_factor(self):
        out = [
            self._ILLIQUIDITY(window=20),
            self._SKEWNESS(window=20),
            self._VOLATILITY(window=20),
            self._WVAD(window=24),
            self._MONEYFLOW(window=20),
            self._TVMA(window=20),
            self._TVMA(window=6),
            self._TVSTD(window=20),
            self._TVSTD(window=6),
            self._VEMA(window=5),
            self._VEMA(window=10),
            self._VEMA(window=12),
            self._VEMA(window=26),
            self._VSTD(window=10),
            self._VSTD(window=20),
            self._GAINVARIANCE(window=120),
            self._PEHIST(window=120),
            self._PEHIST(window=60),
            self._TOBT(window=504),
            self._OPERATINGPROFITPSLATEST(),
            self._ROECUT()
        ]
        self.factor = pd.concat(out, axis=1).sort_index().replace([np.inf, -np.inf], None)
        del out

    @func_timer
    @fac_label()
    def _ILLIQUIDITY(self, window: int):
        # def _illiquidity(df: pd.DataFrame, window: int):
        #     df = df.abs().rolling(window).sum()
        #     return (df['ret'] / df['amount']).to_frame('fac')
        # self.data[['ret', 'amount']].groupby('wind_code').apply(_illiquidity, window=window) * 1e7
        temp = self.data[['ret', 'amount']].abs().groupby('wind_code', group_keys=False).rolling(window).sum().droplevel(0).sort_index()
        return (temp['ret'] / temp['amount']).to_frame('fac') * 1e7

    @func_timer
    @fac_label()
    def _SKEWNESS(self, window: int = 20):
        # return self.data[['close']].groupby('wind_code').apply(lambda df: df.rolling(window).skew())
        return self.data[['close']].groupby('wind_code', group_keys=False).rolling(window).skew().droplevel(0).sort_index()

    @func_timer
    @fac_label()
    def _VOLATILITY(self, window: int):
        # return self.data[['tor']].groupby('wind_code').apply(lambda df: df.rolling(window).std() / df.rolling(window).mean())
        temp = self.data[['tor']].groupby('wind_code').rolling(window).std()
        return (temp / self.data[['tor']].groupby('wind_code').rolling(window).mean()).droplevel(0).sort_index()

    @func_timer
    @fac_label()
    def _WVAD(self, window: int = 24):
        # def _wvad(df: pd.DataFrame, window: int):
        #     wvad = (df['close'] - df['open']) / (df['high'] - df['low']) * df['vol']
        #     return wvad.rolling(window).sum().to_frame('fac')
        # return self.data[['open', 'close', 'high', 'low', 'vol']].groupby('wind_code').apply(_wvad, window) / 1e6
        temp = (self.data['close'] - self.data['open']) / (self.data['high'] - self.data['low']) * self.data['vol']
        return temp.groupby('wind_code').rolling(window).sum().droplevel(0).sort_index().to_frame('fac') / 1e6

    @func_timer
    @fac_label(addsuffix=True)
    def _MONEYFLOW(self, window: int = 20):
        # def _moneyflow(df: pd.DataFrame, window: int):
        #     moneyflow = (df['close'] + df['low'] + df['high']) * df['vol']
        #     return moneyflow.rolling(window).mean().to_frame('fac') / 0.15
        # return self.data[['close', 'high', 'low', 'vol']].groupby('wind_code').apply(_moneyflow, window)
        temp = (self.data['close'] + self.data['low'] + self.data['high']) * self.data['vol']
        return temp.groupby('wind_code').rolling(window).mean().droplevel(0).sort_index().to_frame('fac') / 0.15

    @func_timer
    @fac_label(addsuffix=True)
    def _TVMA(self, window: int):
        return self.data[['amount']].groupby('wind_code').rolling(window).mean().droplevel(0).sort_index() / 1e6

    @func_timer
    @fac_label(addsuffix=True)
    def _TVSTD(self, window: int):
        return self.data[['amount']].groupby('wind_code').rolling(window).std().droplevel(0).sort_index() / 1e6

    @func_timer
    @fac_label(addsuffix=True)
    def _VEMA(self, window: int):
        return self.data[['vol']].groupby('wind_code').ewm(span=window, ignore_na=True).mean().droplevel(0).sort_index()

    @func_timer
    @fac_label(addsuffix=True)
    def _VSTD(self, window: int):
        return self.data[['vol']].groupby('wind_code').rolling(window).std().droplevel(0).sort_index()

    @func_timer
    @fac_label(addsuffix=True)
    def _GAINVARIANCE(self, window: int):
        # return self.data[['ret']].groupby('wind_code').apply(lambda df: df.where(df > 0).rolling(window, min_periods=1).var()) * 250
        temp = self.data[['ret']].where(self.data[['ret']] > 0)
        return temp.groupby('wind_code').rolling(window, min_periods=1).var().droplevel(0).sort_index() * 250

    @func_timer
    @fac_label(addsuffix=True)
    def _PEHIST(self, window: int):
        # return self._load_data(['pettm'], 's', expand=True).groupby('wind_code').apply(lambda df: df / df.rolling(window).mean())
        temp = self._load_data(['pettm'], 's', expand=True)
        return temp / temp.groupby('wind_code').rolling(window).mean().droplevel(0).sort_index()

    @func_timer
    @lru_cache(maxsize=16)
    def _PE(self):
        return self.data[['close']].div(self._load_data(['dilutedeps'], 'i', expand=True).values, axis=0)

    @func_timer
    @fac_label()
    def _TOBT(self, window: int = 504):
        ret = self.data['ret'].abs()
        tor = self.data['tor']
        ret_m: pd.DataFrame
        try:
            _, ret_m = w.wsd(fields='close', codes=['000300.SH'], beginTime=C.start, endTime=C.end, usedf=True)
        except Exception as e:
            self.logger.error('wind数据读取失败! 查看wind连接是否成功!')
            raise e
        ret_m.index.name = 'tradedate'
        ret_m, _ = ret_m.pct_change().abs().align(ret, join='right', axis=0)
        _retg = ret.groupby('wind_code')
        _ret_mg = ret_m.groupby('wind_code')
        _data = [
            ret, tor,
            _retg.shift(1),
            _retg.shift(2),
            _retg.shift(3),
            _retg.shift(4),
            _retg.shift(5),
            _ret_mg.shift(1),
            _ret_mg.shift(2),
            _ret_mg.shift(3),
            _ret_mg.shift(4),
            _ret_mg.shift(5)
        ]
        data = pd.concat(_data, axis=1)
        data.columns = ['ret', 'tor', 'ret_1', 'ret_2', 'ret_3', 'ret_4', 'ret_5', 'ret_m_1', 'ret_m_2', 'ret_m_3', 'ret_m_4', 'ret_m_5']
        data['const'] = 1.0

        def _ols(df: pd.DataFrame, window):
            try:
                return RWLS(df.iloc[:, :1], df.iloc[:, 1:], window=window, min_nobs=180, missing='drop').fit().params['tor'].to_frame('out')
            except:
                try:
                    return PandasRollingOLS(df.iloc[:, :1], df.iloc[:, 1:], window=window, has_const=True).beta['tor'].to_frame('out')
                except:
                    return pd.DataFrame([np.nan], columns=['out']).reindex(df.index)

        return data.groupby('wind_code', group_keys=False).apply(_ols, window=window)

    @func_timer
    @fac_label()
    def _OPERATINGPROFITPSLATEST(self):
        return self._load_data(['operprofitps'], 'mi', expand=True)

    @func_timer
    @fac_label()
    def _ROECUT(self):
        return self._load_data(['roebyreport'], 'md', expand=True)


if __name__ == '__main__':
    try:
        vp = VPFactor()
        vp.cal_factor()
        vp.close_conn()
        ed = pd.to_datetime(C.end).date()
        st = ed - timedelta(days=15)
        data = vp.factor.loc[idx[st:ed, :], :]
        sql = WriteSQLV2(C.write_c)
        with sql.start():
            sql.write(data, C.write_tab, rec_exists='ignore')
        vp.logger.info('%s因子更新成功!' % C.end)
    except Exception as e:
        vp.logger.error('%s因子更新失败!' % C.end)
        raise e
