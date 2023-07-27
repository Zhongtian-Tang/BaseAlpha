#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cal_factor_tl.py
@Time    :   2022/08/03 10:20:42
@Author  :   Liu Junli
@Version :   1.0
@Contact :   liujunli.xmu@foxmail.com
'''

import os
import re
import inspect
import numpy as np
import pandas as pd
from typing import List, Callable
from functools import wraps, lru_cache
from pprint import pprint
from pyfinance.ols import PandasRollingOLS
from statsmodels.regression.rolling import RollingWLS as RWLS

from BaseAlpha import Base, settings, idx
from utils import func_timer
from basesql import WriteSQLV2

pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

settings['start_date'] = '20050101'
settings['end_date'] = '20050201'
# settings['end_date'] = '20220815'
settings['data_sources'] = 'mysql'
SAVE_PATH = r'Z:\factor\liujl\factor'


def fac_label(addsuffix: bool = False):
    def decorate(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if inspect.signature(func).parameters['source'].default is not inspect._empty:
                args[0].fac_sources.update({func.__name__.replace('_', ''): inspect.signature(func).parameters['source'].default})
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


class CalFactor(Base):
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

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.data = pd.DataFrame()
        self.factor_ori = pd.DataFrame()
        self.fac_sources = {}
        self.data = self.__load_daily(self.data_cols, 'd')
        self.__preprocess()

    def __load_daily(self, fields: List[str], tab: str) -> pd.DataFrame:
        exp = """
        SELECT tradedate, wind_code, %s FROM %s WHERE tradedate BETWEEN %s AND %s
        """ % (', '.join(fields), self.table[tab], str(self.settings['start_date']), str(self.settings['end_date']))
        return self.read_sql(exp)

    def __load_lowfreq(self, fields: List[str], tab: str) -> pd.DataFrame:
        exp = """
        SELECT wind_code, enddate, %s FROM %s WHERE enddate BETWEEN %s AND %s
        """ % (','.join(fields), self.table[tab], str(self.settings['start_date']), str(self.settings['end_date']))
        data = self.read_sql(exp).rename(columns={'enddate': 'tradedate'}).set_index(['tradedate', 'wind_code'])
        index = self.data.index.append(data.index).drop_duplicates()
        return data.reindex(index).groupby('wind_code').fillna(method='ffill').reindex(self.data.index)

    def __load_factor(self, fields: List[str], tab: str) -> pd.DataFrame:

        exp = """
        SELECT trade_date as tradedate, security_id as wind_code, %s FROM %s WHERE trade_date BETWEEN %s AND %s
        """ % (', '.join(fields), tab, self.settings['start_date'], self.settings['end_date'])
        return self.read_sql(exp, con=self.cr_or)

    def __preprocess(self) -> None:
        # *字段名的简化: 如, openprice -> open
        # *一些基础的计算: 复权后的价格
        self.data.rename(columns=self.cols_abbr, inplace=True)
        self.data['close_adj'] = self.data['close'] * self.data['adj']

    @func_timer
    def cal_factor(self):
        out = [
            # self._ILLIQUIDITY(window=20),
            # self._SKEWNESS(window=20),
            # self._VOLATILITY(window=20),
            # self._WVAD(window=24),
            # self._MONEYFLOW(window=20),
            # self._TVMA(window=20),
            # self._TVMA(window=6),
            # self._TVSTD(window=20),
            # self._TVSTD(window=6),
            self._VEMA(window=5),
            self._VEMA(window=10),
            self._VEMA(window=12),
            self._VEMA(window=26),
            # self._VSTD(window=10),
            # self._VSTD(window=20),
            # self._GAINVARIANCE(window=120),
            # self._PEHIST(window=120),
            # self._PEHIST(window=60),
            # self._TOBT(),
            # self._OPERATINGPROFITPSLATEST(),
            # self._ROECUT()
        ]
        self.factor = pd.concat(out, axis=1).sort_index().replace([np.inf, -np.inf], None)
        # self.factor = factor[~factor.isin([np.nan, np.inf, -np.inf]).any(1)]
        # self.save('factor.pkl', self.factor)
        # [self.save(os.path.join(SAVE_PATH, '%s.pkl' % col), self.factor[col]) for col in self.factor.columns]
        # [self.factor[col].to_csv(os.path.join(SAVE_PATH, '%s.csv' % col)) for col in self.factor.columns]
        # [self.factor[[col]].to_pickle(os.path.join(SAVE_PATH, '%s.pkl' % col)) for col in self.factor.columns]
        # self.save(os.path.join(SAVE_PATH, 'allfactors.pkl'), self.factor)
        # self.factor.to_csv(os.path.join(SAVE_PATH, 'allfactors.csv'))
        # self.factor.to_pickle(os.path.join(SAVE_PATH, 'allfactors.pkl'), compression='bz2')

    @func_timer
    def get_factor_ori(self, aligned: bool = True):
        # *按表 load
        # field_source = {}
        # [field_source.setdefault(self.fac_sources[re.sub(r'[0-9]+', '', fac)], []).append(fac) for fac in self.factor.columns]
        # self.factor_ori = pd.concat([self.__load_factor(fields, source) for source, fields in field_source.items()], axis=1)
        # *单因子load
        fac_ls = [
            self.__load_factor([fac], self.fac_sources[re.sub(r'[0-9]+', '', fac)]).reindex(self.factor.index)
            if aligned else self.__load_factor([fac], self.fac_sources[re.sub(r'[0-9]+', '', fac)]) for fac in self.factor.columns
        ]
        self.factor_ori = pd.concat(fac_ls, axis=1).sort_index()

    @func_timer
    def check(self, ifret: bool = True):
        # 一些其他的比较操作
        _corr = self._check_corr()
        print('相关系数检测结果:\n', '*' * 50)
        pprint(_corr)
        _abs = self._cross_section_abs()
        print('误差绝对值占比:\n', '*' * 50)
        pprint(_abs)
        if ifret:
            return _corr, _abs

    def _check_corr(self, detail: bool = True):
        # 一些其他的比较操作
        def corr(x: pd.DataFrame, y: pd.DataFrame, dim: int = 0) -> pd.DataFrame:
            idx_ = sorted(list(set(x.index.levels[dim]) & set(y.index.levels[dim])))
            out = pd.concat([x.xs(i, level=dim).corrwith(y.xs(i, level=dim)) for i in idx_], axis=1).T
            out.index = idx_
            return out

        if not detail:
            return self.factor.corrwith(self.factor_ori)
        # 相关系数
        a = corr(self.factor, self.factor_ori, dim=0)
        b = corr(self.factor, self.factor_ori, dim=1)
        out = pd.DataFrame()
        out['截面平均'] = a.mean()
        out['时序平均'] = b.mean()
        out['截面>0.99占比'] = (a > 0.99).sum() / a.shape[0]
        out['时序>0.99占比'] = (b > 0.99).sum() / b.shape[0]
        out['截面>0.90占比'] = (a > 0.90).sum() / a.shape[0]
        out['时序>0.90占比'] = (b > 0.90).sum() / b.shape[0]
        return out

    def _cross_section_abs(self):
        _stats = (self.factor - self.factor_ori).abs() / self.factor_ori
        out = pd.DataFrame()
        out['mean'] = _stats.replace(np.inf, np.nan).mean()
        out['平均值<0.01占比'] = (_stats < 0.01).sum() / _stats.shape[0]
        out['平均值<0.001占比'] = (_stats < 0.001).sum() / _stats.shape[0]
        out['平均值<0.001的天数占比'] = (_stats.mean(level=0) < 0.001).sum() / _stats.index.levels[0].shape
        out['平均值<0.001的股票占比'] = (_stats.mean(level=1) < 0.001).sum() / _stats.index.levels[1].shape
        return out

    @func_timer
    @fac_label()
    def _ILLIQUIDITY(self, window: int, cols: List[str] = ['ret', 'amount'], source: str = 'EQU_FACTOR_OBOS'):
        def _illiquidity(df: pd.DataFrame, window: int):
            df = df.abs().rolling(window).sum()
            return (df['ret'] / df['amount']).to_frame('fac').replace(np.inf, np.nan)

        return self.data[cols].groupby('wind_code').apply(_illiquidity, window=window) * 1e7

    @func_timer
    @fac_label()
    def _SKEWNESS(self, window: int = 20, cols: List[str] = ['close'], source: str = 'EQU_FACTOR_OBOS'):
        return self.data[cols].groupby('wind_code').apply(lambda df: df.rolling(window).skew())

    @func_timer
    @fac_label()
    def _VOLATILITY(self, window: int, cols: List[str] = ['tor'], source: str = 'EQU_FACTOR_OBOS'):
        return self.data[cols].groupby('wind_code').apply(lambda df: df.rolling(window).std() / df.rolling(window).mean()).replace(np.inf, np.nan)

    @func_timer
    @fac_label()
    def _WVAD(self, window: int = 24, cols: List[str] = ['open', 'close', 'high', 'low', 'vol'], source: str = 'EQU_FACTOR_OBOS'):
        def _wvad(df: pd.DataFrame, window: int):
            wvad = (df['close'] - df['open']) / (df['high'] - df['low']) * df['vol']
            return wvad.rolling(window).sum().to_frame('fac')

        return self.data[cols].groupby('wind_code').apply(_wvad, window) / 1e6

    @func_timer
    @fac_label(addsuffix=True)
    def _MONEYFLOW(self, window: int = 20, cols: List[str] = ['close', 'high', 'low', 'vol'], source: str = 'EQU_FACTOR_VOLUME'):
        def _moneyflow(df: pd.DataFrame, window: int):
            moneyflow = (df['close'] + df['low'] + df['high']) * df['vol']
            return moneyflow.rolling(window).mean().to_frame('fac') / 0.15

        return self.data[cols].groupby('wind_code').apply(_moneyflow, window)

    @func_timer
    @fac_label(addsuffix=True)
    def _TVMA(self, window: int, cols: List[str] = ['amount'], source: str = 'EQU_FACTOR_VOLUME'):
        return self.data[cols].groupby('wind_code').apply(lambda df: df.rolling(window).mean()) / 1e6

    @func_timer
    @fac_label(addsuffix=True)
    def _TVSTD(self, window: int, cols: List[str] = ['amount'], source: str = 'EQU_FACTOR_VOLUME'):
        return self.data[cols].groupby('wind_code').apply(lambda df: df.rolling(window).std()) / 1e6

    @func_timer
    @fac_label(addsuffix=True)
    def _VEMA(self, window: int, cols: List[str] = ['vol'], source: str = 'EQU_FACTOR_VOLUME'):
        # return self.data[cols].groupby('wind_code').apply(lambda df: df.ewm(halflife=window / 2, ignore_na=True).mean())
        return self.data[cols].groupby('wind_code').apply(lambda df: df.ewm(span=window, ignore_na=True).mean())

    @func_timer
    @fac_label(addsuffix=True)
    def _VSTD(self, window: int, cols: List[str] = ['vol'], source: str = 'EQU_FACTOR_VOLUME'):
        return self.data[cols].groupby('wind_code').apply(lambda df: df.rolling(window).std())

    @func_timer
    @fac_label(addsuffix=True)
    def _GAINVARIANCE(self, window: int, cols: List[str] = ['ret'], source: str = 'EQU_FACTOR_RETURN'):
        # def _gainvariance(df: pd.DataFrame, window: int):
        #     return df.where(df > 0).pow(2).rolling(window, min_periods=1).mean() - df.rolling(window, min_periods=1).mean()

        # return self.data[cols].groupby('wind_code').apply(_gainvariance, window=window) * 250
        return self.data[cols].groupby('wind_code').apply(lambda df: df.where(df > 0).rolling(window, min_periods=1).var()) * 250

    @func_timer
    @fac_label(addsuffix=True)
    def _PEHIST(self, window: int, source: str = 'EQU_FACTOR_VS'):
        # return self._PE().groupby('wind_code').apply(lambda df: df / df.rolling(window).mean())
        return self.__load_daily(['pettm'], 's').groupby('wind_code').apply(lambda df: df / df.rolling(window).mean())

    @func_timer
    @lru_cache(maxsize=16)
    def _PE(self):
        # return self.data[['close']].mul(self.data['af'], axis=0).div(self.__load_lowfreq(['netprofit'], 'i').values, axis=0)
        return self.data[['close']].div(self.__load_lowfreq(['dilutedeps'], 'i').values, axis=0)

    @func_timer
    @fac_label()
    @lru_cache(maxsize=16)
    def _TOBT(self, window: int = 504, cols: List[str] = ['ret'], source: str = 'EQU_FACTOR_POWER'):
        ret = self.data['ret'].abs()
        tor = self.data['tor']
        cond = "wind_code=='000300.SH'&tradedate>='%s'&tradedate<='%s'" % (str(self.settings['start_date']), str(self.settings['end_date']))
        ret_m: pd.DataFrame
        ret_m, _ = self.read_h5('index_daily.h5', key='index_quote', cond=cond)['changepct'].abs().droplevel(1).align(ret, join='right', axis=0)
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
    def _OPERATINGPROFITPSLATEST(self, cols: List[str] = ['af'], source: str = 'EQU_FACTOR_PSI'):
        # return self.__load_lowfreq(['operatingprofit'], 'mi').div(self.data[cols].values, axis=0)
        return self.__load_lowfreq(['operprofitps'], 'mi')

    @func_timer
    @fac_label()
    def _ROECUT(self, source: str = 'EQU_FACTOR_PQ'):
        return self.__load_lowfreq(['roebyreport'], 'md')


calf = CalFactor(settings)
calf.cal_factor()
# calf.factor
# calf.get_factor_ori(aligned=True)
# corr_, abs_ = calf.check(ifret=True)

# CONN = dict(host='10.224.1.70', user='liujl', passwd='CEQZqwer', port=3306, charset='utf8', auth_plugin='mysql_native_password')
# DATABASE = 'liujl'
# table_name = 'tl_factor'
# sql = WriteSQLV2()
# sql.CONN.update(CONN)
# sql.start_conn(DATABASE)
# sql._check_tab_status(table_name)
# sql.set_active_table(table_name)
# sql.write(data=calf.factor, rec_exists='fill')
# sql.close_conn()

# df = sql.read(table_name, date_range=dict(end='2007-12-31'))
# df.first_valid_index
# df = df.swaplevel().astype(object)
# df = df.where(pd.notnull(df), None)
# out = {}
# for col in df.columns:
#     out.update({col: df[col].first_valid_index()})
# pprint(out)
# df['OPERATINGPROFITPSLATEST'].first_valid_index()