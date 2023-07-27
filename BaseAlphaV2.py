#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   BaseAlphaV2.py
@Time    :   2022/09/02 10:05:34
@Author  :   Liu Junli
@Version :   1.0
@Contact :   liujunli.xmu@foxmail.com
'''
import numpy as np
import pandas as pd
import multiprocessing as mp

from BaseAlphaBase import BaseDataSet, BaseAlphaSet, BaseFactorCheck
from basesqlv3 import WriteSQLV2
from config import Config
from logger import get_module_logger, set_log_with_config
from utils import func_timer
import utils

idx = pd.IndexSlice
set_log_with_config()
BM = {
    'sz50': '000016.SH',
    'hs300': '000300.SH',
    'zz1000': '000852.SH',
    'zzlt': '000902.CSI',
    'zz500': '000905.SH',
    'zz800': '000906.SH',
    'zzqz': '000985.CSI'
}

default_config = {
    'date_range': {
        'start': '2017-01-01',
        'end': '2022-08-15'
    },
    'BM': BM,
    'cne5': ['Beta', 'BooktoPrice', 'EarningsYield', 'Growth', 'Leverage', 'Liquidity', 'Momentum', 'NonLinearSize', 'ResidualVolatility', 'Size'],
    'connect': dict(host='10.224.16.81', user='haquant', passwd='haquant', database='jydb'),
    'stock_pools': ['all_A', 'user_defined']
}

C = Config(default_config)


class DataSet(BaseDataSet):
    _provider = WriteSQLV2
    logger = get_module_logger('file.dataset')

    def __init__(self) -> None:
        super().__init__(self._provider, conn=C.connect, read_only=True)

    def _load(self):
        cols = ['closeprice', 'firstindustrycode', 'afloats', 'adjclose']
        self.load_data = self._provider.read('dailydata', cols, C.date_range)
        ind_sql = """
        SELECT DISTINCT firstindustryname as ind_name, firstindustrycode as ind_code FROM dailydata WHERE tradedate='2020-12-31'
        """
        self.inds_name_code = self._provider._read(ind_sql).dropna(how='any').astype({'ind_code': int}).set_index('ind_code')['ind_name'].to_dict()
        self.style_factor = self._provider.read('barra_cne5_factor', C.cne5, C.date_range)
        self.fields = pd.concat([self._load_fields(stock_pool) for stock_pool in C.stock_pools], axis=1)

    @func_timer
    def _load_fields(self, stock_pool: str):
        if stock_pool == 'all_A':
            fields = pd.DataFrame(1, index=self.load_data.index, columns=['all_A'])

        elif stock_pool == 'user_defined':
            fields = self._provider.read('dailydata', ['ifin_selfdefined_stockpool'], C.date_range)

        elif stock_pool in C.BM.keys():
            fields = self._provider.read('index_weight_daily', ['weight'], C.date_range, {
                'index_code': C.BM[stock_pool]
            }).rename(columns={
                'component_code': 'wind_code'
            }).reset_index().drop(columns=['index_code']).set_index(['tradedate', 'wind_code'])

        fields = fields.astype(bool).sort_index()
        fields.columns = [stock_pool]
        return fields

    @func_timer
    def prepare_data(self):
        with self._provider.start():
            self._load()
        self.logger.info('load data done!')
        self.base_data['ind'] = self.load_data['firstindustrycode'].astype('int64', errors='ignore')
        ind_col = [''.join(['ind', str(i)]) for i in self.inds_name_code.keys()]
        self.ind_dummy = pd.get_dummies(self.base_data['ind'], sparse=True)
        self.ind_dummy.columns = ind_col
        self.base_data['cap'] = np.log(self.load_data['afloats'].mul(self.load_data['closeprice'])).to_frame()
        self.base_data['size'] = self.base_data['cap'].groupby('tradedate').apply(lambda x: pd.qcut(x, 3, labels=['small', 'medium', 'big']))
        self.base_data['weight'] = self.base_data['cap']
        self.base_data['close'] = self.load_data['adjclose'][:]
        del self.load_data
        self.logger.info('prepare data done!')


class AlphaSet(BaseAlphaSet):
    logger = get_module_logger('file.alphaset')

    def load_data(self):
        raise NotImplementedError

    @func_timer
    def pre_process(self, dataset: BaseDataSet, mode='mad', multiple=5, method='ffill', ifmp=False, w_method='cap', freq=None):
        if freq:
            self.alpha = utils.resample(self.alpha, freq)

        if ifmp:
            pool = mp.Pool(processes=3)
            runs = []
            runs.append(
                pool.apply_async(utils.pre_process,
                                 args=(self.alpha, dataset.base_data, dataset.style_factor, w_method, 'cne5', mode, multiple, method)))
            runs.append(
                pool.apply_async(utils.pre_process,
                                 args=(self.alpha, dataset.base_data, dataset.ind_dummy.join(dataset.base_data['cap']), w_method, 'indu', mode,
                                       multiple, method)))
            runs.append(
                pool.apply_async(utils.pre_process, args=(self.alpha, dataset.base_data, None, w_method, 'normal', mode, multiple, method, False)))
            self.alpha = pd.concat([self.alpha] + [r.get() for r in runs], axis=1)
            pool.close()
            pool.join()
        else:
            result = []
            result.append(utils.pre_process(self.alpha, dataset.base_data, dataset.style_factor, w_method, 'cne5', mode, multiple, method))
            result.append(
                utils.pre_process(self.alpha, dataset.base_data, dataset.ind_dummy.join(dataset.base_data['cap']), w_method, 'indu', mode, multiple,
                                  method))
            result.append(utils.pre_process(self.alpha, dataset.base_data, None, w_method, 'normal', mode, multiple, method, False))
            self.alpha = pd.concat([self.alpha] + result, axis=1)


class FactorCheck(BaseFactorCheck):
    def __init__(self, dataset: BaseDataSet, alphaset: BaseAlphaSet) -> None:
        super().__init__(dataset, alphaset)
        self.factor_suffix = 'cne5'

    def get_table(self, alpha_name):
        self.factor_check_info['因子名称'] = alpha_name
        # self.factor_check_info['回测因子']
        # self.factor_check_info['对照因子']
        # self.factor_check_info['回测耗时']
        self.table.update({'info': pd.DataFrame(self.factor_check_info, index=['值']).T})
        cols = self.factor_data.columns[self.factor_data.columns.str.contains('_' + alpha_name + '_')]
        self.table.update({'stats': utils.factor_return_stats(self.factor_data[cols], self.factor_return['allA'][cols])})
        self.table.update({'IC': utils.print_IC_table(self.IC['allA'][cols]).iloc[-15:, :]})
        cols_ = self.group_return['userdefined'].columns[self.group_return['userdefined'].columns.str.contains('_' + alpha_name + '_')]
        self.table.update({'groups': utils.group_return_stats(self.group_return['userdefined'][cols_])})

    def plot(self, alpha_name: str):
        alpha_cols = self.factor_data.columns[self.factor_data.columns.str.contains('_' + alpha_name + '_')]
        col_suffix = 'alpha_' + alpha_name + '_' + self.factor_suffix
        tit = 'fac_' + self.factor_suffix

        self.image.clear()

        data = self.IC['allA'][alpha_cols]
        self.image.update(dict(IC=utils.plot_line(data, ylabel='IC累计值', method='sum', suptitle='全市场股票池IC累计值', alpha_name=alpha_name)))

        data = self.long_short['long_short_fields'].loc[:, idx[:, col_suffix]]
        self.image.update(
            dict(long_short1=utils.plot_line(data, suptitle='不同选股域的因子%s多空收益LSR' % tit, leg_txt=['自定义选股域', '全A选股域'], alpha_name=alpha_name)))

        data = self.long_short['long_short_fields_ind'].loc[:, idx[:, col_suffix]]
        self.image.update(
            dict(long_short2=utils.plot_line(data, suptitle='不同选股域的因子%s行业分层多空收益LSR' % tit, leg_txt=['自定义选股域', '全A选股域'], alpha_name=alpha_name)))

        data = self.group_return['userdefined'].loc[:, col_suffix].squeeze().unstack(level='fac_qt')
        self.image.update(dict(groups1=utils.plot_line(data, test='mttest', suptitle='%s分组测试' % tit, alpha_name=alpha_name)))

        data = self.group_return['userdefined_i_ind'].loc[:, col_suffix].squeeze().unstack(level='fac_qt')
        self.image.update(dict(groups2=utils.plot_line(data, test='mttest', suptitle='%s行业分层分组测试' % tit, alpha_name=alpha_name)))

        data = self.long_short['long_short_userdefined_o_cap'].xs(col_suffix, level=0, axis=1)
        self.image.update(dict(groups_cap=utils.plot_line(data, suptitle='不同自由流通市值的因子多空收益LSR', alpha_name=alpha_name)))

        data = self.factor_return['allA'].xs('beta', level=1).loc[:, alpha_cols]
        self.image.update(dict(reg=utils.plot_line(data, test='fmtest', suptitle='回归系数累计值', alpha_name=alpha_name)))

    @utils.func_timer
    def factor_group_analysis(self):
        freq = None
        alpha_flag = self.factor_data.columns.str.startswith('alpha')
        suffix_flag = self.factor_data.columns.str.endswith('_' + self.factor_suffix)  # cne5因子
        alpha_cols = self.factor_data.columns[(~alpha_flag) | (suffix_flag)]

        fields = self.dataset.fields.reindex(self.factor_data.index)
        index = fields.index[fields['user_defined'].fillna(False)]
        factor_data = self.factor_data.reindex(index)
        if suffix_flag.sum() > 1:
            self.group_return.update(dict(allA=utils.group_return(self.factor_data[alpha_cols], freq=freq)))
            self.group_return.update(dict(allA_i_ind=utils.group_return(self.factor_data[alpha_cols], group_adjust=['ind'], freq=freq)))
            self.group_return.update(dict(userdefined=utils.group_return(factor_data[alpha_cols], freq=freq)))
            self.group_return.update(dict(userdefined_i_ind=utils.group_return(factor_data[alpha_cols], group_adjust=['ind'], freq=freq)))
            self.group_return.update(dict(userdefined_o_cap=utils.group_return(factor_data[alpha_cols], out_group=['size'], freq=freq)))
        else:
            pool = mp.Pool(processes=5)
            results = []
            results.append(pool.apply_async(utils.group_return, args=(self.factor_data[alpha_cols], ['ind'], None, freq)))
            results.append(pool.apply_async(utils.group_return, args=(factor_data[alpha_cols], ['ind'], None, freq)))
            results.append(pool.apply_async(utils.group_return, args=(factor_data[alpha_cols], None, ['size'], freq)))
            results.append(pool.apply_async(utils.group_return, args=(self.factor_data[alpha_cols], None, None, freq)))
            results.append(pool.apply_async(utils.group_return, args=(factor_data[alpha_cols], None, None, freq)))
            dict_keys = ['allA_i_ind', 'userdefined_i_ind', 'userdefined_o_cap', 'allA', 'userdefined']
            for i, p in enumerate(results):
                self.group_return.update({dict_keys[i]: p.get()})
            pool.close()
            pool.join()

    @utils.func_timer
    def long_short_analysis(self):
        def long_short_return(field: str, grouer='tradedate') -> pd.DataFrame:
            def _long_short_return(df: pd.DataFrame, field=field):
                out = df.xs('p5', level=1, axis=0) - df.xs('p1', level=1, axis=0)
                out.columns = pd.MultiIndex.from_product([[field], out.columns], names=['field', 'factor'])
                return out

            return self.group_return[field].groupby(grouer).apply(_long_short_return).droplevel(level=0, axis=0)

        self.long_short.update(dict(long_short_fields=pd.concat([long_short_return(field) for field in ['userdefined', 'allA']], axis=1)))
        self.long_short.update(
            dict(long_short_fields_ind=pd.concat([long_short_return(field) for field in ['userdefined_i_ind', 'allA_i_ind']], axis=1)))

        self.long_short.update(
            dict(long_short_userdefined_o_cap=long_short_return('userdefined_o_cap', grouer=['tradedate', 'size']).droplevel(
                level=0, axis=0).droplevel(level=0, axis=1).unstack(level='size').sort_index()))

    @utils.func_timer
    def IC_analysis(self):
        self.IC.update(allA=utils.factor_information_coefficient(self.factor_data))

    @utils.func_timer
    def regression_analysis(self):
        self.factor_return.update(allA=utils.factor_return(self.factor_data))
