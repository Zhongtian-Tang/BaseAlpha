#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   factor_test.py
@Time    :   2022/07/24 22:11:02
@Author  :   Liu Junli
@Version :   1.0
@Contact :   liujunli.xmu@foxmail.com
'''

from BaseAlpha import *

settings = {
    "start_date": 20170101,  # 表示因子的开始结束时间，closeprice数据时间应该在end_date后再加一天
    "end_date": 20220831,
    "stock_pools": ['all_A', 'user_defined'],  # 'all_A', 'sz50', 'hs300', 'zz1000', 'zz800', 'zz500', 'zzqz'(中证全指), 'zzlt'(中证流通), 'user_defined';
    "style_factor":
    ['Beta', 'BooktoPrice', 'EarningsYield', 'Growth', 'Leverage', 'Liquidity', 'Momentum', 'NonLinearSize', 'ResidualVolatility', 'Size'],
    "alpha_factor_list": ['ACCA'],  #['fearng'], #['rec', 'sfy12p'], #['Beta'], 
    "data_sources": 'mysql',  # 'mysql'
    "alpha_factor_source":
    'EQU_FACTOR_CF',  #'equ_factor_af.h5',  #'EQU_FACTOR_AF',  #'tl_factor\equ_factor_af.h5',  #'barra_cne5_factor',  #'EQU_FACTOR_CF'
    "connect": dict(host='10.224.16.81', user='haquant', passwd='haquant', database='jydb', port=3306, charset='utf8'),
    "connect_or": dict(part1='rededata', part2='HAzc_1805@10.224.0.21:1521/datayesdb'),
    # 'rededata/HAzc_1805@10.224.0.21:1521/datayesdb',
    "file_path": r'D:\Data',  #r'\\10.224.1.70\public\data\dailydata_copy',
}


class AlphaSet(BaseAlphaSet):
    def __init__(self, settings, ifload) -> None:
        super().__init__(settings, ifload)

    @utils.func_timer
    def pre_process(self, dataset: BaseDataSet, mode='mad', multiple=5, method='ffill', ifmp=False, w_method='cap', freq=None):
        self.alpha.columns = str('alpha_') + self.alpha.columns + str('_raw')
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
        self.factor_suffix = 'indu'

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

        # factor_suffix = 'indu'  # 'cne5', 'indu', 'normal'

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
        # factor_suffix = 'indu'

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


if __name__ == '__main__':

    # !index.names=['tradedate','wind_code']
    # !'tradedate'应该是datetime.date格式, pd.to_datetime(yourcols).dt.date转换即可

    # example:
    # *factor.index.names=['tradedate,'wind_code']
    # *factor.reset_index(inplace=True)
    # *factor['%date'] = pd.to_datetime(factor['%date']).dt.date
    # *factor.set_index(['tradedate','wind_code'])

    # settings['start_date'] = 20170101
    # settings['end_date'] = 20220630
    # dataset = Base.load('dataset.pkl')  # 加载数据集
    dataset = BaseDataSet(settings)  # 初始化数据集
    # dataset.save('dataset.pkl')  # 保存数据集
    # alphaset = Base.load('alphaset.pkl')  # 加载因子集
    # alphaset = AlphaSet(settings)  # 初始化alpha集
    alphaset = AlphaSet(settings, ifload=False)  # 初始化alpha集
    # alphaset.alpha = ...
    # alphaset.alpha: pd.DataFrame = pd.read_pickle(r"D:\Data\similarity_0818.pkl", compression='bz2')  # 加载因子数据
    alphaset.alpha = factor[:]
    # alphaset.alpha.reset_index(inplace=True)
    # alphaset.alpha['tradedate'] = pd.to_datetime(alphaset.alpha['tradedate']).dt.date
    # alphaset.alpha.set_index(['tradedate', 'wind_code'], inplace=True)
    alphaset.pre_process(dataset, ifmp=True, w_method='equal')  # 预处理, 耗时：cap: 127.33s, 145.60s; equal: 59.63s, 76.52s
    fc = FactorCheck(dataset, alphaset)  # 初始化因子检验
    fc.factor_suffix = 'indu'
    fc.run()  # 运行因子检验, 耗时：105.43s

    PATH = r'D:\factor'
    facs = os.listdir(PATH)
    facs.pop(6)
    for fac in facs:
        alphaset.alpha = pd.read_pickle(os.path.join(PATH, fac), compression='bz2').to_frame().sort_index()  # 加载因子数据,sort_index不是很有必要
        # alphaset.settings['alpha_factor_list'] = []
        # alphaset.start()
        # alphaset.load_alpha()
        # alphaset.save('alphaset.pkl')
        alphaset.pre_process(dataset, ifmp=True, w_method='equal', freq='M')  # 预处理, 耗时：cap: 127.33s, 145.60s; equal: 59.63s, 76.52s
        fc = FactorCheck(dataset, alphaset)  # 初始化因子检验
        fc.run()  # 运行因子检验, 耗时：105.43s
