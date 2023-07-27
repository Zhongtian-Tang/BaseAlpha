#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   windapi.py
@Time    :   2022/08/22 20:18:51
@Author  :   Liu Junli
@Version :   1.0
@Contact :   liujunli.xmu@foxmail.com
'''

import pandas as pd
from typing import Optional, Union
from WindPy import w
from datetime import date as dt
import datetime
from logger import get_module_logger, set_log_with_config

# 1.2.4
set_log_with_config()

w.wss  # 限定时间维度
w.wsd  # 限定品种或指标维度
# 最指标最循环会比较快
w.wsq  # 速度很快？

w.isconnected()  # 判断WindPy是否已经登录成功
# 默认命令超时时间为120秒，如需设置超时时间可以加入waitTime参数，例如waitTime=60,即设置命令超时时间为60秒

w.isconnected()  # 判断WindPy是否已经登录成功
w.start()
sw_index = w.wset("sectorconstituent", "date=2018-06-12;sectorid=a39901011g000000", usedf=True)
sw_index[1].head(5)

# beginTime = date(2019, 1, 1)
# endTime = date.today()
# offset = -10
# options = ''
# datarange = w.tdays(beginTime, endTime, options)
# date = w.tdaysoffset(offset, beginTime, options)  #上一个交易日
# datecount = w.tdayscount(beginTime, endTime, options)
from pathlib import Path

data = pd.HDFStore(Path('D:\Data') / 'index_daily.h5')

pd.read_pickle(r"Z:\factor\liujl\factor\SKEWNESS.pkl")
data.close()
data.keys()
data.get('info/tradedates')
dff = pd.read_hdf(Path('D:\Data') / 'index_daily.h5', key='index_quote')


class Wind:
    def __init__(self):
        self.state: bool
        self.logger = get_module_logger('windapi')

    def start(self):
        if not self.state:
            e = w.start(waitTime=60)
            if e.ErrorCode == 0:
                self.logger.info('WindPy start successfully!')
            else:
                self.logger.error('WindPy start failed!')

    @property
    def state(self):
        return w.isconnected()

    def close(self):
        w.close()

    def _get_date_range(start: Optional[Union[str, dt]] = None, end: Optional[Union[str, dt]] = None):
        if end is None:
            end = dt.today()
        if start is None:
            start = end
        return start, end

    def get_data(self,
                 codes: Optional[Union[str, list]] = None,
                 fields: Optional[Union[str, list]] = None,
                 start: Optional[Union[str, dt]] = None,
                 end: Optional[Union[str, dt]] = None):
        # end None: end=today
        # start None: start=end
        # codes None: codes=all
        # fields None: fields=['open','close','high','low','volume']

        start, end = self._get_date_range(start, end)
        if codes is None:
            codes = self.get_code(date=end)['wind_code'].to_list()
        if fields is None:
            fields = ['open', 'close', 'high', 'low', 'volume']
        out = []
        f = 'open'
        for f in fields:
            out.append(w.wsd(codes, fields, start, end, usedf=True))
        out = pd.concat(out, axis=1)
        # return w.wsd(code, fields, start_date, end_date, usedf=True)

    def get_code(self, tablename=None, sectorid: Optional[str] = None, date: Optional[Union[str, dt]] = None):
        if tablename is None:
            tablename = "sectorconstituent"
        if sectorid is None:
            sectorid = 'a001010100000000'
        if date is None:
            date = dt.today()
        _, out = w.wset(tablename, date=date, sectorid=sectorid, usedf=True)
        return out

    def date_range(self, start: Optional[Union[str, dt]] = None, end: Optional[Union[str, dt]] = None, offset=None):
        if end is None:
            end = dt.today()
        if start is None:
            start = end
        return w.tdays(start, end, offset)


w.start()
df: pd.DataFrame
_, df = w.wsd(fields='close', codes=['000300.SH'], beginTime='2017-01-01', usedf=True)
df = df.pct_change()
wind = Wind()