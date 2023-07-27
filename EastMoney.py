#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   EastMoney.py
@Time    :   2022/08/24 09:33:52
@Author  :   Liu Junli
@Version :   1.0
@Contact :   liujunli.xmu@foxmail.com
'''

import json
import requests
import time
from pathlib import Path
from datetime import date, datetime
from functools import lru_cache
from typing import Optional, Union, List, Tuple, Literal, Any, Dict

import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from fake_useragent import UserAgent
from tqdm import trange
from contextlib import contextmanager

from utils import func_timer
from logger import get_module_logger, set_log_with_config
from constants import BASEURI, HEADERS, SORT, REPORTNAME, CALLBACK, WEB, ALL, Yesterday, P, FORMAT

set_log_with_config()

# 一级：dict_keys(['version', 'result', 'success', 'message', 'code'])
# 二级：dict_keys(['pages', 'data', 'count'])

# NOTE: filter: date_range必须是单引号, 其他条件双引号
# NOTE: 有三个keys时, 单独的 PARTICIPANT_CODE, HOLD_DATE 无法翻页


def isdate(value: str):
    try:
        _dt = datetime.strptime(value, FORMAT)
        return True if isinstance(_dt, date) else False
    except:
        return False


def check_type(value: Any):
    return date if type(value) is str and isdate(value) else type(value)


def todate(value: str):
    assert isdate(value)
    return datetime.strptime(value, FORMAT).date()


class Params:
    _date_range = None
    _pageSize: int = 50
    _sortColumns = None
    _sortTypes = None
    logger = get_module_logger('Params')

    def __init__(self,
                 callback: str,
                 reportName: str,
                 columns: str = ALL,
                 sortColumns: Optional[Union[str, List[str], Tuple[str]]] = None,
                 sortTypes: Optional[Union[SORT, List[SORT], Tuple[SORT]]] = None,
                 pageNumber: int = 1,
                 pageSize: int = 50,
                 source: Optional[str] = None,
                 client: Optional[str] = None,
                 date_range: Optional[Union[str, list, tuple, dict]] = None,
                 other_fields: Optional[Dict[str, Union[str, list, tuple]]] = None,
                 str_filter: Optional[str] = None):
        self.callback = callback  # 这两个参数是决定一个Params的关键key
        self.reportName = reportName  # 这两个参数是决定一个Params的关键key
        self.columns = columns
        self.sortColumns = sortColumns
        self.sortTypes = sortTypes
        self.pageNumber = pageNumber
        self.pageSize = pageSize
        self.source = source
        self.client = client
        self.date_range = date_range
        self.other_fields = other_fields
        self.str_filter = str_filter

        self.skip = False  #避免无限迭代

    def __str__(self) -> str:
        return json.dumps(self.to_dict())

    def __repr__(self) -> str:
        return json.dumps(self.to_dict())

    @property
    @lru_cache
    def date_field(self):
        with self.set_skip():
            temp = [k for k, dtype in self.list_fields(detail=True).items() if dtype is date]
            return temp if temp else None

    @property
    @lru_cache
    def fields(self):
        with self.set_skip():
            return self.list_fields(detail=False)

    def to_dict(self):
        return {f: getattr(self, f) for f in dir(self) if (f in P) and (getattr(self, f) is not None)}

    def new(self) -> 'Params':
        return Params(**{k: v for k, v in self.to_dict().items() if k in ('reportName', 'callback')})

    @func_timer
    def _get_uri(self, params: Optional[Dict[str, Union[str, int, Tuple[int]]]] = None):
        params = self.to_dict() if params is None else params
        _data = requests.get(BASEURI, params=params)
        try:
            _js = json.loads(_data.text[len(params['callback']) + 1:-2])
        except:
            if 'false' in _data.text:
                raise Exception(_data.text)
        assert _js['success'], _js['message']
        return _js

    def list_fields(self, detail: bool = False):
        _js = self._get_uri(self.new().to_dict())

        fields = list(_js['result']['data'][0].keys())

        if not detail:
            return fields

        _df = pd.DataFrame(_js['result']['data'])
        return {field: check_type(_df[field][_df[field].first_valid_index() or 0]) for field in fields}

    @contextmanager
    def set_skip(self):
        self.skip = True
        yield
        self.skip = False

    @property
    @lru_cache
    def date_maxmin(self):
        with self.set_skip():
            if not self.date_field:
                return None
            _params = self.new()
            _params.sortColumns = _params.columns = self.date_field
            _params.sortTypes = -1
            _js = self._get_uri(_params.to_dict())
            ed_date = list(_js['result']['data'][0].values())[0]
            _params.pageNumber = _js['result']['pages']
            _js = self._get_uri(_params.to_dict())
            st_date = list(_js['result']['data'][0].values())[-1]
            return todate(st_date), todate(ed_date)

    @property
    def date_range(self):
        return self._date_range

    @date_range.setter
    def date_range(self, value: Optional[Union[str, int, date, Tuple[str, int, date], List[Union[str, int, date]], Dict[str, Union[str, int,
                                                                                                                                   date]]]]):
        self._date_range = self._get_date(value)

    def _get_date(self, date_: Optional[Union[str, int, date, Tuple[str, int, date], List[Union[str, int, date]], Dict[str, Union[str, int, date]]]]):

        if date_ is None:
            return None

        if (isinstance(date_, str) and len(date_) == 8):
            return str(pd.to_datetime(date_).date())

        if isinstance(date_, str):
            return date_

        if isinstance(date_, int):
            return self._get_date(str(date_))

        if isinstance(date_, date):
            return str(date_)

        if isinstance(date_, tuple):
            return tuple([self._get_date(idate) for idate in date_])

        if isinstance(date_, list):
            return [self._get_date(idate) for idate in date_]

        if isinstance(date_, dict):
            st = date_.get('start', None) if date_.get('start', None) is not None else self.date_maxmin[0]
            ed = date_.get('end', None) if date_.get('end', None) is not None else self.date_maxmin[-1]
            return self._get_date((st, ed))

        raise Exception('日期格式错误!')

    @property
    def pageSize(self):
        return self._pageSize

    @pageSize.setter
    def pageSize(self, value: int):
        self._pageSize = min(500, value)

    @property
    def filter(self):
        if self.skip:
            return self.str_filter

        assert set(self.other_fields.keys()) <= set(self.fields) if self.other_fields else True, '部分字段不正确'
        cond = [
            self._getfilter(self.date_field[0], self.date_range) if self.date_range and self.date_field else '',
            ''.join([self._getfilter(f, c) for f, c in self.other_fields.items()]).replace("'", "\"") if self.other_fields else '',
            self.str_filter if self.str_filter else ''
        ]
        cond = [c for c in cond if c != '']
        return ''.join(cond) if cond else None

    def _getfilter(self, field: str, cond: Union[str, list, tuple]):
        if isinstance(cond, str):
            return "(%s='%s')" % (field, cond)

        if isinstance(cond, list):
            return "(%s in ('%s'))" % (field, "','".join(cond))

        if isinstance(cond, tuple):
            assert len(cond) == 2, '%s 范围参数错误' % (field)
            return "(%s>='%s')(%s<='%s')" % (field, cond[0], field, cond[1])

    @property
    def sortColumns(self):
        return self._sortColumns

    @sortColumns.setter
    def sortColumns(self, value: Optional[Union[str, List[str], Tuple[str]]]):
        if value is None or isinstance(value, str):
            self._sortColumns = value

        if isinstance(value, (tuple, list)):
            self._sortColumns = ','.join(list(value))

    @property
    def sortTypes(self):
        return self._sortTypes

    @sortTypes.setter
    def sortTypes(self, value: Optional[Union[SORT, List[SORT], Tuple[SORT]]]):
        if value is None:
            self._sortTypes = value
            return

        if self.sortColumns is None:
            self._sortTypes = value if isinstance(value, int) else tuple(value)
        else:
            value = [value] if isinstance(value, int) else value
            assert (len(self.sortColumns.split(',')) == len(value)), '长度不一致!'
            self._sortTypes = tuple(value)

    @property
    def pages(self):
        return self._get_uri()['result']['pages']

    @property
    def count(self):
        return self._get_uri()['result']['count']


class StockDetailParams(Params):
    _callback = CALLBACK['stock_detail']
    _reportName = REPORTNAME['stock_detail']
    _other_fields = {'MARKET_CODE': ['001', '003']}
    _source = WEB
    _client = WEB
    _columns = ALL
    _sortColumns = ('HOLD_DATE', 'PARTICIPANT_CODE')
    _sortTypes = (-1, 1)
    _date_range = Yesterday

    def __init__(self,
                 pageNumber: int = 1,
                 pageSize: int = 50,
                 date_range: Optional[Union[str, list, Tuple, dict]] = None,
                 str_filter: Optional[str] = None):
        super().__init__(self._callback, self._reportName, self._columns, self._sortColumns, self._sortTypes, pageNumber, pageSize, self._source,
                         self._client, date_range or self._date_range, self._other_fields, str_filter)


class StockStatParams(Params):
    _callback = CALLBACK['stock_stat']
    _reportName = REPORTNAME['stock_stat']
    _other_fields = {'INTERVAL_TYPE': "1", 'MUTUAL_TYPE': ['001', '003']}
    _source = WEB
    _client = WEB
    _columns = ALL
    _sortColumns = 'TRADE_DATE'
    _sortTypes = -1
    _date_range = Yesterday

    def __init__(self,
                 pageNumber: int = 1,
                 pageSize: int = 50,
                 date_range: Optional[Union[str, list, Tuple, dict]] = None,
                 str_filter: Optional[str] = None):
        super().__init__(self._callback, self._reportName, self._columns, self._sortColumns, self._sortTypes, pageNumber, pageSize, self._source,
                         self._client, date_range or self._date_range, self._other_fields, str_filter)


class InstStatParams(Params):
    _callback = CALLBACK['inst_stat']
    _reportName = REPORTNAME['inst_stat']
    _other_fields = {'MARKET_TYPE': "N"}
    _source = WEB
    _client = WEB
    _columns = ALL
    _sortColumns = 'HOLD_DATE'
    _sortTypes = -1
    _date_range = Yesterday

    def __init__(self,
                 pageNumber: int = 1,
                 pageSize: int = 50,
                 date_range: Optional[Union[str, list, Tuple, dict]] = None,
                 str_filter: Optional[str] = None):
        super().__init__(self._callback, self._reportName, self._columns, self._sortColumns, self._sortTypes, pageNumber, pageSize, self._source,
                         self._client, date_range or self._date_range, self._other_fields, str_filter)


class InstDetailParams(Params):
    _callback = CALLBACK['inst_detail']
    _reportName = REPORTNAME['inst_detail']
    _other_fields = {'MARKET_CODE': ['001', '003']}
    _other_fields = None
    _source = WEB
    _client = WEB
    _columns = ALL
    _sortColumns = ('HOLD_DATE', 'SECURITY_CODE')
    _sortTypes = (-1, 1)
    _date_range = Yesterday

    def __init__(self,
                 pageNumber: int = 1,
                 pageSize: int = 50,
                 date_range: Optional[Union[str, list, Tuple, dict]] = None,
                 str_filter: Optional[str] = None):
        super().__init__(self._callback, self._reportName, self._columns, self._sortColumns, self._sortTypes, pageNumber, pageSize, self._source,
                         self._client, date_range or self._date_range, self._other_fields, str_filter)


class GetData:
    uri = BASEURI

    def __init__(self, params: Params) -> None:
        self.logger = get_module_logger('GetData')
        self.params = params
        self._len = len(self.params.callback) + 1

    def __call__(self, page: int, headers: Optional[dict] = None, retry_time=3, retry_interval=3, timeout=5):
        self.params.pageNumber = page
        while True:
            try:
                headers = self._get_headers() if headers is None else headers
                _raw = requests.get(self.uri, params=self.params.to_dict(), headers=headers, timeout=timeout)
                _js = json.loads(_raw.text[self._len:-2])
                assert _js['success'], '抓取失败'
                self.logger.info('%s 页/ %s (* %s) 页' % (self.params.pageNumber, self.params.pages, self.params.pageSize))
                return pd.DataFrame(_js['result']['data'])
            except Exception as e:
                self.logger.error("requests error: %s" % str(e))
                retry_time -= 1
                if retry_time <= 0:
                    self.logger.error(f'抓取失败!返回空值. 相关参数: pageSize: {self.params.pageSize}, pageNumber: {self.params.pageNumber}')
                    return pd.DataFrame()
                self.logger.info("retry %s seconds after" % retry_interval)
                time.sleep(retry_interval)

    def _get_headers(self):
        _headers = HEADERS.copy()
        _headers['User-Agent'] = UserAgent().random
        return _headers


class NorthFundV3:
    logger = get_module_logger('file.NorthFundV3')

    @func_timer
    def get_data(self, params: Params, parallel: bool = False):
        handle = GetData(params)
        params.pageSize = 500
        if not parallel:
            out = [handle(p) for p in trange(1, params.pages + 1)]
            return pd.concat(out, axis=0, ignore_index=True).drop_duplicates()
        pool = Pool(processes=int(mp.cpu_count()))
        out = pool.map(handle, list(range(params.pages)))
        pool.close()
        pool.join()
        return pd.concat(out, axis=0, ignore_index=True).drop_duplicates()

    def get_history(self, params: Params, date_range=None):
        params.date_range = {} if date_range is None else date_range
        return self.get_data(params, parallel=True)

    def update(self, params: Params):
        params.date_range = Yesterday
        return self.get_data(params, parallel=True)

    def get_stock_stat(self, history: bool = False):
        params = StockStatParams()
        return self.update(params) if not history else self.get_history(params)

    def get_inst_stat(self, history: bool = False):
        params = InstStatParams()
        return self.update(params) if not history else self.get_history(params)

    def get_stock_detail(self, history: bool = False):
        params = StockDetailParams()
        return self.update(params) if not history else self.get_history(params)

    def get_inst_detail(self, history: bool = False, save=True):
        params = InstDetailParams()
        out = self.update(params) if not history else self.get_history(params)
        if save:
            f_path = Path(__file__).resolve().parent.parent / 'Data' / 'NorthFund'
            file_name = f'{Yesterday}_history.csv' if history else f'{Yesterday}.csv'
            out.to_csv(str(f_path / file_name), index=False, encoding='utf-8_sig')
        return out


if __name__ == '__main__':

    _ = NorthFundV3().get_inst_detail(history=True)
