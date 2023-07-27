#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   constants.py
@Time    :   2022/08/24 14:01:34
@Author  :   Liu Junli
@Version :   1.0
@Contact :   liujunli.xmu@foxmail.com
'''
from typing import Literal
import QuantLib as ql
# 数据接口
BASEURI = 'https://datacenter-web.eastmoney.com/api/data/v1/get'

WEB = 'WEB'
ALL = 'ALL'
HOLD_DATE = 'HOLD_DATE'

CAL = ql.China(ql.China.SSE)
Yesterday = str(CAL.advance(ql.Date.todaysDate(), ql.Period(-1, ql.Days)).to_date())

SORT = Literal[1, -1]

P = ['reportName', 'callback', 'sortColumns', 'pageNumber', 'sortTypes', 'pageSize', 'columns', 'source', 'client', 'filter']

FORMAT = '%Y-%m-%d %H:%M:%S'

HEADERS = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'Connection': 'keep-alive',
    'Host': 'datacenter-web.eastmoney.com',
    'Referer': 'https://data.eastmoney.com/',
    'sec-ch-ua': '"Microsoft Edge";v="105", " Not;A Brand";v="99", "Chromium";v="105"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': "Windows",
    'Sec-Fetch-Dest': 'script',
    'Sec-Fetch-Mode': 'no-cors',
    'Sec-Fetch-Site': 'cross-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36',
}

CALLBACK = {
    'inst_stat': 'jQuery1123012984548903807314_1661307472049',
    'inst_detail': 'jQuery112309953797292661735_1661311558963',
    'stock_stat': 'jQuery1123049041688193023725_1661309274245',
    'stock_detail': 'jQuery112307209306494450614_1661310812995',
}

REPORTNAME = {
    'inst_stat': 'PRT_MUTUAL_ORG_STA',
    'inst_detail': 'RPT_MUTUAL_HOLD_DET',
    'stock_stat': 'RPT_MUTUAL_STOCK_NORTHSTA',
    'stock_detail': 'RPT_MUTUAL_HOLD_DET',
}
