#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   start.py
@Time    :   2022/08/17 11:01:05
@Author  :   Liu Junli
@Version :   1.0
@Contact :   liujunli.xmu@foxmail.com
'''

from config import C
from BaseAlphaV2 import BaseDataSet, BaseAlphaSet, BaseFactorCheck

connect = {
    'huaan': dict(host='10.224.16.81', user='haquant', passwd='haquant', database='jydb', port=3306, charset='utf8'),
    "tonglian": dict(host='10.224.0.21', user='HAzc_1805', passwd='rededata', database='datayesdb', port=1521, charset='utf8'),
    "write": dict(host='10.224.1.70',
                  user='liujl',
                  passwd='CEQZqwer',
                  port=3306,
                  database='liujl',
                  charset='utf8',
                  auth_plugin='mysql_native_password'),
    "local": dict(host='localhost', user='root', passwd='huaan', port=3306, charset='utf8', auth_plugin='mysql_native_password')
}

C.set(dict(connect=connect['huaan']))  # 设置默认配置, 可以传入默认参数
C.register()