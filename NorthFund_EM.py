#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   NorthFund_EM.py
@Time    :   2022/09/02 00:06:47
@Author  :   Liu Junli
@Version :   1.0
@Contact :   liujunli.xmu@foxmail.com
'''

from EastMoney import NorthFundV3, InstDetailParams
from basesqlv3 import WriteSQLV2

CONN = dict(host='10.224.1.70', user='liujl', passwd='CEQZqwer', database='liujl')
TABLE_NAME = 'northbound_shareholding'

if __name__ == '__main__':
    data = NorthFundV3().get_inst_detail(history=False)
    # date_range = ['2022-09-05', '2022-09-06', '2022-09-07', '2022-09-08', '2022-09-09']
    # params=InstDetailParams()
    # params.date_range = date_range
    # data = NorthFundV3().get_data(params, parallel=False)
    # data = NorthFundV3().get_history(params=InstDetailParams(), date_range=date_range)
    sql = WriteSQLV2(CONN)
    # sql.start_conn()
    # sql.delete_records_by_fields(TABLE_NAME, date_range=date_range)
    with sql.start():
        sql.write(data, TABLE_NAME, index=False, rec_exists='ignore')
    sql.logger.info('NorthFund更新成功!')
    NorthFundV3().logger.info('NorthFund更新成功!')