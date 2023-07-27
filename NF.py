#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   NorthFund.py
@Time    :   2022/08/09 15:47:06
@Author  :   Liu Junli
@Version :   1.0
@Contact :   liujunli.xmu@foxmail.com
'''
import numpy as np
import pandas as pd
from typing import Union, List, Optional
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from tqdm import trange
from pathlib import Path
import time
import json
from utils import func_timer
from logger import get_module_logger, set_log_with_config

set_log_with_config()


class NorthFundV2:
    url = 'https://www3.hkexnews.hk/sdw/search/mutualmarket_c.aspx?t=sh&t=%s'
    url_inst = 'https://www3.hkexnews.hk/sdw/search/searchsdw_c.aspx'
    region = ['sh', 'sz']
    dir = Path('NorthFund')

    def __init__(self, date: Optional[Union[str, int, pd.Timestamp]] = None, dir: Optional[Union[str, Path]] = None) -> None:
        self._date = pd.to_datetime(str(date)).date() if date is not None else pd.to_datetime('today').date()
        self.dir = Path(dir) if dir is not None else self.dir
        if not self.dir.exists():
            self.dir.mkdir(parents=True, exist_ok=True)
        self.state: bool = False
        self.code = {}
        self.code_map = {}
        self.logger = get_module_logger('NorthFundV2')

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, date: Union[str, int, pd.Timestamp]):
        self._date = pd.to_datetime(str(date)).date()

    def start(self):
        options = Options()
        options.add_argument("--headless")  # 无界面
        self.brow = webdriver.Edge(options=options)
        self.state = True

    def exit(self):
        if self.state:
            self.brow.quit()
            self.state = False

    @func_timer
    def _get_code(self, ret: bool = False) -> Optional[bool]:
        if not self.state:
            self.start()
        data = []
        for region in self.region:
            self.logger.info('开始抓取 %s' % region)
            self.brow.get(self.url % region)
            # 英文有code
            if 'EN' in self.brow.find_element(By.XPATH, '//*[@id="hkex_news_header"]/header/div[2]/div/div').text:
                self.brow.find_element(By.XPATH, '//*[@id="hkex_news_header"]/header/div[2]/div/div/a[1]').click()
            # 输入日期
            js = 'document.getElementById("txtShareholdingDate").value="{}";'.format(self.date.strftime('%Y/%m/%d'))
            self.brow.execute_script(js)
            # 点击搜索
            self.brow.find_element(By.XPATH, '//*[@id="btnSearch"]').click()
            # 对比日期是否一致：
            date = self.brow.find_element(By.XPATH, '//*[@id="pnlResult"]/h2/span').text.split(':')[1]
            if pd.to_datetime(date).date() != self.date:
                self.logger.error('%s不存在数据,请检查' % self.date)
                return False
            # 列名
            # col_names = self.brow.find_element(By.XPATH, '//*[@id="mutualmarket-result"]/thead').text.split(' ')
            col_names = ['Code', 'Name', 'Shareholding', 'Percentage']
            ncol = len(col_names)
            # 表格主体
            _text = self.brow.find_element(By.XPATH, '//*[@id="mutualmarket-result"]/tbody').text.split('\n')
            out = pd.DataFrame([_text[i:i + ncol] for i in range(0, len(_text), ncol)], columns=col_names)
            # 日期列：
            out['Date'] = date
            out['ACode'] = out['Name'].apply(lambda s: s.strip(' ')[-7:-1] + '.' + region.upper())
            data.append(out)
            self.logger.info('%s: %s 抓取完成' % (region, self.date))

        data = pd.concat(data, axis=0).reset_index(drop=True).apply(lambda s: s.str.strip(' '))
        data['Date'] = pd.to_datetime(data['Date']).dt.date
        data['Shareholding'] = data['Shareholding'].str.replace(',', '').astype(float)
        data['Percentage'] = data['Percentage'].str.strip('%').astype(float) / 100
        data['Name'] = data['Name'].str.strip(' ')
        self.code.update(data.groupby('Date')['Code'].apply(list).to_dict())
        self.code_map.update(data.set_index('Code')['ACode'].to_dict())
        data.to_csv(self.dir / ('totalshareholding_%s.csv' % self.date), index=False, encoding="utf_8_sig")
        self.exit()
        if ret:
            return data

    def _get(self, code, retry_time=3, retry_interval=5):
        while True:
            try:
                self.brow.find_element(By.XPATH, '//*[@id="txtStockCode"]').clear()
                self.brow.find_element(By.XPATH, '//*[@id="txtStockCode"]').send_keys(code)
                # 点击搜索
                self.brow.find_element(By.XPATH, '//*[@id="btnSearch"]').click()
                # 列名
                # col_names = self.brow.find_element(By.XPATH, '//*[@id="pnlResultNormal"]/div[2]/div/div[2]/table/thead').text.split(' ')
                col_names = ['ID', 'Name', 'Address', 'Shareholding', 'Percentage']
                ncol = len(col_names)
                # 表格主体
                _text = self.brow.find_element(By.XPATH, '//*[@id="pnlResultNormal"]/div[2]/div/div[2]/table/tbody').text.split('\n')
                _out = pd.DataFrame([_text[i:i + ncol] for i in range(0, len(_text), ncol)], columns=col_names)
                # 日期列
                _out['Date'] = self.brow.find_element(By.XPATH, '//*[@id="txtShareholdingDate"]').get_attribute('value')
                _out['Code'] = code
                _out['ACode'] = self.code_map[code]
                self.logger.info('%s: %s 抓取完成' % (self.date, code))
                return _out
            except Exception as e:
                self.logger.error("requests error: %s" % str(e))
                retry_time -= 1
                if retry_time <= 0:
                    return pd.DataFrame()
                self.logger.info("retry %s second after" % retry_interval)
                time.sleep(retry_interval)

    @func_timer
    def get_data(self):
        if not self.state:
            self.start()
        if self.code.get(self.date, None) is None:
            e = self._get_code()
            if e is not None and not e:
                return
        self.brow.get(self.url_inst)
        js = 'document.getElementById("txtShareholdingDate").value="{}";'.format(self.date.strftime('%Y/%m/%d'))
        self.brow.execute_script(js)
        if 'EN' in self.brow.find_element(By.XPATH, '//*[@id="hkex_news_header"]/header/div[2]/div/div').text:
            self.brow.find_element(By.XPATH, '//*[@id="hkex_news_header"]/header/div[2]/div/div/a[1]').click()
        self.logger.info('开始抓取 %s' % self.date)
        out = [self._get(self.code[self.date][i]) for i in trange(len(self.code[self.date]))]
        data = pd.concat(out, axis=0).reset_index(drop=True).apply(lambda s: s.str.strip(' '))
        data['Shareholding'] = data['Shareholding'].str.replace(',', '').astype(float)
        data['Percentage'] = data['Percentage'].str.strip('%').astype(float) / 100
        data['Date'] = pd.to_datetime(data['Date']).dt.date
        # data['股票代码'] = data['股票代码'].map(handle_code)
        self.exit()
        self.data = data
        data.to_csv(self.dir / ('shareholdingdetail_%s.csv' % self.date), index=False, encoding="utf_8_sig")
        self.logger.info('%s: 抓取完成' % self.date)


if __name__ == '__main__':
    nf = NorthFundV2(date=20220823)
    nf.get_data()
