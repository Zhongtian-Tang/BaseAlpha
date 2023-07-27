#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   factor_testV2.py
@Time    :   2022/08/19 14:30:00
@Author  :   Liu Junli
@Version :   1.0
@Contact :   liujunli.xmu@foxmail.com
'''

from BaseAlphaV2 import DataSet, AlphaSet, FactorCheck

if __name__ == '__main__':
    dataset = DataSet()
    alphaset = AlphaSet()

    alphaset.load_data()
    alphaset.alpha = ...

    alphaset.pre_process(dataset, ifmp=True, w_method='equal')
    fc = FactorCheck(dataset, alphaset)
    fc.factor_suffix = 'cne5'
    fc.run()
