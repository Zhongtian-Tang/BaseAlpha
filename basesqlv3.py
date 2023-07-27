#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   basesqlv3.py
@Time    :   2022/08/31 11:29:36
@Author  :   Liu Junli
@Version :   1.0
@Contact :   liujunli.xmu@foxmail.com
'''

import os
import pickle
import datetime
import hashlib
import mysql.connector
import pandas as pd
import numpy as np
from enum import Enum
from typing import Union, List, Optional, Dict, Text, Any, Tuple
from sqlalchemy import create_engine, VARCHAR, DATE, FLOAT, INT, DECIMAL, DATETIME, TIMESTAMP, BOOLEAN
from contextlib import contextmanager
from prettytable import from_db_cursor
from pprint import pprint

from utils import split_date, func_timer
from logger import get_module_logger, set_log_with_config


class BaseSQL:
    CONN = dict(host='localhost', user='root', passwd='huaan', port=3306, charset='utf8', auth_plugin='mysql_native_password')
    set_log_with_config()
    logger = get_module_logger('sql')

    STATE_SQL = 'sql'
    STATE_DB = 'database'

    TAB_TEMP = 'temp'

    DT_INT = 'int'
    DT_FLOAT = 'float'
    DT_VARCHAR = 'varchar'
    DT_DATE = 'date'
    DT_DECIMAL = 'decimal'
    DT_DOUBLE = 'double'
    DT_TINYINT = 'tinyint'
    DT_DATETIME = 'datetime'
    DT_TIMESTAMP = 'timestamp'
    DT_BOOLEAN = 'boolean'

    # 把python的数据类型转换为mysql的数据类型, mysql.execute()提交
    _py_cls_map = {
        str: DT_VARCHAR,
        bool: DT_BOOLEAN,
        int: DT_INT,
        float: DT_DECIMAL + '(20,6)',
        np.int64: DT_INT,
        np.float64: DT_DECIMAL + '(20, 6)',
        datetime.date: DT_DATE,
        datetime.datetime: DT_DATETIME,
    }
    # 把mysql.execute()读取出来的mysql数据类型转换为sqlalchemy类型, 用于创建新表
    _cls_sql_map = {
        DT_VARCHAR: VARCHAR,
        DT_INT: INT,
        DT_FLOAT: FLOAT,
        DT_DATE: DATE,
        DT_DECIMAL: DECIMAL(19, 6),
        DT_TINYINT: INT,
        DT_DATETIME: DATETIME,
        DT_TIMESTAMP: TIMESTAMP,
        DT_BOOLEAN: BOOLEAN
    }

    class EXIST(Enum):
        RAISE = 'raise'
        REPLACE = 'replace'
        IGNORE = 'ignore'
        APPEND = 'append'
        FILL = 'fill'

    def __init__(self, conn: Optional[dict] = None, database: Optional[str] = None, read_only: bool = False) -> None:
        self.state = None
        # self.active_database = None # 2022-08-31 Junli
        self.active_table = None
        self.CONN.update(conn) if conn is not None else None
        self.last_database = database  #初始化时，可以传入数据库名，以便连接时不用输入数据库名
        self.read_only = read_only

    @contextmanager
    def start(self, conn2db: Optional[bool] = None, database_name: Optional[str] = None):
        try:
            self.start_conn(conn2db, database_name)
            yield
        except Exception as e:
            self.close_conn()
            raise e
        self.close_conn()

    def start_conn(self, database_name: Optional[str] = None, conn2db: Optional[bool] = None):

        if database_name is not None or conn2db is True:
            return self.connect2database(database_name)

        if conn2db is False:
            return self.connect2sql()

        return self.connect()

    def close_conn(self):
        if self.state is not None:
            if self.state == BaseSQL.STATE_DB:
                self.last_database = self.active_database
            self.db.commit()
            self.db.close()
            self.state = None
            self.logger.info('数据库连接关闭')

    def connect2database(self, database_name: str):
        if (database_name is None or database_name == '') and ('database' not in self.CONN or self.CONN['database'] is None):
            self.logger.warning('数据库未指定, 尝试连接至上一次连接的数据库')
            assert self.last_database is not None, '连接失败: 上一次连接的数据库不存在. 请指定数据库名.'
            database_name = self.last_database

        self.CONN.update(dict(database=database_name))  # NOTE: 唯一允许更改CONN的地方
        self.connect()
        self.logger.info('%s 数据库连接成功' % database_name)

    def connect2sql(self):
        self.last_database = self.CONN.pop('database', None)
        self.connect()
        self.logger.info('sql连接成功')
        return 0

    def connect(self):
        if self.state is not None:
            self.logger.warning('数据库已连接, 尝试关闭连接')
            self.close_conn()

        if 'database' in self.CONN.keys() and self.CONN['database'] is None:
            del self.CONN['database']

        if 'database' not in self.CONN.keys():
            self.logger.warning('数据库未指定! 尝试连接至sql')

        try:
            self.db = mysql.connector.connect(**self.CONN)
            self.cr = self.db.cursor(buffered=True)

        except:
            if 'database' in self.CONN.keys():
                _db = self.CONN.pop('database', None)
                try:
                    self.connect()
                except:
                    raise Exception('数据库连接失败: %s' % _db)

                if _db not in self.list_databases():  # 2022-08-31 Junli
                    self.logger.warning('企图连接的数据库 %s 不存在, sql存在以下数据库: %s,请尝试连接' % (_db, self.list_databases()))
                else:
                    self.logger.warning('企图连接的数据库 %s 存在, 但未能成功连接, 连接至sql' % _db)
            else:
                raise Exception('连接失败')

        self.logger.info('连接成功')
        if 'database' not in self.CONN.keys():
            self.state = BaseSQL.STATE_SQL
        else:
            self.state = BaseSQL.STATE_DB

        return 0

    def refresh(self):
        pre_database = self.active_database
        pre_table = self.active_table
        self.start_conn(pre_database)
        if pre_table is not None:  # 2022-08-31 修改
            self.set_active_table(pre_table)

    @contextmanager
    def refresh_with(self):
        pre_database = self.active_database
        pre_table = self.active_table
        yield
        self.start_conn(pre_database)
        if pre_table is not None:  # 2022-08-31 修改
            self.set_active_table(pre_table)

    def set_mode(self, read_only: bool):
        self.read_only = read_only

    def set_active_database(self, db_name: str):
        assert db_name in self.list_databases(), '数据库不存在'
        self.execute('use %s' % db_name)
        return db_name

    @property
    def active_database(self):
        if self.state is not None:
            self.execute("SELECT DATABASE ()")
            return [db[0] for db in self.cr][0]
        else:
            return None

    def set_active_table(self, table_name: str):
        assert table_name in self.list_tables(), '数据表不存在'
        self.active_table = table_name
        return self.active_table

    def list_databases(self):
        # 任何位置都可以列出
        self.cr.execute('show databases')
        return [db[0] for db in self.cr]

    def list_tables(self):
        self._check_db_status()
        self.cr.execute('show tables')
        return [db[0] for db in self.cr]

    def list_all_tables(self, database_names: List[str] = None):
        assert self.state is not None, '未连接!'  # 2022-08-31 Junli
        database_names = database_names or self.list_databases()
        with self.refresh_with():
            _tabs = {}
            for db in database_names:
                self.set_active_database(db)
                _tabs.update({db: self.list_tables()})
        pprint(_tabs)

    def preview_table(self, table_name: Optional[str] = None, lines: int = 30):
        table_name = self._check_tab_status(table_name)
        exp = 'select * from %s  LIMIT 0,%s;' % (table_name, lines)
        self.execute(exp)
        print(from_db_cursor(self.cr))

    def list_date_range(self, table_name: Optional[str] = None):
        table_name = self._check_tab_status(table_name)
        field_date = self.list_date_key(table_name)
        exp = 'SELECT MIN(%s), MAX(%s) FROM %s' % (field_date, field_date, table_name)
        self.execute(exp)
        from_db_cursor(self.cr)
        return [idate.strftime('%Y-%m-%d') for idate in pd.read_sql(exp, self._engine).squeeze().to_list()]

    def list_date_key(self, table_name: Optional[str] = None):
        table_name = self._check_tab_status(table_name)
        fields_date = [k for k, dt in self.list_keys(table_name, detail=True).items() if dt is DATE]
        assert len(fields_date) >= 1, '表中无日期字段'
        return fields_date[0]

    def list_keys(self, table_name: Optional[str] = None, detail: bool = False) -> Union[List, Dict]:
        table_name = self._check_tab_status(table_name)
        if not detail:
            self.execute(""" SELECT column_name FROM INFORMATION_SCHEMA.`KEY_COLUMN_USAGE` 
                             WHERE table_name='%s' AND constraint_name='PRIMARY'
                         """ % (table_name))
            return [db[0] for db in self.cr]
        else:
            self.execute(""" SELECT column_name,data_type FROM information_schema.columns
                             WHERE table_name='%s'
                         """ % (table_name))
            _data_type = {db[0]: str(db[1], encoding='utf8') if type(db[1]) is bytes else db[1] for db in self.cr}
            return {
                k: self._cls_sql_map[v] if (v in self._cls_sql_map.keys()) else v
                for k, v in _data_type.items() if k in self.list_keys(table_name, detail=False)
            }

    def list_key_values(self, table_name: Optional[str] = None) -> pd.DataFrame:
        table_name = self._check_tab_status(table_name)
        return pd.read_sql(""" SELECT %s from %s """ % (','.join(self.list_keys(table_name)), table_name), self._engine)

    def list_fields(self, table_name: Optional[str] = None, detail: bool = False) -> Union[List, Dict]:
        table_name = self._check_tab_status(table_name)
        if not detail:
            self.execute("SELECT column_name FROM information_schema.columns WHERE table_name='%s'" % (table_name))
            return [db[0] for db in self.cr]
        else:
            self.execute(""" SECELT column_name,data_type FROM information_schema.columns
                             WHERE table_name='%s'
                         """ % (table_name))
            _data_type = {db[0]: str(db[1], encoding='utf8') if type(db[1]) is bytes else db[1] for db in self.cr}
            return {k: self._cls_sql_map[v] if (v in self._cls_sql_map.keys()) else v for k, v in _data_type.items()}

    def _check_db_status(self):
        assert self.state == BaseSQL.STATE_DB, '未连接至数据库！'

    def _check_tab_status(self, table_name: Optional[str] = None):
        self._check_db_status()
        table_name = table_name or self.active_table
        assert table_name and (table_name in self.list_tables()), '数据表未指定 或 数据表不存在'
        return table_name

    def _check_duplicate_fields(self, table_config: Dict[Text, Any]):
        table_name = table_config.get('table_name', None) or self.active_table
        field_names = [field_config['field_name'] for field_config in table_config['field_configs']]
        _fields_exists = set(field_names) & set(self.list_fields(table_name))
        if bool(_fields_exists):
            self.logger.warning('字段 %s 已存在' % ','.join(list(_fields_exists)))
        return list(set(field_names) - _fields_exists), bool(_fields_exists)

    def _check_permissions(self):
        assert not self.read_only, '没有写入权限！'

    @staticmethod
    def _remove_inf_nan(data: pd.DataFrame) -> pd.DataFrame:
        # return data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
        return data.replace([np.inf, -np.inf, np.nan], None)

    def _fieldconfigs2sqlexp(self, exps: List[Dict[Text, Any]]) -> str:

        return ','.join([self._fieldconfig2sqlexp(exp) for exp in exps])

    def _fieldconfig2sqlexp(self, exp: Dict[Text, Any]) -> str:
        assert exp['field_name'] is not None, '字段名未指定'
        out = ['%s %s' % (exp['field_name'], exp.get('field_type', self.DT_FLOAT))]
        if 'notnull' in exp.keys() and exp['notnull']:
            out.append('NOT NULL')
        else:
            out.append('DEFAULT NULL')
        if 'comment' in exp.keys():
            out.append("COMMENT '%s'" % exp['comment'])
        if 'pos' in exp.keys():
            out.append(' %s' % exp.get('pos'))

        return ' '.join(out)

    def _get_simple_config(self, field_names: List[str], table_name: Optional[str] = None):
        _get_field_config = lambda _field: {
            'field_name': _field,
            'field_type': self.DT_DECIMAL,
        }
        return {
            'table_name': table_name or self.active_table,
            'field_configs': [_get_field_config(field_name) for field_name in field_names],
        }

    def _get_detail_config(self, data: pd.DataFrame, table_name: Optional[str] = None, index: bool = True):
        _keys = list(data.index.names)
        index = index and _keys != [None]
        if index:
            data = data.reset_index()
        field_names = list(data.columns)
        _type = [self._py_cls_map.get(type(data[_field][data[_field].first_valid_index()]), self.DT_FLOAT) for _field in field_names]
        _len = [len(str(data[_field][0])) for _field in field_names]
        # _fulltype = [itype + '(' + str(ilen) + ')' if itype in [self.DT_VARCHAR] else itype for itype, ilen in zip(_type, _len)]
        _fulltype = [itype + '(' + str(20) + ')' if itype in [self.DT_VARCHAR] else itype for itype, ilen in zip(_type, _len)]
        # Junli 2022-09-02: 0:44

        _get_notnull = lambda _field: _field in _keys
        _get_field_config = lambda _field, _type: {'field_name': _field, 'field_type': _type, 'notnull': _get_notnull(_field)}
        out = {
            'table_name': table_name or self.active_table,
            'field_configs': [_get_field_config(field_name, _type) for field_name, _type in zip(field_names, _fulltype)],
        }
        if index:
            out.update({'keys': _keys})
        return out

    @property
    def _engine(self):
        return create_engine('mysql+mysqlconnector://%s:%s@%s:%s/%s?auth_plugin=%s' %
                             (self.CONN['user'], self.CONN['passwd'], self.CONN['host'], self.CONN['port'], self.CONN['database']
                              or self.active_database, self.CONN['auth_plugin']),
                             encoding=self.CONN['charset'])

    def execute(self, exp: str):
        self.cr.execute(exp)
        self.db.commit()

    @func_timer
    def executemany(self, exp: str, values: Union[List[Tuple], Tuple[Tuple]]):
        try:
            self.cr.executemany(exp, values)
            self.logger.info('执行成功')
        except:
            self.db.rollback()
            self.logger.warning('执行失败')
            raise Exception('执行失败')
        finally:
            self.db.commit()

    @staticmethod
    def load(file_path: str):
        return pickle.load(open(file_path, 'rb'))

    @staticmethod
    def save(file_path: str, data):
        return pickle.dump(data, open(file_path, 'wb'))

    @func_timer
    def read(self,
             table_name: Optional[str] = None,
             field_names: Optional[Union[List[str], str]] = None,
             date_range: Union[str, list, Tuple, Dict[str, str]] = {},
             other_fields: Optional[Dict[str, Union[str, list, Tuple]]] = None,
             other_cond: Optional[str] = None) -> pd.DataFrame:

        table_name = self._check_tab_status(table_name)
        field_names = [field_names] if isinstance(field_names, str) else field_names

        field_names = sorted(list(set(self.list_keys(table_name)) | set(field_names)) if field_names is not None else self.list_fields(table_name))
        assert set(field_names) <= set(self.list_fields(table_name)), '部分字段不存在'
        exp = 'SELECT %s FROM %s WHERE' % (','.join(field_names) or '*', table_name)

        if isinstance(date_range, dict):
            st = date_range.get('start', None) if date_range.get('start', None) is not None else self.list_date_range(table_name)[0]
            ed = date_range.get('end', None) if date_range.get('end', None) is not None else self.list_date_range(table_name)[1]
            return pd.concat([self.read(table_name, field_names, (st_, ed_), other_fields, other_cond) for st_, ed_ in zip(*split_date(st, ed))])

        cond = self.getwhere(table_name, date_range, other_fields, other_cond)

        return self._read(exp + cond) if cond != '' else pd.DataFrame()

    def _read(self, exp: str) -> pd.DataFrame:
        # 提取table_name
        #! exp = exp.upper()
        #! 约定成俗,所有的sql关键字用大写!!!
        assert 'FROM' in exp, '请检查语句是否正确'
        table_name = exp.split('FROM')[1].split('WHERE')[0].strip() if 'WHERE' in exp else exp.split('FROM')[1].replace(';', '').strip()
        path = os.path.join('temp', hashlib.md5(exp.replace(', ', ' ').replace(',', ' ').replace(' ', '_').encode('utf-8')).hexdigest())  #不要改
        if os.path.exists(path):
            data = self.load(path)
        else:
            # 这里默认每次读取数据一定包括索引
            data = pd.read_sql(exp, con=self._engine)
            _key_read = set(self.list_keys(table_name)) & set(data.columns)
            if _key_read:
                data.set_index(sorted(list(_key_read)), inplace=True)
            # if len(path)
            os.makedirs('temp') if not os.path.exists('temp') else None
            self.save(path, data)
        return data

    def _getwhere(self, field: str, cond: Union[str, list, Tuple]):
        if isinstance(cond, str):
            return " (%s='%s') " % (field, cond)

        if isinstance(cond, list):
            return ' (%s IN %s) ' % (field, tuple(cond))

        if isinstance(cond, tuple):
            assert len(cond) == 2, '%s 范围参数错误' % (field)
            return " (%s BETWEEN '%s' AND '%s')" % (field, cond[0], cond[1])

    def getwhere(self,
                 table_name: Optional[str] = None,
                 date_range: Optional[Union[str, list, Tuple]] = None,
                 other_fields: Optional[Dict[str, Union[str, list, Tuple]]] = None,
                 other_cond: Optional[str] = None) -> pd.DataFrame:
        table_name = self._check_tab_status(table_name)
        assert set(other_fields.keys()) <= set(self.list_fields(table_name)) if other_fields else True, '部分字段不正确'
        cond = []
        fields_date = [k for k, dt in self.list_keys(table_name, detail=True).items() if dt is DATE]
        cond.append(self._getwhere(fields_date[0], date_range) if date_range and fields_date != [] else '')
        cond.append(' AND '.join([self._getwhere(f, c) for f, c in other_fields.items()]) if other_fields else '')
        cond.append(other_cond if other_cond else '')
        cond = [c for c in cond if c != '']
        return ' AND '.join(cond)


class WriteSQL(BaseSQL):
    def __init__(self, database: Optional[str] = None, read_only: bool = False):
        super().__init__(database, read_only)

    def write(self,
              data: pd.DataFrame,
              table_name: Optional[str],
              tab_exists: str = 'append',
              index: bool = True,
              rec_exists: Enum = BaseSQL.EXIST.IGNORE,
              dtype: Optional[Dict[str, object]] = None):
        self.add_records(data, table_name, tab_exists, index, rec_exists, dtype)

    def create_database(self, database_name: str, conn: bool = True):
        self._check_permissions()
        # 任何位置都可以创建
        assert database_name not in self.list_databases(), '数据库已存在'
        try:
            self.cr.execute('create database %s' % database_name)
            self.db.commit()
            self.logger.info('数据库创建成功')
            self.active_database = database_name
        except:
            raise Exception('数据库创建失败')

        if conn:
            self.start_conn(database_name)
            self.logger.info('数据库连接成功')
        return 0

    def delete_database(self, database_name: str):
        self._check_permissions()
        # 任何位置都可以删除
        assert database_name in self.list_databases(), '数据库不存在'
        if database_name == self.active_database:
            self.logger.warning('数据库正在使用中, 尝试退出数据库, 重新连接至sql')
            self.connect2sql()
        try:
            self.cr.execute('drop database %s' % database_name)
            self.db.commit()
            self.logger.info('数据库删除成功')
        except:
            raise Exception('数据库删除失败')
        return 0

    def create_table(self, table_config: Dict[Text, Any], set_active_table: bool = False):
        self._check_permissions()
        self._check_db_status()
        table_name = table_config.get('table_name', None) or self.active_table
        assert table_name not in self.list_tables(), '数据表已存在'
        exp_prefix = 'create table %s (' % table_name
        assert 'field_configs' in table_config, 'field_configs未指定'
        exp_m = self._fieldconfigs2sqlexp(table_config['field_configs'])
        field_names = [field_config['field_name'] for field_config in table_config['field_configs']]
        assert set(table_config.get('keys', [])) <= set(field_names), 'keys不存在于field_configs'
        exp_suffix = ',primary key (%s) );' % ', '.join(table_config.get('keys', []))
        exp = '%s %s %s' % (exp_prefix, exp_m, exp_suffix if table_config.get('keys', None) else exp_suffix[-2:])

        try:
            self.execute(exp)
            self.logger.info('数据表 %s 创建成功' % table_name)
            return 0
        except:
            self.logger.error('数据表创建失败')
        self.set_active_table(table_name) if set_active_table else None

    def create_table_like(self, new_table_name: Optional[str], data: Optional[pd.DataFrame] = None, table_name: Optional[str] = None):
        self._check_permissions()
        if data is not None:
            table_config = self._get_detail_config(data, new_table_name)
            return self.create_table(table_config)
        table_name = table_name or self.active_table
        assert table_name and (table_name in self.list_tables()), '数据表不存在'
        self.execute('create table %s like %s' % (new_table_name, table_name))
        self.refresh()

    def delete_table(self, table_name: str):
        self._check_permissions()
        self._check_db_status()
        assert table_name in self.list_tables(), '数据表不存在'
        self.logger.warning('当前活跃数据表 %s 即将被删除！' % table_name) if self.active_table == table_name else None
        try:
            self.cr.execute('drop table if exists %s ' % table_name)
            self.db.commit()
            self.active_table = None if self.active_table == table_name else self.active_table
            self.logger.warning('当前活跃数据表 %s 即将被删除' % table_name)
            self.logger.info('数据表 %s 删除成功' % table_name)
        except:
            raise Exception('数据表删除失败')
        return 0

    def clear_table(self, table_name: str):
        self._check_permissions()
        self.execute('truncate table %s' % table_name)

    def add_fields(self, data: pd.DataFrame, table_config: Optional[Dict[Text, Any]] = None, exist: Enum = BaseSQL.EXIST.IGNORE, index: bool = True):
        """add_field 

        增加一个新的字段

        Args:
            table_config (Dict[Text, Any]): 
                eg: {'table_name': 'table_name', 
                     'field_configs': [{'field_name': 'field_name', 'field_type': 'field_type'}, {}, ...],
                     'keys': ['field_name',...]}
                    }
        # Note: 完全依照table_config来添加字段, 不考虑数据表是否存在 
        """
        self._check_permissions()
        if table_config is None:
            table_config = self._get_detail_config(data[list(set(data.columns) - set(self.list_keys(self.active_table)))], self.active_table, False)
        _fields_remain, _duplicate = self._check_duplicate_fields(table_config)
        if _duplicate and exist == self.EXIST.RAISE:
            raise Exception('存在重复的字段')
        elif _duplicate and exist == self.EXIST.IGNORE:
            self.logger.warning('存在重复的字段, 忽略')
            field_names = _fields_remain
        else:
            field_names = [field_config['field_name'] for field_config in table_config['field_configs']]
        if bool(field_names):
            self.add_field_names(table_config)
            table_name = table_config.get('table_name', None) or self.active_table
            self.update_fields(data, field_names, table_name, index)

    def add_field_names(self, table_config: Dict[Text, Any]):
        self._check_permissions()
        self._check_db_status()
        table_name = table_config.get('table_name', None) or self.active_table
        assert table_name and (table_name in self.list_tables()), '数据表未指定 或 数据表不存在'
        exp_prefix = 'alter table %s add column (' % table_name
        assert 'field_configs' in table_config and isinstance(table_config['field_configs'], list), 'field_configs未指定或以非list形式传入'
        _fields_remain, _ = self._check_duplicate_fields(table_config)
        if len(_fields_remain) > 0:
            _field_config = [f for f in table_config['field_configs'] if f['field_name'] in _fields_remain]
            exp_m = self._fieldconfigs2sqlexp(_field_config)
            _filed_names_exp = '%s %s %s' % (exp_prefix, exp_m, ')')
            self.execute(_filed_names_exp)
            self.logger.info('数据表字段添加成功')

    def update_fields(self, data: pd.DataFrame, field_names: List[str] = None, table_name: Optional[str] = None, index: bool = True):
        self.update(data, field_names, table_name, index)

    def delete_field(self, field_name: str, table_name: Optional[str] = None):
        self._check_permissions()
        table_name = self._check_tab_status(table_name)
        assert field_name in self.list_fields(table_name), '字段%s不存在' % field_name
        try:
            self.execute('ALTER TABLE %s  DROP %s' % (table_name, field_name))
            self.logger.info('表 %s 中字段 %s 删除成功' % (table_name, field_name))
            return 0
        except:
            self.logger.error('表 %s 中字段 %s 删除失败' % (table_name, field_name))
            return -1

    def add_records(self,
                    data: pd.DataFrame,
                    table_name: Optional[str] = None,
                    tab_exists: Enum = BaseSQL.EXIST.APPEND,
                    index: bool = True,
                    rec_exists: Enum = BaseSQL.EXIST.IGNORE,
                    dtype: Optional[Dict[str, object]] = None):
        """
        1. 创建新表
        2. 覆盖原表
        3. 追加数据: 重复数据处理: ignore, replace, error
        """
        self._check_permissions()
        self._check_db_status()
        table_name = table_name or self.active_table
        assert table_name, '数据表未指定 或 数据表不存在'
        tab_exists = self.EXIST(tab_exists) if isinstance(tab_exists, str) else tab_exists
        assert tab_exists in self.EXIST.__members__.values(), 'if_exists参数错误'

        if not index:
            data.set_index(self.list_keys(table_name), inplace=True)

        if tab_exists == self.EXIST.REPLACE:
            self.logger.warning('数据表 %s 存在，将被替换' % table_name)
            self.clear_table(table_name)
            tab_exists = self.EXIST.APPEND

        if table_name not in self.list_tables():
            try:
                self.create_table_like(table_name, data=data)
            except:
                if dtype is None and self.active_table is not None:
                    dtype = self.list_keys(self.active_table, detail=True)
            data.to_sql(table_name, con=self._engine, chunksize=10000, if_exists=tab_exists.value, index=index, dtype=dtype)
            return self.refresh()

        # 多一列的处理
        _cols_more = list(set(data.reset_index().columns) - set(self.list_fields(table_name)))
        if bool(_cols_more):
            self.add_field_names(self._get_detail_config(data[_cols_more], table_name, index=False))
        try:
            # 有重复的记录会报错
            data.to_sql(table_name, con=self._engine, chunksize=10000, if_exists=tab_exists.value, index=index, dtype=dtype)
            self.refresh()
        except Exception as e:
            if rec_exists == self.EXIST.RAISE:
                self.logger.warning('与已有数据表有重复数据')
                raise e

            self.add_records(data, self.TAB_TEMP, tab_exists, index, rec_exists, dtype)
            field_names = list(set(data.columns) - set(self.list_keys(table_name)))
            _all_fields = ','.join(self.list_keys(table_name) + field_names)
            if rec_exists == self.EXIST.IGNORE:
                self.logger.warning('数据表 %s 存在重复记录，将被忽略' % table_name)
                exp = """insert ignore into %s (%s) select %s from %s """
            elif rec_exists == self.EXIST.REPLACE:
                self.logger.warning('数据表 %s 存在重复记录，将被替换' % table_name)
                exp = """replace into %s (%s) select %s from %s """
            self.execute(exp % (table_name, _all_fields, _all_fields, self.TAB_TEMP))
            self.clear_temp()
        self.logger.info('数据表更新成功')

    def update_records(self, data: pd.DataFrame, table_name: Optional[str] = None, index: bool = True):
        self._check_permissions()
        self.update(data, index=index, table_name=table_name)

    def delete_records_by_fields(self,
                                 table_name: Optional[str] = None,
                                 date_range: Union[str, list, Tuple] = None,
                                 other_fields: Optional[Dict[str, Union[str, list, Tuple]]] = None,
                                 other_cond: Optional[str] = None) -> pd.DataFrame:
        table_name = self._check_tab_status(table_name)
        exp = 'DELETE FROM %s WHERE' % (table_name)
        fields_date = [k for k, dt in self.list_keys(table_name, detail=True).items() if dt is DATE]
        cond = []
        cond.append(self._getwhere(fields_date[0], date_range) if date_range and fields_date != [] else '')
        cond.append(' And '.join([self._getwhere(f, c) for f, c in other_fields.items()]) if other_fields else '')
        cond.append(other_cond if other_cond else '')
        cond = [c for c in cond if c != '']
        return self.execute(exp + ' And '.join(cond)) if cond != [] else None

    def update(self, data: pd.DataFrame, field_names: List[str] = None, table_name: Optional[str] = None, index: bool = True):
        # 只能新增字段，不能新增记录
        self._check_permissions()
        table_name = table_name or self.active_table
        self.clear_temp()
        self.add_records(data, table_name=self.TAB_TEMP, index=index)
        if field_names is None:
            field_names = list(set(data.columns) - set(self.list_keys(table_name)))
        assert set(field_names) <= set(self.list_fields(table_name)), '部分字段不存在, 请先增加该部分字段'

        _update_exp = """update %s inner join temp on %s set %s
                      """ % (table_name, ' and '.join(['%s.%s=temp.%s' % (table_name, key, key) for key in self.list_keys(table_name)]), ','.join(
            ['%s.%s=temp.%s' % (table_name, field_name, field_name) for field_name in field_names]))
        _all_fields = ','.join(self.list_keys(table_name) + field_names)
        _replace_exp = """replace into %s (%s) select %s from %s""" % (table_name, _all_fields, _all_fields, self.TAB_TEMP)
        try:
            self.execute(_update_exp)
        except:
            self.execute(_replace_exp)
        self.clear_temp()
        self.logger.info('数据表字段更新成功')

    def clear_temp(self):
        self._check_permissions()
        self.delete_table(self.TAB_TEMP) if self.TAB_TEMP in self.list_tables() else None


class WriteSQLV2(BaseSQL):
    def __init__(self, conn: Optional[dict] = None, database: Optional[str] = None, read_only: bool = False):
        super().__init__(conn, database, read_only)

    @func_timer
    def write(self,
              data: pd.DataFrame,
              table_name: Optional[str] = None,
              tab_exists: Union[BaseSQL.EXIST, str] = 'append',
              index: bool = True,
              rec_exists: Union[BaseSQL.EXIST, str] = BaseSQL.EXIST.FILL):
        """
        tab_exists: self.EXIST.RAISE, self.EXIST.REPLACE, self.EXIST.APPEND
        rec_exists: self.EXIST.RAISE, self.EXIST.REPLACE, self.EXIST.IGNORE, self.EXIST.FILL
        #* note: 慎重为rec_exists赋值self.EXIST.REPLACE, 未指定的已有字段会被替换为默认值
        """
        self._check_permissions()
        self._check_db_status()
        table_name = table_name or self.active_table
        assert table_name, '数据表未指定 或 数据表不存在'
        data = self._remove_inf_nan(data)
        tab_exists = self.EXIST(tab_exists) if isinstance(tab_exists, str) else tab_exists
        rec_exists = self.EXIST(rec_exists) if isinstance(rec_exists, str) else rec_exists

        if table_name not in self.list_tables():
            self.create_table_like(table_name, data=data)
            tab_exists = self.EXIST.APPEND

        if index:
            data = data.reset_index()

        if tab_exists == self.EXIST.RAISE:
            raise Exception('数据表 %s 已存在' % table_name)

        if tab_exists == self.EXIST.REPLACE:
            self.logger.warning('数据表 %s 存在，将被替换' % table_name)
            self.clear_table(table_name)
            tab_exists = self.EXIST.APPEND

        field_names = list(data.columns)
        # 多一列的处理
        _cols_more = list(set(field_names) - set(self.list_fields(table_name)))
        if bool(_cols_more):
            self.add_field_names(self._get_detail_config(data[_cols_more], table_name, index=False))

        if rec_exists == self.EXIST.RAISE:
            exp = """insert into %s (%s) values (%s) """
        elif rec_exists == self.EXIST.IGNORE:
            exp = """insert ignore into %s (%s) values (%s) """
        elif rec_exists == self.EXIST.FILL:
            exp = f"""insert into %s (%s) values (%s) on duplicate key update {','.join(['%s=values(%s)' % (f, f) for f in field_names])} """
        elif rec_exists == self.EXIST.REPLACE:
            exp = """replace into %s (%s) values (%s) """
        else:
            self.logger.error('未知的记录覆盖模式！')
            raise Exception('未知的记录覆盖模式！')

        self.executemany(exp % (table_name, ','.join(field_names), ','.join(['%s'] * len(field_names))),
                         [tuple(row) for row in data.astype(object).where((pd.notnull(data)), None).values])
        self.logger.info('数据表 %s 写入成功' % table_name)

    def create_database(self, database_name: str, conn: bool = True):
        self._check_permissions()
        # 任何位置都可以创建
        assert database_name not in self.list_databases(), '数据库已存在'
        try:
            self.cr.execute('create database %s' % database_name)
            self.db.commit()
            self.logger.info('数据库创建成功')
            self.active_database = database_name
        except:
            raise Exception('数据库创建失败')

        if conn:
            self.start_conn(database_name)
            self.logger.info('数据库连接成功')
        return 0

    def delete_database(self, database_name: str):
        self._check_permissions()
        # 任何位置都可以删除
        assert database_name in self.list_databases(), '数据库不存在'
        if database_name == self.active_database:
            self.logger.warning('数据库正在使用中, 尝试退出数据库, 重新连接至sql')
            self.connect2sql()
        try:
            self.cr.execute('drop database %s' % database_name)
            self.db.commit()
            self.logger.info('数据库删除成功')
        except:
            raise Exception('数据库删除失败')
        return 0

    def create_table(self, table_config: Dict[Text, Any], set_active_table: bool = False):
        self._check_permissions()
        self._check_db_status()
        table_name = table_config.get('table_name', None) or self.active_table
        assert table_name not in self.list_tables(), '数据表已存在'
        exp_prefix = 'create table %s (' % table_name
        assert 'field_configs' in table_config, 'field_configs未指定'
        exp_m = self._fieldconfigs2sqlexp(table_config['field_configs'])
        field_names = [field_config['field_name'] for field_config in table_config['field_configs']]
        assert set(table_config.get('keys', [])) <= set(field_names), 'keys不存在于field_configs'
        exp_suffix = ',primary key (%s) );' % ', '.join(table_config.get('keys', []))
        exp = '%s %s %s' % (exp_prefix, exp_m, exp_suffix if table_config.get('keys', None) else exp_suffix[-2:])

        try:
            self.execute(exp)
            self.logger.info('数据表 %s 创建成功' % table_name)
            return 0
        except:
            self.logger.error('数据表创建失败')
        self.set_active_table(table_name) if set_active_table else None

    def create_table_like(self, new_table_name: Optional[str], data: Optional[pd.DataFrame] = None, table_name: Optional[str] = None):
        self._check_permissions()
        if data is not None:
            table_config = self._get_detail_config(data, new_table_name)
            return self.create_table(table_config)
        table_name = table_name or self.active_table
        assert table_name and (table_name in self.list_tables()), '数据表不存在'
        self.execute('create table %s like %s' % (new_table_name, table_name))
        self.refresh()

    def delete_table(self, table_name: str):
        self._check_permissions()
        self._check_db_status()
        assert table_name in self.list_tables(), '数据表不存在'
        self.logger.warning('当前活跃数据表 %s 即将被删除！' % table_name) if self.active_table == table_name else None
        try:
            self.cr.execute('drop table if exists %s ' % table_name)
            self.db.commit()
            self.active_table = None if self.active_table == table_name else self.active_table
            self.logger.info('数据表 %s 删除成功' % table_name)
        except:
            raise Exception('数据表删除失败')
        return 0

    def clear_table(self, table_name: str):
        self._check_permissions()
        self.execute('truncate table %s' % table_name)

    def add_field_names(self, table_config: Dict[Text, Any]):
        self._check_permissions()
        self._check_db_status()
        table_name = table_config.get('table_name', None) or self.active_table
        assert table_name and (table_name in self.list_tables()), '数据表未指定 或 数据表不存在'
        exp_prefix = 'alter table %s add column (' % table_name
        assert 'field_configs' in table_config and isinstance(table_config['field_configs'], list), 'field_configs未指定或以非list形式传入'
        _fields_remain, _ = self._check_duplicate_fields(table_config)
        if len(_fields_remain) > 0:
            _field_config = [f for f in table_config['field_configs'] if f['field_name'] in _fields_remain]
            exp_m = self._fieldconfigs2sqlexp(_field_config)
            _filed_names_exp = '%s %s %s' % (exp_prefix, exp_m, ')')
            self.execute(_filed_names_exp)
            self.logger.info('数据表字段添加成功')

    def delete_field(self, field_name: str, table_name: Optional[str] = None):
        self._check_permissions()
        table_name = self._check_tab_status(table_name)
        assert field_name in self.list_fields(table_name), '字段%s不存在' % field_name
        try:
            self.execute('ALTER TABLE %s  DROP %s' % (table_name, field_name))
            self.logger.info('表 %s 中字段 %s 删除成功' % (table_name, field_name))
            return 0
        except:
            self.logger.error('表 %s 中字段 %s 删除失败' % (table_name, field_name))
            return -1

    @func_timer
    def delete_records_by_fields(self,
                                 table_name: Optional[str] = None,
                                 date_range: Union[str, list, Tuple] = None,
                                 other_fields: Optional[Dict[str, Union[str, list, Tuple]]] = None,
                                 other_cond: Optional[str] = None) -> pd.DataFrame:
        table_name = self._check_tab_status(table_name)
        exp = 'DELETE FROM %s WHERE' % (table_name)
        cond = self.getwhere(table_name, date_range, other_fields, other_cond)
        return self.execute(exp + cond) if cond != '' else None

    def clear_temp(self):
        self._check_permissions()
        self.delete_table(self.TAB_TEMP) if self.TAB_TEMP in self.list_tables() else None


if __name__ == '__main__':
    database = 'test'
    sql = WriteSQL()
    sql.start_conn(database)
    sql.list_databases()
    sql.set_active_table('table_test')
    # sql.create_database(database_name)
    sql.list_keys(detail=True)
    sql.list_tables()
    sql.list_all_tables()
    sql.preview_table()
    sql.clear_temp()
    sql.clear_table('table_test')
    sql.list_fields()
    sql.delete_field('WVAD5')
    sql.refresh()
    sql.read()

    index = pd.MultiIndex.from_product(
        [pd.to_datetime(['2020-01-23', '2020-01-25', '2020-01-20'], format='%Y-%m-%d').date, ['000002.SZ', '000005.SZ']],
        names=['tradedate', 'wind_code'])
    data = pd.DataFrame(999, index=index, columns=['WVAD8', 'WVAD5', 'WVAD6'])
    sql.add_fields(data)
    sql.add_records(data)
    sql.write(data, 'table_test')
    pprint(data)
