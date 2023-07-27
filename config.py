from __future__ import annotations
import copy
import logging


class Config:
    def __init__(self, default_conf: dict):
        self.__dict__["_default_config"] = copy.deepcopy(default_conf)  # avoiding conflicts with __getattr__
        self.reset()

    def __getitem__(self, key: str):
        return self.__dict__["_config"][key]

    def __getattr__(self, attr: str):
        if attr in self.__dict__["_config"]:
            return self.__dict__["_config"][attr]

        raise AttributeError(f"No such `{attr}` in self._config")

    def get(self, key, default=None):
        return self.__dict__["_config"].get(key, default)

    def __setitem__(self, key, value):
        self.__dict__["_config"][key] = value

    def __setattr__(self, attr, value):
        self.__dict__["_config"][attr] = value

    def __contains__(self, item):
        return item in self.__dict__["_config"]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return str(self.__dict__["_config"])

    def __repr__(self):
        return str(self.__dict__["_config"])

    def reset(self):
        self.__dict__["_config"] = copy.deepcopy(self._default_config)

    def update(self, *args, **kwargs):
        self.__dict__["_config"].update(*args, **kwargs)

    def set_conf_from_C(self, config_c):
        self.update(**config_c.__dict__["_config"])

    def register_from_C(self, config, skip_register=True):
        from logger import set_log_with_config  # pylint: disable=C0415

        if C.registered and skip_register:
            return

        C.set_conf_from_C(config)
        if C.logging_config:
            set_log_with_config(C.logging_config)
        C.register()


# pickle.dump protocol version: https://docs.python.org/3/library/pickle.html#data-stream-format
PROTOCOL_VERSION = 4

BM = {
    'sz50': '000016.SH',
    'hs300': '000300.SH',
    'zz1000': '000852.SH',
    'zzlt': '000902.CSI',
    'zz500': '000905.SH',
    'zz800': '000906.SH',
    'zzqz': '000985.CSI'
}

_default_config = {
    'date_range': {
        'start': '2017-01-01',
        'end': '2022-06-30'
    },
    "connect": dict(host='10.224.16.81', user='haquant', passwd='haquant', database='jydb', port=3306, charset='utf8'),
    'cne5': ['Beta', 'BooktoPrice', 'EarningsYield', 'Growth', 'Leverage', 'Liquidity', 'Momentum', 'NonLinearSize', 'ResidualVolatility', 'Size'],
    'BM': BM,
    'logging_config': {
        "version": 1,
        "formatters": {
            "default": {
                'format': "[%(process)s:%(threadName)s](%(asctime)s) %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": 20,
                "filename": "./log.txt",
                "formatter": "default",
            }
        },
        "loggers": {
            "ba": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            }
        },
        "disable_existing_loggers": True,
    },
    "dump_protocol_version": PROTOCOL_VERSION,
}


class BaConfig(Config):
    def __init__(self, default_conf):
        super().__init__(default_conf)
        self._registered = False

    def set_connect(self, connect):
        self.update(connect)

    def set(self, **kwargs):
        from logger import set_log_with_config, get_module_logger  # pylint: disable=C0415

        self.reset()
        _logging_config = kwargs.get("logging_config", self.logging_config)

        # set global config
        if _logging_config:
            set_log_with_config(_logging_config)

        # FIXME: this logger ignored the level in config
        logger = get_module_logger("Initialization", level=logging.INFO)
        logger.info(f"重设以下参数: {kwargs}.")

        for k, v in kwargs.items():
            if k not in self:
                logger.warning("Unrecognized config %s" % k)
            self[k] = v

    def register(self):
        """
        这里可以做很多初始化的操作
        """
        self._registered = True

    @property
    def registered(self):
        return self._registered


# global config
C = BaConfig(_default_config)
