import logging
import time
import logging.config


def init_logger(name: str = str(int(round(time.time())))):
    dict_log_config = {
        "version": 1,
        "handlers": {
            "fileHandler": {
                "class": "logging.FileHandler",
                "formatter": "myFormatter",
                "filename": "./log/" + name
            },
            "stdHandler": {
                "class": "logging.StreamHandler",
                "formatter": "myFormatter",
                "stream": "ext://sys.stderr"
            }
        },
        "loggers": {
            "thesis": {
                "handlers": ["fileHandler", "stdHandler"],
                "level": "INFO",
            }
        },
        "formatters": {
            "myFormatter": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    }
    logging.config.dictConfig(dict_log_config)
    return logging.getLogger('thesis')


def get_logger(name: str = None):
    if name is None:
        return init_logger()
    return logging.getLogger("thesis." + name)
