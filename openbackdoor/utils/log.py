# -*- coding: utf-8 -*-
import logging
import os
import datetime
from typing import *

def init_logger(
    log_file: Optional[str] = None,
    log_file_level=logging.NOTSET,
    log_level=logging.INFO,
):  
    if isinstance(log_file_level, str):
        log_file_level = getattr(logging, log_file_level)
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level)
    log_format = logging.Formatter("[\033[032m%(asctime)s\033[0m %(levelname)s] %(module)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

logger = init_logger()
