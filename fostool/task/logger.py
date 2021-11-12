# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding:utf-8
from __future__ import absolute_import, division, print_function
import logging
import os,sys

class PackagePathFilter(logging.Filter):
    def filter(self, record):
        """add relative path to record
        """
        pathname = record.pathname
        record.relativepath = None
        abs_sys_paths = map(os.path.abspath, sys.path)
        for path in sorted(abs_sys_paths, key=len, reverse=True):  # longer paths first
            if not path.endswith(os.sep):
                path += os.sep
            if pathname.startswith(path):
                record.relativepath = os.path.relpath(pathname, path)
                break
        return True

def setLogger():
    """set logger formatter and level
    """
    LOG_FORMAT = "%(asctime)s %(relativepath)s %(lineno)s \ - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    logger=logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addFilter(PackagePathFilter())
    formatter=logging.Formatter(LOG_FORMAT,DATE_FORMAT)
    logger.setFormatter(formatter)
    
    logging.getLogger().addHandler(logger)
