#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : record.py
# Create date : 2019-01-30 21:37
# Modified date : 2019-02-14 15:35
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os

def _get_param_str(config):
    # pylint: disable=bad-continuation
    param_str = "%s_%s_%s_%s_%s_%s" % (
                                config["dataset"],
                                config["loss"],
                                config["optimizer"],
                                config["epochs"],
                                config["batch_size"],
                                config["learn_rate"],
                                )
    # pylint: enable=bad-continuation
    return param_str

def get_check_point_path(config):
    param_str = _get_param_str(config)
    directory = "%s/save/%s/" % (config["data_path"], param_str)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_check_point_file_full_path(config):
    path = get_check_point_path(config)
    param_str = _get_param_str(config)
    file_full_path = "%s%scheckpoint.tar" % (path, param_str)
    return file_full_path

def _write_output(config, con, tp="a"):
    save_path = get_check_point_path(config)
    file_full_path = "%s/output" % save_path
    f = open(file_full_path, tp)
    f.write("%s\n" %  con)
    f.close()

def check_output_file(config):
    save_path = get_check_point_path(config)
    file_full_path = "%s/output" % save_path
    return os.path.exists(file_full_path)

def create_output(config, con="", tp="w"):
    save_path = get_check_point_path(config)
    file_full_path = "%s/output" % save_path
    f = open(file_full_path, tp)
    f.write("%s" %  con)
    f.close()

def record_dict(config, dic):
    save_content(config, "config:")
    for key in dic:
        dic_str = "%s : %s" % (key, dic[key])
        save_content(config, dic_str)

def save_content(config, con):
    print(con)
    _write_output(config, con)
