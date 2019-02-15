#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : etc.py
# Create date : 2019-01-30 15:17
# Modified date : 2019-02-15 19:32
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import torch

config = {}
#base
config["epoch_only"] = True
config["early_stop_epoch"] = True
config["early_stop_epoch_limit"] = 5
config["early_stop_step"] = True
config["early_stop_step_limit"] = 2000

config["train_load_check_point_file"] = True
config["batch_size"] = 128
config["print_every"] = 100
config["num_workers"] = 4
config["epochs"] = 100
config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config["loss"] = "NLL"
config["optimizer"] = "SGD"
#config["optimizer"] = "momentum"
#base

config["dataset"] = "mnist"
config["data_path"] = "./raw"
#   config["data_path"] = "./data/%s" % config["dataset"]
config["learn_rate"] = 0.01
config["momentum"] = 0.9
