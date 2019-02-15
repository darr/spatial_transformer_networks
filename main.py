#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : main.py
# Create date : 2019-01-30 13:35
# Modified date : 2019-02-14 15:03
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from etc import config
import mnist_data_set
import show
import train_graph
import test_graph

def run():
    data_dict = mnist_data_set.get_data_dict(config)
    train_g = train_graph.TrainSpatialTransferGraph(data_dict, config)
    model = train_g.train_the_model()

    show.visualize_stn(data_dict, model, config)
    test_g = test_graph.TestSpatialTransferGraph(data_dict, config)
    test_g.test_the_model()

run()
