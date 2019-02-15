#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : graph.py
# Create date : 2019-01-30 14:25
# Modified date : 2019-02-15 19:42
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.optim as optim

from stn_model import Net
from base_graph import BaseGraph

def _get_criterion():
    #return nn.CrossEntropyLoss()
    return nn.NLLLoss()

def _get_SGD_optimizer(model, config):
    learn_rate = config["learn_rate"]
    optimizer = optim.SGD(model.parameters(), lr=learn_rate)
    return optimizer

def _get_momentum_optimizer(model, config):
    learn_rate = config["learn_rate"]
    momentum = config["momentum"]
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum)
    return optimizer

def _get_model(config):
    model = Net().to(config["device"])
    return model

class SpatialTransferGraph(BaseGraph):
    def __init__(self, data_dict, config):
        super(SpatialTransferGraph, self).__init__(data_dict, config)

    def _init_graph_dict(self, config):
        graph_dict = {}
        graph_dict["model"] = _get_model(config)
        if config["loss"] == "NLL":
            graph_dict["criterion"] = _get_criterion()

        if config["optimizer"] == "momentum":
            graph_dict["optimizer"] = _get_momentum_optimizer(graph_dict["model"], config)

        if config["optimizer"] == "SGD":
            graph_dict["optimizer"] = _get_SGD_optimizer(graph_dict["model"], config)


        return graph_dict
