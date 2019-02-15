#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : base_graph.py
# Create date : 2019-02-11 17:54
# Modified date : 2019-02-14 15:40
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os
from abc import abstractmethod

import torch

import record
import status
import show

class BaseGraph(object):
    def __init__(self, data_dict, config):
        super(BaseGraph, self).__init__()
        self.early_stop = False
        self.config = config
        self.data_dict = data_dict
        self.graph_dict = self._init_graph_dict(config)
        self.status_dict = status.get_status_dict()
        self._load_train_model()

    def _get_progress_str(self, batch_idx, data, train_loader):
        # pylint: disable=bad-continuation
        progress_str = ' {}/{} ({:.0f}%)'.format(
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader)
                            )
        # pylint: enable=bad-continuation
        return progress_str

    def _get_accuracy_str(self, correct, test_loader):
        accuracy_str = '{}/{} ({:.0f}%)'.format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset))
        return accuracy_str

    def _check_early_stop(self, stop_category):
        if self.early_stop == True:
            return True
        config = self.config

        if config["early_stop_%s" % stop_category] == True:
            step = self.status_dict[stop_category]
            best_step = self.status_dict["best_%s"% stop_category]
            if step - best_step > config["early_stop_%s_limit" % stop_category]:
                record.save_content(config, "Early Stop With %s: %s" % (stop_category, step))
                self.early_stop = True
                return True

        return False

    def _load_model_dict(self, checkpoint):
        self.graph_dict["model"].load_state_dict(checkpoint["model"])
        self.graph_dict["criterion"].load_state_dict(checkpoint["criterion"])
        self.graph_dict["optimizer"].load_state_dict(checkpoint["optimizer"])

        self.status_dict = checkpoint["status_dict"]
        self.config = checkpoint["config"]

    def _save_trained_model(self):
        model_dict = self._get_model_dict()
        file_full_path = record.get_check_point_file_full_path(self.config)
        torch.save(model_dict, file_full_path)

    def _get_model_dict(self):
        model_dict = {}
        model_dict["model"] = self.graph_dict["model"].state_dict()
        model_dict["criterion"] = self.graph_dict["criterion"].state_dict()
        model_dict["optimizer"] = self.graph_dict["optimizer"].state_dict()

        model_dict["status_dict"] = self.status_dict
        model_dict["config"] = self.config
        return model_dict

    def _create_output(self):
        if self.config["train_load_check_point_file"] == False:
            record.create_output(self.config)

        if not record.check_output_file(self.config):
            model = self.graph_dict["model"]
            record.save_content(self.config, model)
            record.record_dict(self.config, self.config)

    def _load_train_model(self, mode="train"):
        file_full_path = record.get_check_point_file_full_path(self.config)
        if os.path.exists(file_full_path):
            if mode == "train" and self.config["train_load_check_point_file"]:
                checkpoint = torch.load(file_full_path)
                self._load_model_dict(checkpoint)

            if mode == "test":
                checkpoint = torch.load(file_full_path)
                self._load_model_dict(checkpoint)

    def _record_config(self):
        record.record_dict(self.config, self.config)

    def show_the_value(self):
        show.show_epoch_loss(self.status_dict, self.config)
        show.show_step_loss(self.status_dict, self.config)
        show.show_epoch_acc(self.status_dict, self.config)
        show.show_step_acc(self.status_dict, self.config)

    def check_epoch_early_stop(self):
        return self._check_early_stop("epoch")

    def check_step_early_stop(self):
        return self._check_early_stop("step")

    @abstractmethod
    def _init_graph_dict(self, config):
        '''Implement in subclass'''
