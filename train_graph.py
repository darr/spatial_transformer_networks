#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : train_graph.py
# Create date : 2019-02-01 17:22
# Modified date : 2019-02-15 19:34
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import copy
import time

import test_graph
#import record
import status
from graph import SpatialTransferGraph

class TrainSpatialTransferGraph(SpatialTransferGraph):
    def __init__(self, data_dict, config):
        super(TrainSpatialTransferGraph, self).__init__(data_dict, config)

    def _train_a_step(self, inputs, labels,):
        model = self.graph_dict["model"]
        criterion = self.graph_dict["criterion"]
        optimizer = self.graph_dict["optimizer"]
        device = self.config["device"]

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        pred = output.max(1, keepdim=True)[1]
        corrects = pred.eq(labels.view_as(pred)).sum().item()

        return loss, corrects

    def _train_a_epoch(self):
        train_loader = self.data_dict["train_loader"]
        model = self.graph_dict["model"]
        start_step = self.status_dict["step"]
        config = self.config

        running_loss = 0.0
        running_corrects = 0.0

        step = start_step
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            loss, corrects = self._train_a_step(data, target)
            step += 1

            if step % config["print_every"] == 0:
                status.update_step(step, self.status_dict)
                status.train_step_update_status_dict(loss, corrects / config["batch_size"], self.status_dict)

                progress_str = self._get_progress_str(batch_idx, data, train_loader)
                status.update_progress_str(progress_str, self.status_dict)

                self._eval()
                status.save_step_status(self.status_dict, self.config)
                if self.check_step_early_stop():
                    break
            running_loss += loss
            running_corrects += corrects

        train_epoch_loss = running_loss / (step - start_step)
        train_epoch_acc = running_corrects / ((step - start_step) * config["batch_size"])
        status.train_epoch_update_status_dict(train_epoch_loss, train_epoch_acc, self.status_dict)

    def _run_a_epoch(self, epoch):
        status.update_epoch(epoch, self.status_dict)
        start = time.time()
        self._train_a_epoch()
        self._eval_a_epoch()
        end = time.time()

        status.update_eplapsed_time(start, end, self.status_dict)
        status.save_epoch_status(self.status_dict, self.config)
        self._save_trained_model()

    def _eval_a_epoch(self):
        self._eval()

    def _eval(self):
        eval_graph = test_graph.TestSpatialTransferGraph(self.data_dict, self.config)
        model = self.graph_dict["model"]
        criterion = self.graph_dict["criterion"]
        eval_graph.eval_the_model(model, criterion, self.status_dict)

    def train_the_model(self):
        model = self.graph_dict["model"]
        self._create_output()
        num_epochs = self.config["epochs"]

        self.status_dict["best_model_wts"] = copy.deepcopy(model.state_dict())
        epoch_start = self.status_dict["epoch"]

        for epoch in range(epoch_start + 1, num_epochs + 1):
            if not self.check_epoch_early_stop():
                self._run_a_epoch(epoch)

        self.show_the_value()
        return self.graph_dict["model"]
