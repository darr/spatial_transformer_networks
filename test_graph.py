#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : test_graph.py
# Create date : 2019-02-01 17:21
# Modified date : 2019-02-13 21:23
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import record
from graph import SpatialTransferGraph
import status

def _print_status(test_loss, correct, test_loader, config):
    print_str = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset))
    record.save_content(config, print_str)

def _eval_a_step(data, target, model, criterion, config):
    device = config["device"]
    data, target = data.to(device), target.to(device)
    output = model(data)
    l = F.nll_loss(output, target, size_average=False).item()
#    l = criterion(output, target).item()
    pred = output.max(1, keepdim=True)[1]
    c = pred.eq(target.view_as(pred)).sum().item()
    return l, c

class TestSpatialTransferGraph(SpatialTransferGraph):
    def __init__(self, data_dict, config):
        super(TestSpatialTransferGraph, self).__init__(data_dict, config)
        self._load_train_model("test")

    def _get_eval_value(self, model, test_loader, criterion):
        config = self.config
        with torch.no_grad():
            model.eval()
            test_loss = 0
            test_correct = 0
            step = 0
            data_counts = 0
            for data, target in test_loader:
                l, c = _eval_a_step(data, target, model, criterion, config)
                test_loss += l
                test_correct += c
                step += 1
                data_counts += len(data)

            test_loss /= len(test_loader.dataset)
            test_acc = test_correct / data_counts
            return test_loss, test_acc, test_correct

    def _eval_a_epoch(self, model, criterion, status_dict):
        test_loader = self.data_dict["test_loader"]
        test_loss, test_acc, test_correct = self._get_eval_value(model, test_loader, criterion)
        accuracy_str = self._get_accuracy_str(test_correct, test_loader)
        status.update_acc_str(accuracy_str, status_dict)
        status.val_step_update_status_dict(test_loss, test_acc, model, status_dict)
        return test_loss, test_acc

    def _test_a_epoch(self, model, criterion):
        test_loader = self.data_dict["test_loader"]
        config = self.config
        test_loss, test_acc, test_correct = self._get_eval_value(model, test_loader, criterion)
        _print_status(test_loss, test_correct, test_loader, config)
        return test_loss, test_acc

    def _test_best_step_model(self):
        model = self.graph_dict["model"]
        model.load_state_dict(self.status_dict["best_step_model_wts"])
        criterion = self.graph_dict["criterion"]
        self._test_a_epoch(model, criterion)

    def _test_best_epoch_model(self):
        model = self.graph_dict["model"]
        model.load_state_dict(self.status_dict["best_epoch_model_wts"])
        criterion = self.graph_dict["criterion"]
        self._test_a_epoch(model, criterion)

    def eval_the_model(self, model, criterion, status_dict):
        eval_loss, eval_acc = self._eval_a_epoch(model, criterion, status_dict)
        status.val_epoch_update_status_dict(eval_loss, eval_acc, model, status_dict)

    def test_the_model(self):
        self._test_best_step_model()
        self._test_best_epoch_model()
