#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : status.py
# Create date : 2019-02-01 13:41
# Modified date : 2019-02-14 15:25
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import copy
import record

def get_status_dict():
    status_dict = {}
    status_dict["elapsed_time"] = 0.0
    status_dict["epoch"] = 0
    status_dict["step"] = 0
    status_dict["epoch_eplapsed_time"] = 0.0
    status_dict["step_eplapsed_time"] = 0.0
    status_dict["so_far_elapsed_time"] = 0.0

    status_dict["acc_str"] = ""
    status_dict["progress_str"] = ""


    status_dict["best_epoch"] = 0
    status_dict["best_epoch_acc"] = 0.0
    status_dict["best_epoch_loss"] = None
    status_dict["best_epoch_model_wts"] = None
    status_dict["train_epoch_loss"] = 0.0
    status_dict["train_epoch_acc"] = 0.0
    status_dict["eval_epoch_loss"] = 0.0
    status_dict["eval_epoch_acc"] = 0.0
    status_dict["train_epoch_loss_list"] = []
    status_dict["train_epoch_acc_list"] = []
    status_dict["eval_epoch_loss_list"] = []
    status_dict["eval_epoch_acc_list"] = []


    status_dict["best_step"] = 0
    status_dict["best_step_acc"] = 0
    status_dict["best_step_loss"] = None
    status_dict["best_step_model_wts"] = None
    status_dict["train_step_loss"] = 0.0
    status_dict["train_step_acc"] = 0.0
    status_dict["eval_step_loss"] = 0.0
    status_dict["eval_step_acc"] = 0.0
    status_dict["train_step_loss_list"] = []
    status_dict["train_step_acc_list"] = []
    status_dict["eval_step_loss_list"] = []
    status_dict["eval_step_acc_list"] = []

    return status_dict

def _append_record_list(data, data_key, list_name, cat_name, status_dict):
    dic = {}
    dic[cat_name] = status_dict[cat_name]
    dic[data_key] = data
    status_dict[list_name].append(dic)

def _update_data(category_name, data_key, data, mode, status_dict):
    status_key = "%s_%s_%s" % (mode, category_name, data_key)
    status_dict[status_key] = data
    _append_record_list(data, data_key, "%s_list" % status_key, category_name, status_dict)

def _update_eval_data(category_name, data_key, data, status_dict):
    mode = "eval"
    _update_data(category_name, data_key, data, mode, status_dict)

def _update_step_eval_data(eval_loss, eval_acc, status_dict):
    category_name = "step"
    _update_eval_data(category_name, "loss", eval_loss, status_dict)
    _update_eval_data(category_name, "acc", eval_acc, status_dict)

def _update_epoch_eval_data(eval_loss, eval_acc, status_dict):
    category_name = "epoch"
    _update_eval_data(category_name, "loss", eval_loss, status_dict)
    _update_eval_data(category_name, "acc", eval_acc, status_dict)

def _update_train_data(category_name, data_key, data, status_dict):
    mode = "train"
    _update_data(category_name, data_key, data, mode, status_dict)

def _update_step_train_data(eval_loss, eval_acc, status_dict):
    category_name = "step"
    _update_train_data(category_name, "loss", eval_loss, status_dict)
    _update_train_data(category_name, "acc", eval_acc, status_dict)

def _update_epoch_train_data(eval_loss, eval_acc, status_dict):
    category_name = "epoch"
    _update_train_data(category_name, "loss", eval_loss, status_dict)
    _update_train_data(category_name, "acc", eval_acc, status_dict)

def val_epoch_update_status_dict(loss, acc, model, status_dict):
    _update_epoch_eval_data(loss, acc, status_dict)

    if status_dict["best_epoch_loss"] == None or loss < status_dict["best_epoch_loss"]:
        status_dict["best_epoch"] = status_dict["epoch"]
        status_dict["best_epoch_loss"] = loss
        status_dict["best_epoch_acc"] = acc
        status_dict["best_epoch_model_wts"] = copy.deepcopy(model.state_dict())

def val_step_update_status_dict(loss, acc, model, status_dict):
    _update_step_eval_data(loss, acc, status_dict)

    if status_dict["best_step_loss"] == None or loss < status_dict["best_step_loss"]:
        status_dict["best_step"] = status_dict["step"]
        status_dict["best_step_loss"] = loss
        status_dict["best_step_acc"] = acc
        status_dict["best_step_model_wts"] = copy.deepcopy(model.state_dict())

def train_step_update_status_dict(loss, acc, status_dict):
    _update_step_train_data(loss.item(), acc, status_dict)

def train_epoch_update_status_dict(loss, acc, status_dict):
    _update_epoch_train_data(loss.item(), acc, status_dict)

def update_eplapsed_time(start, end, status_dict):
    status_dict["epoch_eplapsed_time"] = end - start
    status_dict["so_far_elapsed_time"] += status_dict["epoch_eplapsed_time"]

def update_epoch(epoch, status_dict):
    status_dict["epoch"] = epoch

def update_step(step, status_dict):
    status_dict["step"] = step

def update_acc_str(acc_str, status_dict):
    status_dict["acc_str"] = acc_str

def update_progress_str(progress_str, status_dict):
    status_dict["progress_str"] = progress_str

def save_step_status(status_dict, config):
    category_name = "step"
    num_epochs = config["epochs"]
    epoch = status_dict["epoch"]
    step = status_dict["step"]

    train_loss = status_dict["train_%s_loss" % category_name]
    train_acc = status_dict["train_%s_acc" % category_name]

    eval_loss = status_dict["eval_%s_loss" % category_name]
    eval_acc = status_dict["eval_%s_acc" % category_name]

    acc_str = status_dict["acc_str"]
    progress_str = status_dict["progress_str"]

    best_epoch = status_dict["best_epoch"]
    best_epoch_acc = status_dict["best_epoch_acc"]
    best_epoch_loss = status_dict["best_epoch_loss"]

    best_step = status_dict["best_step"]
    best_step_acc = status_dict["best_step_acc"]
    best_step_loss = status_dict["best_step_loss"]

    # pylint: disable=bad-continuation
    save_str = '[%s/%s] [Step:%s] [Train Loss:%.6f Acc:%.6f%s] [Val Loss:%.6f Acc:%.6f %s] [Best Epoch:%s Loss:%.6f Acc:%.6f] [Best Step:%s Loss:%.6f Acc:%.6f]' % (
                            epoch,
                            num_epochs - 1,
                            step,
                            train_loss,
                            train_acc,
                            progress_str,
                            eval_loss,
                            eval_acc,
                            acc_str,
                            best_epoch,
                            best_epoch_loss,
                            best_epoch_acc,
                            best_step,
                            best_step_loss,
                            best_step_acc,
                            )

    # pylint: enable=bad-continuation
    record.save_content(config, save_str)

def save_epoch_status(status_dict, config):
    num_epochs = config["epochs"]
    epoch = status_dict["epoch"]
    step = status_dict["step"]

    train_loss = status_dict["train_epoch_loss"]
    train_acc = status_dict["train_epoch_acc"]

    eval_loss = status_dict["eval_epoch_loss"]
    eval_acc = status_dict["eval_epoch_acc"]

    acc_str = status_dict["acc_str"]

    best_epoch = status_dict["best_epoch"]
    best_epoch_acc = status_dict["best_epoch_acc"]
    best_epoch_loss = status_dict["best_epoch_loss"]

    best_step = status_dict["best_step"]
    best_step_acc = status_dict["best_step_acc"]
    best_step_loss = status_dict["best_step_loss"]

    epoch_elapsed_time = status_dict["epoch_eplapsed_time"]
    so_far_elapsed_time = status_dict["so_far_elapsed_time"]

    # pylint: disable=bad-continuation
    save_str = '[%s/%s] [Step:%s] [Train Loss:%.6f Acc:%.6f] [Val Loss:%.6f Acc:%.6f %s] [Best Epoch:%s Loss:%.6f Acc:%.6f] [Best Step:%s Loss:%.6f Acc:%.6f] [%.2fs %.1fs]' % (
                            epoch,
                            num_epochs - 1,
                            step,
                            train_loss,
                            train_acc,
                            eval_loss,
                            eval_acc,
                            acc_str,
                            best_epoch,
                            best_epoch_loss,
                            best_epoch_acc,
                            best_step,
                            best_step_loss,
                            best_step_acc,
                            epoch_elapsed_time,
                            so_far_elapsed_time,
                            )

    # pylint: enable=bad-continuation
    record.save_content(config, save_str)
