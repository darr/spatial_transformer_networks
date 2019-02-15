#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : show.py
# Create date : 2019-02-11 14:37
# Modified date : 2019-02-13 20:06
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import record

def _get_list(category_name, key_name, list_name, status_dict):
    l = status_dict[list_name]
    x = []
    y = []
    for dic in l:
        y_value = dic[key_name]
        y.append(y_value)
        x_value = dic[category_name]
        x.append(x_value)
    return y, x

def _show_list(category_name, data_key, status_dict, config):
    name = "%s_%s.jpg" % (category_name, data_key)
    train_list_name = "train_%s_%s" % (category_name, data_key)
    eval_list_name = "eval_%s_%s" % (category_name, data_key)

    train_lt, train_step = _get_list(category_name, data_key, "%s_list" % train_list_name, status_dict)
    eval_lt, eval_step = _get_list(category_name, data_key, "%s_list" % eval_list_name, status_dict)

    l1, = plt.plot(train_step, train_lt)
    l2, = plt.plot(eval_step, eval_lt)

    line_lt = [l1, l2]
    labels_lt = (train_list_name, eval_list_name)
    _write_jpg(line_lt, labels_lt, name, config)

def _write_jpg(line_lt, labels_lt, name, config):
    plt.legend(handles=line_lt, labels=labels_lt, loc='best')
    save_path = record.get_check_point_path(config)
    full_path_name = "%s/%s" % (save_path, name)
    plt.savefig(full_path_name)
    plt.close()

def show_epoch_loss(status_dict, config):
    category_name = "epoch"
    data_key = "loss"
    _show_list(category_name, data_key, status_dict, config)

def show_step_loss(status_dict, config):
    category_name = "step"
    data_key = "loss"
    _show_list(category_name, data_key, status_dict, config)

def show_epoch_acc(status_dict, config):
    category_name = "epoch"
    data_key = "acc"
    _show_list(category_name, data_key, status_dict, config)

def show_step_acc(status_dict, config):
    category_name = "step"
    data_key = "acc"
    _show_list(category_name, data_key, status_dict, config)

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize_stn(data_dict, model, config):
    test_loader = data_dict["test_loader"]
    device = config["device"]
    with torch.no_grad():
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor))

        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

    save_path = record.get_check_point_path(config)
    name = "compare.jpg"
    full_path_name = "%s/%s" % (save_path, name)
    plt.savefig(full_path_name)
    #plt.show()
