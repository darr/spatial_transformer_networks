#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : mnist_data_set.py
# Create date : 2019-02-10 20:38
# Modified date : 2019-02-13 20:21
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import torch
from torchvision import datasets, transforms

def _get_transforms():
    # pylint: disable=bad-continuation
    trans = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                            ])
    # pylint: enable=bad-continuation
    return trans

def _get_data_loader(train, config):
    trans = _get_transforms()
    # pylint: disable=bad-continuation
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=train, download=True, transform=trans),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
        )
    # pylint: enable=bad-continuation
    return train_loader

def _get_train_dataloader(config):
    return _get_data_loader(True, config)

def _get_test_dataloader(config):
    return _get_data_loader(False, config)

def _get_dataloader(config):
    train_loader = _get_train_dataloader(config)
    test_loader = _get_test_dataloader(config)
    return train_loader, test_loader

def get_data_dict(config):
    train_loader, test_loader = _get_dataloader(config)
    data_dict = {}
    data_dict["train_loader"] = train_loader
    data_dict["test_loader"] = test_loader
    return data_dict
