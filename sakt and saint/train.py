import os
import argparse
import json
import pickle
import pandas as pd

import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam
from sklearn.model_selection import GroupKFold
from processingData import processingData

from utils import match_seq_len, collate_fn, reset_weights
from model_sakt import SAKT
from model_saint import SAINT

def cv_train(model_name, config, for_what = "train"):
    if not os.path.isdir("ckpts"):
        os.mkdir("ckpts")

    ckpt_path = os.path.join("ckpts", model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    model_config = config[model_name]
    train_config = config["train_config"]

    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    optimizer = train_config["optimizer"]  # can be [sgd, adam]
    seq_len = train_config["seq_len"]
    
       
    dataset = processingData(seq_len, for_what = for_what)
    
    gkFold = GroupKFold(5)
    groups = []
    for sidpid in dataset.u_list:
        sid = sidpid.split("pid")[0]
        groups.append(sid)

        
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"  
    aucs_list = []
    for fold, (train_index, test_index) in enumerate(gkFold.split(X = dataset.u_list, groups = groups)):
        print('*******************************************')
        print('*******************************************')
        print(f'FOLD {fold}')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)

        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(
                            dataset,
                            batch_size=batch_size,  collate_fn=collate_fn, 
                            sampler=train_subsampler, 
                            generator=torch.Generator(device=device))
        test_loader = DataLoader(
                            dataset,
                            batch_size=len(test_index),  collate_fn=collate_fn, 
                            sampler=test_subsampler,
                            generator=torch.Generator(device=device))   

        # Initiliaze the model
        if model_name == "sakt":
            model = SAKT(dataset.num_q, **model_config).to(device)
        elif model_name == "saint":
            model = SAINT(**model_config).to(device)
        else:
            print("The wrong model name was used...")
            # return

        ## Reset the parameters to avoid weight leakage 
        model.apply(reset_weights)

        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)

        aucs, loss_means = \
            model.train_model(
                train_loader, test_loader, num_epochs, opt, ckpt_path, for_what = for_what, do_test = True
            )
        aucs_list.append(aucs)

    return aucs_list

def train(model_name, config, for_what = "train"):
    if not os.path.isdir("ckpts"):
        os.mkdir("ckpts")

    ckpt_path = os.path.join("ckpts", model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)


    model_config = config[model_name]
    train_config = config["train_config"]

    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    optimizer = train_config["optimizer"]  # can be [sgd, adam]
    seq_len = train_config["seq_len"]    
       
    dataset = processingData(seq_len, for_what = for_what)
           
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"  
        
    train_loader = DataLoader(
                        dataset,
                        batch_size=batch_size,  collate_fn=collate_fn, 
                        shuffle=True, 
                        generator=torch.Generator(device=device))
    test_loader = DataLoader(
                        dataset,
                        batch_size=batch_size,  collate_fn=collate_fn, 
                        shuffle=False,
                        generator=torch.Generator(device=device))   
    # Initiliaze the model
    #model = SAKT(dataset.num_q, **model_config).to(device)
    if model_name == "sakt":
        model = SAKT(dataset.num_q, **model_config).to(device)
    elif model_name == "saint":
        model = SAINT(**model_config).to(device)
    else:
        print("The wrong model name was used...")
        # return

    ## Reset the parameters to avoid weight leakage 
    model.apply(reset_weights)

    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)

    loss_means = \
        model.train_model(
            train_loader, test_loader, num_epochs, opt, ckpt_path, for_what = for_what, do_test=False
        )