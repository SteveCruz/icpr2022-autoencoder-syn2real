##############################################################################################################################################################
##############################################################################################################################################################
"""
Repeat training of autoencoder model for different seeds.

Replace or modify the config file in the following part of the code to make changes to train different models.

# load the config file
config = toml.load("cfg/extractor_ae.toml")
"""
#####################################################################################################################################################
#####################################################################################################################################################

import toml
import torch
import random
import numpy as np

from train_ae import train

#####################################################################################################################################################
#####################################################################################################################################################

if __name__ == '__main__':

    # load the config file
    # config = toml.load("cfg/conv_ae.toml")
    config = toml.load("cfg/extractor_ae.toml")

    # define some fair seeds    
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # for each seed defined above
    for seed in seeds:

        print("-"*77)
        print("Running experiment for seed: ", seed)
        print("-"*77)

        # reproducibility
        # set the seed to the current seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

        # run the training
        train(config)

#####################################################################################################################################################