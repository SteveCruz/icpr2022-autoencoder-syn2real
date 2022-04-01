##############################################################################################################################################################
##############################################################################################################################################################
"""

Compute the error between the reconstruction of the real images (unseen) 
and the corresponding synthetic training images (toy or realistic).

The metric can be switched between L1, SSIM and LPIPS.

# list of all experiments to check
experiments = [


]
"""
##############################################################################################################################################################
##############################################################################################################################################################

import os
import sys
import copy
import toml
import torch
import numpy as np
from pathlib import Path
from importlib import import_module

import torchvision 
from torch import nn

import utils
import dataset

from pytorch_msssim import SSIM
import lpips

##############################################################################################################################################################
##############################################################################################################################################################

def get_folder(experiment):

    folders = dict()
    folders["experiment"] = Path("results") / experiment
    folders["scripts"] = folders["experiment"] / "scripts"
    folders["logs"] = folders["experiment"] / "logs"
    folders["latent"] = folders["experiment"] / "data" / "latent"
    folders["checkpoint"] = folders["experiment"] / "checkpoints" 
    folders["images"] = folders["experiment"] / "images" / "eval"
    folders["config"] = folders["experiment"] / "cfg.toml"

    # check if the folder exists
    if not folders["experiment"].is_dir():
        print("The specified experiment does not exist: Exit script.")
        sys.exit()

    folders["images"].mkdir(parents=True, exist_ok=True)
    folders["latent"].mkdir(parents=True, exist_ok=True)

    # save the console output to a file
    sys.stdout = utils.Tee(original_stdout=sys.stdout, file=folders["logs"] / "testing.log")

    return folders

##############################################################################################################################################################        

def get_setup(config, folders, load_model=False):

    if config["model"]["type"] == "classification":
        raise ValueError("Your config is for a classification model, but this script is for autoencoders. Please use eval_classifier.py instead.")

    # load data
    train_loader = dataset.create_dataset(
        which_dataset=config["dataset"]["name"], 
        which_factor=config["dataset"]["factor"], 
        use_triplet=False,
        should_augment=False,
        make_scene_impossible=False,
        make_instance_impossible=False,
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        get_all=False,
    )

    if load_model:

        # get path to the loader script
        model_file = folders["scripts"] / "model"

        # replace the / by .
        string_model_file = str(model_file).replace("/",".")

        # load the model from the file
        model_file = import_module(string_model_file)

        # get the model definition
        model_def = getattr(model_file, "create_ae_model")

        # define model
        model = model_def(config["model"], config["dataset"], config["training"]).to(config["device"])
        model.print_model()

        # load the model weights
        checkpoint = torch.load(folders["checkpoint"] / "last_model.pth", map_location=config["device"])

        # apply the model weights
        model.load_state_dict(checkpoint) 

        return train_loader, model

    return train_loader

##############################################################################################################################################################

def compare_recon_to_train(model, config, train_loader, eval_loader, which_metric):

    # make sure we are in eval mode
    model.eval()

    # criterion
    if which_metric == "ssim":
        criterion = SSIM(data_range=1.0, size_average=False, channel=3)

    elif which_metric == "l1":
        criterion = nn.L1Loss(reduction="none")

    elif which_metric == "lpips":
        # https://github.com/richzhang/PerceptualSimilarity#1-learned-perceptual-image-patch-similarity-lpips-metric
        criterion = lpips.LPIPS(net='alex').to(config["device"])
        # we need to normalize the images such that they are in [-1, 1] and not [0, 1]
        preprocess = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # keep tracko of al errors
    all_errors = []

    # we do not need to keep track of gradients
    with torch.no_grad():

        # for each batch of images
        for batch_train, batch_eval in zip(train_loader, eval_loader):

            # push to gpu
            train_images = batch_train["image"].to(config["device"])
            eval_images = batch_eval["image"].to(config["device"])

            # get the reconstruction
            eval_recon = model(eval_images)["xhat"]

            # compute the error between reconstruction of the eval images
            # and the corresponding eval image, i.e. same scene but once as
            # real and once as toy or realistic rendering

            if which_metric == "ssim":
                error = criterion(eval_recon, train_images)

            elif which_metric == "l1":
                error = criterion(eval_recon, train_images).sum(axis=[1,2,3])

            elif which_metric == "lpips":
                # make sure both images are in [-1, 1]
                eval_recon = preprocess(eval_recon)
                train_images = preprocess(train_images)
                error = criterion.forward(eval_recon, train_images)

            # keep track of the errors and remove all unncessary dimensions
            all_errors.extend(error.cpu().numpy().squeeze())

    # make sure its a numpy array
    all_errors = np.array(all_errors)

    # make sure the number of collected samples matches the dataset size
    assert all_errors.shape[0] == len(train_loader.dataset)

    # compute the mean over the dataset
    return all_errors.mean()

##############################################################################################################################################################

def compare_recon_to_itself(model, config, eval_loader, which_metric):

    # make sure we are in eval mode
    model.eval()

    # criterion
    if which_metric == "ssim":
        criterion = SSIM(data_range=1.0, size_average=False, channel=3)

    elif which_metric == "l1":
        criterion = nn.L1Loss(reduction="none")

    elif which_metric == "lpips":
        # https://github.com/richzhang/PerceptualSimilarity#1-learned-perceptual-image-patch-similarity-lpips-metric
        criterion = lpips.LPIPS(net='alex').to(config["device"])
        # we need to normalize the images such that they are in [-1, 1] and not [0, 1]
        preprocess = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # keep tracko of al errors
    all_errors = []

    # we do not need to keep track of gradients
    with torch.no_grad():

        # for each batch of images
        for batch_eval in eval_loader:

            # push to gpu
            eval_images = batch_eval["image"].to(config["device"])

            # get the reconstruction
            eval_recon = model(eval_images)["xhat"]

            # compute the error between reconstruction of the eval images
            # and the corresponding eval image, i.e. same scene but once as
            # real and once as toy or realistic rendering

            if which_metric == "ssim":
                error = criterion(eval_recon, eval_images)

            elif which_metric == "l1":
                error = criterion(eval_recon, eval_images).sum(axis=[1,2,3])

            elif which_metric == "lpips":
                # make sure both images are in [-1, 1]
                eval_recon = preprocess(eval_recon)
                eval_images = preprocess(eval_images)
                error = criterion.forward(eval_recon, eval_images)

            # keep track of the errors and remove all unncessary dimensions
            all_errors.extend(error.cpu().numpy().squeeze())

    # make sure its a numpy array
    all_errors = np.array(all_errors)

    # make sure the number of collected samples matches the dataset size
    assert all_errors.shape[0] == len(eval_loader.dataset)

    # compute the mean over the dataset
    return all_errors.mean()


##############################################################################################################################################################

def evaluate_experiments(experiments, which_metric):

    # keep track of all the results
    all_results = dict()
    all_results_itself = dict()

    # for each experiment
    for experiment in experiments:
            
        # get the path to all the folder and files
        folders = get_folder(experiment)

        # load the config file
        config = toml.load(folders["config"])

        # define the device
        config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # get the training data and the model
        train_loader, model = get_setup(config, folders, load_model=True)

        # get the config for the real data
        eval_config = copy.deepcopy(config)
        eval_config["dataset"]["name"] = "mpi3d"
        eval_config["dataset"]["factor"] = "real"

        # get the data loader for evaluation
        eval_loader = get_setup(eval_config, folders, load_model=False)

        # compute the recon norm metric for this training and eval data
        all_results[experiment] = compare_recon_to_train(model, eval_config, train_loader, eval_loader, which_metric)
        all_results_itself[experiment] = compare_recon_to_itself(model, eval_config, eval_loader, which_metric)
        

    # for each experiment
    print()
    print("/"*77)
    print("Metric used: ", which_metric)
    print("/"*77)
    for experiment, value in all_results.items():
        print(f"Errors for experiment {experiment} - Mean: {value:.3f}")
    for experiment, value in all_results_itself.items():
        print(f"Errors for experiment {experiment} - Mean: {value:.3f}")
    print("/"*77)


##############################################################################################################################################################

if __name__ == "__main__":

    # specify which gpu should be visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # list of all experiments to check
    experiments = [

    ]

    which_metric = "ssim"
    # which_metric = "l1"
    # which_metric = "lpips"

    # evaluate all the experiments from the list
    evaluate_experiments(experiments, which_metric)

    

