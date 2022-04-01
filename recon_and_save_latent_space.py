##############################################################################################################################################################
##############################################################################################################################################################
"""
This script reconstruct examples and extracts and saves the latent space for different datasets.
You can either use the best or last checkpoint of your models trained.

Simply insert the experiments to be evaluated inside the following list.
The folders need to be located inside the results folder.

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

import torchvision.transforms.functional as TF

import utils
import dataset

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

    folders["images"].mkdir(parents=True, exist_ok=True)
    folders["latent"].mkdir(parents=True, exist_ok=True)

    # save the console output to a file
    sys.stdout = utils.Tee(original_stdout=sys.stdout, file=folders["logs"] / "testing.log")

    return folders

##############################################################################################################################################################        

def get_setup(config, folders, load_model=False, which_model=None, get_all=False):

    if config["model"]["type"] == "classification":
        raise ValueError("Your config is for a classification model, but this script is for autoencoders")

    # load data
    train_loader = dataset.create_dataset(
        which_dataset=config["dataset"]["name"], 
        which_factor=config["dataset"]["factor"], 
        use_triplet=False,
        should_augment=False,
        make_scene_impossible=False,
        make_instance_impossible=False,
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        get_all=get_all,
    )

    if load_model:

        # get path to the loader script
        model_file = folders["scripts"] / "model"

        # replace the / by .
        string_model_file = str(model_file).replace("/",".")

        # load the model from the file
        model_file = import_module(string_model_file, package="autoencoder")

        # get the model definition
        model_def = getattr(model_file, "create_ae_model")

        # define model
        model = model_def(config["model"], config["dataset"], config["training"]).to(config["device"])
        model.print_model()

        # load the model weights
        checkpoint = torch.load(folders["checkpoint"] / (which_model + "_model.pth"), map_location=config["device"])

        # apply the model weights
        model.load_state_dict(checkpoint) 

        return train_loader, model

    return train_loader

##############################################################################################################################################################

def get_and_save_latent_space(model, data_loader, config, folders, split, which_model):

    print("=" * 77)
    print(f"Get latent space distribution: model ...")

    # keep track of latent space
    mus = []
    if config["dataset"]["name"] != "mpi3d": labels = []

    # make sure we are in eval mode
    model.eval()

    # we do not need to keep track of gradients
    with torch.no_grad():

        # for each batch of images
        torch.manual_seed("0")
        for input_images in data_loader:

            # push to gpu
            if config["dataset"]["name"] != "mpi3d":
                gt_left = input_images["gt_left"].cpu().numpy()
                gt_middle = input_images["gt_middle"].cpu().numpy()
                gt_right =input_images["gt_right"].cpu().numpy()
            input_images = input_images["image"].to(config["device"])

            # encode the images
            latent = model(input_images)

            # keep track of latent space
            mus.extend(latent["mu"].cpu().numpy()) 
            if config["dataset"]["name"] != "mpi3d":
                curr_labels = [[x,y,z] for x,y,z in zip(gt_left, gt_middle, gt_right)]
                labels.extend(curr_labels)

            if split in ["train", "ticam"]:
                flipped_input_images = torch.stack([TF.hflip(x) for x in input_images]).to(config["device"])
                flipped_latent = model(flipped_input_images)
                mus.extend(flipped_latent["mu"].cpu().numpy()) 
                if config["dataset"]["name"] != "mpi3d":
                    curr_flipped_labels = [[z,y,x] for x,y,z in zip(gt_left, gt_middle, gt_right)]
                    labels.extend(curr_flipped_labels)

    # otherwise not useable

    mus = np.array(mus)
    save_path = folders["latent"] / (split + "_" + which_model + "_mu")
    np.save(save_path, mus)

    if config["dataset"]["name"] != "mpi3d": 
        labels = np.array(labels)
        save_path = folders["latent"] / (split + "_" + which_model + "_labels")
        np.save(save_path, labels)


##############################################################################################################################################################

def recon_samples(model, data_loader, config, folders, split, which_model):

    # make sure we are in eval mode
    model.eval()

    # do not keep track of gradients
    with torch.no_grad():

        # for each batch of training images
        torch.manual_seed("0")
        for idx, input_images in enumerate(data_loader):
            
            # push to gpu
            input_images = input_images["image"].to(config["device"])

            # encode the images
            model_output = model(input_images)

            # plot the samples
            utils.plot_progress(input_images, model_output["xhat"], folders, idx, text=split+"_"+which_model+"_batch")

            if idx==9:
                return

##############################################################################################################################################################

if __name__ == "__main__":

    # list of all experiments to check
    experiments = [

    ]

    # best or last model
    which_model = "last"
    # which_model = "best"

    # specify which gpu should be visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # the main folder containing all results
    result_folder = Path("results")

    # for each experiment
    for experiment in experiments:
            
        # get the path to all the folder and files
        folders = get_folder(experiment)

        # check if the folder exists
        if not folders["experiment"].is_dir():
            print("The specified experiment does not exist: Exit script.")
            sys.exit()

        # load the config file
        config = toml.load(folders["config"])

        # define the device
        config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # get the training data and the model
        train_loader, model = get_setup(config, folders, load_model=True, which_model=which_model, get_all=False)

        # save the latent space
        get_and_save_latent_space(model, train_loader, config, folders, split="train", which_model=which_model)

        if config["dataset"]["name"].lower() == "mpi3d":

            # recon training samples
            recon_samples(model, train_loader, config, folders, split="mpi3d", which_model=which_model)

            vehicle_config = copy.deepcopy(config)
            vehicle_config["dataset"]["name"] = "mpi3d"
            vehicle_config["dataset"]["factor"] = "real"

            vehicle_train_loader = get_setup(vehicle_config, folders, load_model=False)
            recon_samples(model, vehicle_train_loader, vehicle_config, folders, split="mpi3d", which_model=which_model)
            get_and_save_latent_space(model, vehicle_train_loader, vehicle_config, folders, split="mpi3d", which_model=which_model)

        elif config["dataset"]["name"].lower() == "mnist":

            # recon training samples
            recon_samples(model, train_loader, config, folders, split="mnist", which_model=which_model)

            vehicle_config = copy.deepcopy(config)
            vehicle_config["dataset"]["name"] = "fonts"

            vehicle_train_loader = get_setup(vehicle_config, folders, load_model=False)
            recon_samples(model, vehicle_train_loader, vehicle_config, folders, split="fonts", which_model=which_model)

        else:
            for vehicle in ["cayenne", "kona", "kodiaq"]:
                
                vehicle_config = copy.deepcopy(config)

                vehicle_config["dataset"]["name"] = "sviro_illumination"
                vehicle_config["dataset"]["factor"] = vehicle

                vehicle_train_loader = get_setup(vehicle_config, folders, load_model=False, get_all=True)
                recon_samples(model, vehicle_train_loader, vehicle_config, folders, split=vehicle, which_model=which_model)
                get_and_save_latent_space(model, vehicle_train_loader, vehicle_config, folders, split=vehicle, which_model=which_model)

            vehicle_config = copy.deepcopy(config)
            vehicle_config["dataset"]["name"] = "ticam"
            vehicle_config["dataset"]["factor"] = "all"

            vehicle_train_loader = get_setup(vehicle_config, folders, load_model=False)
            recon_samples(model, vehicle_train_loader, vehicle_config, folders, split="ticam", which_model=which_model)
            get_and_save_latent_space(model, vehicle_train_loader, vehicle_config, folders, split="ticam", which_model=which_model)

            vehicle_config = copy.deepcopy(config)
            vehicle_config["dataset"]["name"] = "sviro"
            vehicle_config["dataset"]["factor"] = "all"

            vehicle_train_loader = get_setup(vehicle_config, folders, load_model=False)
            recon_samples(model, vehicle_train_loader, vehicle_config, folders, split="sviro", which_model=which_model)
            get_and_save_latent_space(model, vehicle_train_loader, vehicle_config, folders, split="sviro", which_model=which_model)

        # reset the stdout with the original one
        # this is necessary when the train function is called several times
        # by another script
        sys.stdout = sys.stdout.end()