##############################################################################################################################################################
##############################################################################################################################################################
"""
Evaluate autoencoder models after training.
You can either evaluate the best model checkpoint obtained during training (according to accuracy) or the last model checkpoint.
We use k-nearest neighbour in the latent space to perform the classification.

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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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

        # get the dict to transform the strings to integer labels
        labels_dict = np.load(folders["checkpoint"] / "label_dict.npy", allow_pickle='TRUE').item()

        return train_loader, model, labels_dict

    return train_loader

##############################################################################################################################################################

def get_data(model, config, data_loader, labels_dict, split):

    # keep track of latent space
    mus = []
    labels = []

    # make sure we are in eval mode
    model.eval()

    # we do not need to keep track of gradients
    with torch.no_grad():

        # for each batch of images
        for batch in data_loader:

            # push to gpu
            if config["dataset"]["name"].lower() == "mnist":
                gt =batch["gt"].numpy()
            else:
                gt_left = batch["gt_left"].numpy()
                gt_middle = batch["gt_middle"].numpy()
                gt_right =batch["gt_right"].numpy()
            input_images = batch["image"].to(config["device"])

            latent = model(input_images)["mu"]

            # keep track of latent space
            mus.extend(latent.cpu().numpy()) 
            if config["dataset"]["name"].lower() == "mnist":
                labels.extend(gt)
            else:
                curr_labels = [labels_dict[utils.stringify([x,y,z])] for x,y,z in zip(gt_left, gt_middle, gt_right)]
                labels.extend(curr_labels)
            
            # get the flipped versions as well
            if split in ["train", "ticam"] and split != "mnist":
                flipped_input_images = torch.stack([TF.hflip(x) for x in input_images])
                flipped_latent = model(flipped_input_images)["mu"]
                mus.extend(flipped_latent.cpu().numpy()) 
                curr_flipped_labels = [labels_dict[utils.stringify([z,y,x])] for x,y,z in zip(gt_left, gt_middle, gt_right)]
                labels.extend(curr_flipped_labels)

        # otherwise not useable
        mus = np.array(mus)
        labels = np.array(labels)

        return mus, labels

##############################################################################################################################################################

def evaluate_model(model, labels_dict, classifier, eval_loader, config, save_folder, split):

    # get the evaluation data
    eval_mu, eval_labels = get_data(model, config, eval_loader, labels_dict=labels_dict, split=split)

    # evaluate the classifier on this data
    score = classifier.score(eval_mu, eval_labels)
    
    print(f"[Testing] \tAccuracy {split}: {100*score:.1f}% (Nbr-Train: {train_labels.shape[0]}, Nbr-Ticam: {eval_labels.shape[0]})")

    # save accuracy to file
    save_path = save_folder["logs"] / f"{split}.accuracy"
    utils.save_accuracy(save_path, score)


##############################################################################################################################################################

if __name__ == "__main__":

    # list of all experiments to check
    experiments = [

    ]

    # best or last model
    # which_model = "last"
    # which_model = "best"

    # specify which gpu should be visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # the main folder containing all results
    result_folder = Path("results")

    # for each experiment
    for experiment in experiments:

        print("-" * 99)
        print(experiment)
        print("-" * 99)
            
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
        train_loader, model, labels_dict = get_setup(config, folders, load_model=True, which_model=which_model, get_all=False)

        # get the training data 
        train_mu, train_labels = get_data(model, config, train_loader, labels_dict=labels_dict, split="train")

        # define the classifier
        classifier = KNeighborsClassifier(n_neighbors=config["model"]["knn"], n_jobs=-1)
        # classifier = SGDClassifier()
        # classifier = SVC(kernel="linear", C=0.025)
        # classifier = RandomForestClassifier()

        # train the classifier
        classifier.fit(train_mu, train_labels)  

        if config["dataset"]["name"].lower() == "mnist":
            eval_config = copy.deepcopy(config)
            eval_config["dataset"]["name"] = "fonts"
            eval_loader = get_setup(eval_config, folders, load_model=False, get_all=False)
            evaluate_model(model, labels_dict, classifier, eval_loader, eval_config, folders, split="svhn")

        else:
            eval_config = copy.deepcopy(config)
            eval_config["dataset"]["name"] = "ticam"
            eval_config["dataset"]["factor"] = "all"
            eval_loader = get_setup(eval_config, folders, load_model=False, get_all=False)
            evaluate_model(model, labels_dict, classifier, eval_loader, eval_config, folders, split="ticam")

            eval_config = copy.deepcopy(config)
            eval_config["dataset"]["name"] = "sviro"
            eval_config["dataset"]["factor"] = "all"
            eval_loader = get_setup(eval_config, folders, load_model=False, get_all=False)
            evaluate_model(model, labels_dict, classifier, eval_loader, eval_config, folders, split="sviro")

