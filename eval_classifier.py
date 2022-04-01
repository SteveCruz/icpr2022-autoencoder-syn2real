##############################################################################################################################################################
##############################################################################################################################################################
"""
Evaluate classification models after training.
The best model checkpoint obtained during training (according to accuracy) is used for the evaluation.

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

import matplotlib as mpl
from matplotlib import pyplot as plt
plt.style.use(['seaborn-white', 'seaborn-paper'])
plt.rc('font', family='serif')
# force matplotlib to not plot images
mpl.use("Agg")

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
    folders["images"] = folders["experiment"] / "images" / "predictions"
    folders["checkpoint"] = folders["experiment"] / "checkpoints" 
    folders["config"] = folders["experiment"] / "cfg.toml"

    folders["images"].mkdir(parents=True, exist_ok=True)
    
    # save the console output to a file
    sys.stdout = utils.Tee(original_stdout=sys.stdout, file=folders["logs"] / "testing.log")

    return folders

##############################################################################################################################################################        

def get_setup(config, folders, load_model=False, get_all=False):

    if config["model"]["type"] != "classification":
        raise ValueError("Your config is for a autoencoder model, but this script is for classification models. Please use eval_ae.py instead.")

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
        get_all=get_all
    )

    if load_model:

        # get path to the loader script
        model_file = folders["scripts"] / "pretrained_model"

        # replace the / by .
        string_model_file = str(model_file).replace("/",".")

        # load the model definitions from the file
        model_file = import_module(string_model_file, package="autoencoder")

        # get the autoencoder model definition
        model_def = getattr(model_file, "PretrainedClassifier")

        # get the dict to transform the strings to integer labels
        labels_dict = np.load(folders["checkpoint"] / "label_dict.npy", allow_pickle='TRUE')

        # this is now a numpy array and not a real dict anymore. 
        # hence we first need to acces the item, which is the dict
        nbr_classes = len(labels_dict.item().keys())

        # define autoencoder model
        model = model_def(config["model"], nbr_classes=nbr_classes, print_model=False).to(config["device"])

        # load the autoencoder model weights
        checkpoint = torch.load(folders["checkpoint"] / "best_model.pth", map_location=config["device"])

        # apply the model weights
        model.load_state_dict(checkpoint) 

        return train_loader, model, labels_dict

    return train_loader

##############################################################################################################################################################

def evaluate_model(model, labels_dict, test_loader, config, folders, split):

    # dict to transform string labels to integers
    string_to_int = labels_dict.item()

    # placeholder
    correct = 0
    total = 0

    # keep track of wrongly classified images
    wrong_classified_images = []
    wrong_classified_labels = []
    wrong_classified_predictions = []

    # make sure we are in eval mode
    model.eval()

    # we do not need to keep track of gradients
    with torch.no_grad():

        # for each batch of images
        torch.manual_seed("0")
        for batch in test_loader:

            images_to_use = []
            labels_to_use = []

            # push to gpu
            input_images = batch["image"].to(config["device"])
            labels_left = batch["gt_left"].cpu().numpy()
            labels_middle = batch["gt_middle"].cpu().numpy()
            labels_right =batch["gt_right"].cpu().numpy()
            string_labels = [str(x.item())+"_"+str(y.item())+"_"+str(z.item()) for x,y,z in zip(labels_left, labels_middle, labels_right)]

            for x,y in zip(input_images, string_labels):
                if y in string_to_int:
                    images_to_use.append(x)
                    labels_to_use.append(y)

            images_to_use = torch.stack(images_to_use)
            labels_to_use = torch.tensor([string_to_int[x] for x in labels_to_use]).to(config["device"])

            classif_output = model(images_to_use)

            _, predictions = torch.max(classif_output, 1)
            correct += (predictions == labels_to_use).sum().item()
            total += labels_to_use.size(0)

            if split in ["train", "ticam"]:
                flipped_images_to_use = []
                flipped_labels_to_use = []
                flipped_input_images = torch.stack([TF.hflip(x) for x in input_images])
                flipped_string_labels = [str(z.item())+"_"+str(y.item())+"_"+str(x.item()) for x,y,z in zip(labels_left, labels_middle, labels_right)]
                for x,y in zip(flipped_input_images, flipped_string_labels):
                    if y in string_to_int:
                        flipped_images_to_use.append(x)
                        flipped_labels_to_use.append(y)
                flipped_images_to_use = torch.stack(flipped_images_to_use)
                flipped_labels_to_use = torch.tensor([string_to_int[x] for x in flipped_labels_to_use]).to(config["device"])
                flipped_classif_output = model(flipped_images_to_use)
                _, flipped_predictions = torch.max(flipped_classif_output, 1)
                correct += (flipped_predictions == flipped_labels_to_use).sum().item()
                total += flipped_labels_to_use.size(0)

            wrong_classified_images.extend(images_to_use[predictions!=labels_to_use])
            wrong_classified_labels.extend(labels_to_use[predictions!=labels_to_use])
            wrong_classified_predictions.extend(predictions[predictions!=labels_to_use])

    # compute the epoch accuracy
    accuracy = correct / total

    print(f"[Testing] Vehicle: {split}, Accuracy: {100*accuracy:.2f}% ({correct}/{total})")

    # save accuracy to file
    save_path = folders["logs"] / f"{split}.accuracy"
    utils.save_accuracy(save_path, accuracy)

##############################################################################################################################################################

if __name__ == "__main__":

    # list of all experiments to check
    experiments = [


    ]

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
        train_loader, model, labels_dict = get_setup(config, folders, load_model=True)

        vehicle_config = copy.deepcopy(config)
        vehicle_config["dataset"]["name"] = "sviro_illumination"
        vehicle_config["dataset"]["factor"] = "all"

        vehicle_train_loader = get_setup(vehicle_config, folders, load_model=False, get_all=True)
        evaluate_model(model, labels_dict, vehicle_train_loader, vehicle_config, folders, split="lightning")

        vehicle_config = copy.deepcopy(config)
        vehicle_config["dataset"]["name"] = "ticam"
        vehicle_config["dataset"]["factor"] = "all"

        vehicle_train_loader = get_setup(vehicle_config, folders, load_model=False, get_all=False)
        evaluate_model(model, labels_dict, vehicle_train_loader, vehicle_config, folders, split="ticam")

        vehicle_config = copy.deepcopy(config)
        vehicle_config["dataset"]["name"] = "sviro"
        vehicle_config["dataset"]["factor"] = "all"

        vehicle_train_loader = get_setup(vehicle_config, folders, load_model=False, get_all=False)
        evaluate_model(model, labels_dict, vehicle_train_loader, vehicle_config, folders, split="sviro")

        # reset the stdout with the original one
        # this is necessary when the train function is called several times
        # by another script
        sys.stdout = sys.stdout.end()
