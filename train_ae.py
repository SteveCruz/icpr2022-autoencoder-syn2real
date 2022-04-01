##############################################################################################################################################################
##############################################################################################################################################################
"""
Training scripts for autoencoder models presented in our paper.

Replace or modify the config file in the following part of the code to make changes to train different models.

# load the config file
config = toml.load("cfg/extractor_ae.toml")
"""
##############################################################################################################################################################
##############################################################################################################################################################

import os
import sys
import time
import toml
import torch
import random

import numpy as np
from joblib import dump
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from shutil import copyfile
from sklearn.neighbors import KNeighborsClassifier

import torchvision.transforms.functional as TF
from torch.nn import functional as F

import utils as utils
from model import create_ae_model
from dataset import create_dataset

##############################################################################################################################################################
##############################################################################################################################################################

def folder_setup(config):

    # define the name of the experiment
    if config["model"]["type"] == "classification":
        experiment_name = time.strftime("%Y%m%d-%H%M%S") + "_" + config["model"]["architecture"] + "_" + config["dataset"]["name"]  + "_pretrained_" + str(config["model"]["use_pretrained"])
    else:
        experiment_name = time.strftime("%Y%m%d-%H%M%S") + "_" + config["model"]["type"] + "_" + config["dataset"]["name"]  + "_latentDimension_" + str(config["model"]["dimension"])

    # define the paths to save the experiment
    save_folder = dict()
    Path("results").mkdir(exist_ok=True)
    save_folder["main"] = Path("results") / experiment_name
    save_folder["checkpoints"] = save_folder["main"] / "checkpoints" 
    save_folder["images"] = save_folder["main"] / "images" / "train"
    save_folder["data"] = save_folder["main"] / "data"
    save_folder["latent"] = save_folder["main"] / "data" / "latent"
    save_folder["scripts"] = save_folder["main"] / "scripts"
    save_folder["logs"] = save_folder["main"] / "logs"

    # create all the folders
    for item in save_folder.values():
        item.mkdir(parents=True)

    # save the console output to a file and to the console
    sys.stdout = utils.Tee(original_stdout=sys.stdout, file=save_folder["logs"] / "training.log")

    # save the accuracy for each evaluation to a file
    save_folder["accuracy"] = save_folder["logs"] / "accuracy.log"
    save_folder["accuracy"].touch()

    # copy files as a version backup
    # this way we know exactly what we did
    # these can also be loaded automatically for testing the models
    copyfile(Path(__file__).absolute(), save_folder["scripts"] / "train_ae.py")
    copyfile(Path().absolute() / "dataset.py", save_folder["scripts"] / "dataset.py")
    copyfile(Path().absolute() / "model.py", save_folder["scripts"] / "model.py")
    copyfile(Path().absolute() / "utils.py", save_folder["scripts"] / "utils.py")
    copyfile(Path().absolute() / "pretrained_model.py", save_folder["scripts"] / "pretrained_model.py")
    copyfile(Path().absolute() / "repeat_train_classifier.py", save_folder["scripts"] / "repeat_train_classifier.py")
    copyfile(Path().absolute() / "repeat_train_ae.py", save_folder["scripts"] / "repeat_train_ae.py")
    copyfile(Path().absolute() / "train_classifier.py", save_folder["scripts"] / "train_classifier.py")

    # save config file
    # remove device info, as it is not properly saved
    config_to_save = config.copy()
    del config_to_save["device"]
    utils.write_toml_to_file(config_to_save, save_folder["main"] / "cfg.toml")

    return save_folder

##############################################################################################################################################################

def model_setup(config, save_folder):

    if config["model"]["type"] == "classification":
        raise ValueError("Your config is for a classification model, but this script is for autoencoders. Please use train_classifier.py instead.")

    # define model and print it
    model = create_ae_model(config["model"], config["dataset"], config["training"]).to(config["device"])
    model.print_model()

    # get the optimizer defined in the config file
    # load it from the torch module
    optim_def = getattr(optim, config["training"]["optimizer"])

    # create the optimizer 
    optimizer = dict()
    optimizer["method"] = []
    
    if model.type == "factorvae":
        ae_param = list(model.encoder.parameters()) + list(model.encoder_fc21.parameters()) + list(model.encoder_fc22.parameters()) + list(model.decoder_fc.parameters()) + list(model.decoder_conv.parameters())
        optimizer["method"].append(optim_def(ae_param, lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"], betas=(0.9, 0.999)))
        optimizer["method"].append(optim_def(model.discriminator.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"], betas=(0.5, 0.9)))
    else:
        if config["training"]["optimizer"] == "SGD":
            optimizer["method"].append(optim_def(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"], momentum=0.9, nesterov=True))
        else:
            optimizer["method"].append(optim_def(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]))


    print('=' * 73)
    print(optimizer["method"])
    print('=' * 73)

    # load data
    train_loader = create_dataset(
        which_dataset=config["dataset"]["name"], 
        which_factor=config["dataset"]["factor"], 
        use_triplet=True if config["model"]["variation"]=="tae" else False,
        should_augment=config["training"]["augment"],
        make_scene_impossible=config["training"]["make_scene_impossible"],
        make_instance_impossible=config["training"]["make_instance_impossible"],
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        get_all=False
    )

    # save the dict to transform labels to int
    np.save(save_folder["checkpoints"] / 'label_dict.npy', train_loader.dataset.string_labels_to_integer_dict) 

    return model, optimizer, train_loader, train_loader.dataset.string_labels_to_integer_dict

##############################################################################################################################################################

def get_test_loader(config, real_only=False, get_train_loader=False):

    # check whether we need vehicle images or mpi3d
    vehicle_images=False if config["dataset"]["name"].lower() in ["mpi3d", "mnist", "fonts"] else True

    # dict to keep a loader for each test vehicle
    test_loader = dict()

    # either we train on vehicle images, or on mpi3d
    if vehicle_images:
        if not real_only:
            for vehicle in ["cayenne", "kona", "kodiaq"]:
                test_loader[vehicle] = create_dataset(
                    which_dataset="sviro_illumination",
                    which_factor=vehicle,
                    use_triplet=False,
                    should_augment=False,
                    batch_size=config["training"]["batch_size"],
                    shuffle=True,
                    get_all=True
                )
                
        test_loader["ticam"] = create_dataset(
            which_dataset="ticam",
            which_factor="all",
            use_triplet=False,
            should_augment=False,
            batch_size=512,
            shuffle=True,
            get_all=False
        )

        if get_train_loader:
            test_loader["train"] = create_dataset(
                which_dataset=config["dataset"]["name"], 
                which_factor=config["dataset"]["factor"], 
                use_triplet=False,
                should_augment=False,
                make_scene_impossible=False,
                make_instance_impossible=False,
                batch_size=512, 
                shuffle=False, 
                get_all=False
            )

    # mpi3d
    else:

        if config["dataset"]["name"].lower() == "mpi3d":
            test_loader["real"] = create_dataset(
                which_dataset="mpi3d",
                which_factor="real", 
                use_triplet=False,
                should_augment=False,
                batch_size=config["training"]["batch_size"], 
                shuffle=True, 
                get_all=False
            )

        elif config["dataset"]["name"].lower() == "mnist":
            test_loader["real"] = create_dataset(
                which_dataset="fonts",
                which_factor="test", 
                use_triplet=False,
                should_augment=False,
                batch_size=config["training"]["batch_size"], 
                shuffle=True, 
                get_all=False
            )

        else:
            test_loader["real"] = create_dataset(
                which_dataset=config["dataset"]["name"].lower(),
                which_factor="validation", 
                use_triplet=False,
                should_augment=False,
                batch_size=config["training"]["batch_size"], 
                shuffle=True, 
                get_all=False
            )

    return test_loader

##############################################################################################################################################################

def train_one_epoch(model, optimizer, scaler, train_loader, config, save_folder, nbr_epoch):

    # make sure we are training
    model.train()

    # init
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_vae_tc_loss = 0
    total_d_tc_loss = 0
    total_tp_loss = 0

    # for each batch
    train_loader = iter(train_loader)
    for batch_idx, batch_images in enumerate(train_loader):

        # init
        batch_loss = 0
        batch_recon_loss = 0
        batch_kl_loss = 0
        batch_vae_tc_loss = 0
        batch_d_tc_loss = 0
        batch_tp_loss = 0
        
        # set gradients to zero
        optimizer["method"][0].zero_grad()

        # push to gpu
        input_images = batch_images["image"].to(config["device"])
        output_images = batch_images["target"].to(config["device"])

        if model.variation == "tae":
            positive_input_images = batch_images["positive"].to(config["device"])
            positive_output_image = batch_images["positive_target"].to(config["device"])
            negative_input_images = batch_images["negative"].to(config["device"])
            negative_output_image = batch_images["negative_target"].to(config["device"])

        # inference
        with autocast():
            model_output = model(input_images)
            
            if model.variation == "tae":
                positive_output = model(positive_input_images)
                negative_output = model(negative_input_images)

        # calculate the loss
        batch_recon_loss = model.loss(prediction=model_output["xhat"].to(torch.float32), target=output_images)
        if config["training"]["loss"] == "SSIM" or config["training"]["loss"] == "BCE":
            batch_loss += batch_recon_loss
        else:
            batch_loss += batch_recon_loss / config["training"]["batch_size"]

        if model.variation == "tae":
            positive_recon_loss = model.loss(prediction=positive_output["xhat"].to(torch.float32), target=positive_output_image)
            negative_recon_loss = model.loss(prediction=negative_output["xhat"].to(torch.float32), target=negative_output_image)
            if config["training"]["loss"] == "SSIM" or config["training"]["loss"] == "BCE":
                batch_loss += positive_recon_loss
                batch_loss += negative_recon_loss
            else:
                batch_loss += positive_recon_loss / config["training"]["batch_size"]
                batch_loss += negative_recon_loss / config["training"]["batch_size"]

            # triplet loss
            batch_tp_loss = model.triplet_loss(anchor=model_output["mu"], positive=positive_output["mu"], negative=negative_output["mu"])
            batch_loss += batch_tp_loss

        if model.type == "vae" or model.type == "factorvae":
            batch_kl_loss = model.kl_divergence_loss(model_output["mu"], model_output["logvar"])
            batch_loss += config["training"]["kl_weight"]*batch_kl_loss

        if model.type == "factorvae":
            D_z_reserve = model.discriminator(model_output["z"])
            batch_vae_tc_loss = (D_z_reserve[:, 0] - D_z_reserve[:, 1]).mean()
            batch_loss += config["training"]["tc_weight"]*batch_vae_tc_loss

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        if model.type == "factorvae":
            scaler.scale(batch_loss).backward(retain_graph=True)
        else:
            scaler.scale(batch_loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer["method"][0])

        # Updates the scale for next iteration.
        scaler.update()

        if model.type == "factorvae":

            batch_images_2 = next(train_loader)
            input_images_2 = batch_images_2["image"].to(config["device"])
            with autocast():
                model_output_2 = model(input_images_2)

            true_labels = torch.ones(input_images_2.size(0), dtype= torch.long, requires_grad=False).to(config["device"])
            false_labels = torch.zeros(input_images_2.size(0), dtype= torch.long, requires_grad=False).to(config["device"])

            z_perm = model.permute_latent(model_output_2["z"]).detach()
            D_z_perm = model.discriminator(z_perm)
            batch_d_tc_loss = 0.5 * (F.cross_entropy(D_z_reserve, false_labels) + F.cross_entropy(D_z_perm, true_labels))

            optimizer["method"][1].zero_grad()
            scaler.scale(batch_d_tc_loss).backward()
            scaler.step(optimizer["method"][1])
            scaler.update()

        # accumulate loss
        total_loss += batch_loss.item()
        total_recon_loss += batch_recon_loss.item()
        if model.variation == "tae":
            total_tp_loss += batch_tp_loss.item()
        if model.type == "vae" or model.type == "factorvae":
            total_kl_loss += batch_kl_loss.item()
        if model.type == "factorvae":
            total_vae_tc_loss += batch_vae_tc_loss.item()
            total_d_tc_loss += batch_d_tc_loss.item()

        if ((nbr_epoch+1) % config["training"]["frequency"] == 0 or nbr_epoch == 1) and batch_idx == 0:
            utils.plot_progress(input_images, model_output["xhat"], save_folder, nbr_epoch, text="epoch")
            if model.variation == "tae":
                utils.plot_progress(positive_input_images, positive_output["xhat"], save_folder, nbr_epoch, text="epoch_positive")
                utils.plot_progress(negative_input_images, negative_output["xhat"], save_folder, nbr_epoch, text="epoch_negative")
            if "target" in batch_images:
                utils.plot_progress(input_images, output_images, save_folder, nbr_epoch, text="epoch_target")
         
    print(f"[Training] \tEpoch: {nbr_epoch+1} Total Loss: {total_loss:.2f} \tRecon Loss: {total_recon_loss:.2f} \tTriplet Loss: {total_tp_loss:.2f} \tKL Loss: {total_kl_loss:.2f} \tVAE-TC Loss: {batch_vae_tc_loss:.2f} \tD-TC Loss: {batch_d_tc_loss:.2f}")

    return model

##############################################################################################################################################################

def recon_one_batch(model, loader_dict, config, save_folder, nbr_epoch, split):

    if (nbr_epoch+1) % config["training"]["frequency"] == 0 or nbr_epoch == 1:

        # save the last model
        torch.save(model.state_dict(), save_folder["checkpoints"] / "last_model.pth")

        # make sure we are in eval mode
        model.eval()

        # do not keep track of gradients
        with torch.no_grad():

            # for the loader of each test vehicle
            for vehicle, loader in loader_dict.items():
                
                # for each batch of training images
                for batch_idx, input_images in enumerate(loader):
                    
                    # push to gpu
                    input_images = input_images["image"].to(config["device"])

                    # encode the images
                    model_output = model(input_images)

                    # plot the samples
                    utils.plot_progress(input_images, model_output["xhat"], save_folder, nbr_epoch, text=f"{split}_{vehicle}_{batch_idx}_epoch")

                    # leave the loop after the first batch
                    break
            

##############################################################################################################################################################

def get_data(model, config, data_loader, labels_dict, is_train):

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
            gt_left = batch["gt_left"].numpy()
            gt_middle = batch["gt_middle"].numpy()
            gt_right =batch["gt_right"].numpy()
            input_images = batch["image"].to(config["device"])

            # get the flipped versions as well
            flipped_input_images = torch.stack([TF.hflip(x) for x in input_images])

            # encode the images
            latent = model(input_images)["mu"]
            flipped_latent = model(flipped_input_images)["mu"]

            # keep track of latent space
            mus.extend(latent.cpu().numpy()) 
            curr_labels = [labels_dict[utils.stringify([x,y,z])] for x,y,z in zip(gt_left, gt_middle, gt_right)]
            labels.extend(curr_labels)
            
            mus.extend(flipped_latent.cpu().numpy()) 
            curr_flipped_labels = [labels_dict[utils.stringify([z,y,x])] for x,y,z in zip(gt_left, gt_middle, gt_right)]
            labels.extend(curr_flipped_labels)

        # otherwise not useable
        mus = np.array(mus)
        labels = np.array(labels)

        return mus, labels
 

def evaluate(model, labels_dict, train_loader, test_loader, config, save_folder, nbr_epoch, best_score):

    if (nbr_epoch+1) % config["training"]["frequency"] == 0:

        # get the training data 
        train_mu, train_labels = get_data(model, config, train_loader, labels_dict=labels_dict, is_train=True)

        # get the evaluation data
        ticam_mu, ticam_labels = get_data(model, config, test_loader["ticam"], labels_dict=labels_dict, is_train=False)

        # define the classifier
        classifier = KNeighborsClassifier(n_neighbors=config["model"]["knn"], n_jobs=-1)

        # train the classifier
        classifier.fit(train_mu, train_labels)   

        # evaluate the classifier on this data
        score = classifier.score(ticam_mu, ticam_labels)
        
        print(f"[Testing] \tAccuracy Ticam: {100*score:.2f}% (best: {100*best_score:.2f}) (Nbr-Train: {train_labels.shape[0]}, Nbr-Ticam: {ticam_labels.shape[0]})")

        # append result to file
        utils.append_accuracy(save_folder, score)

        # check if its the best so far
        if score > best_score:
            torch.save(model.state_dict(), save_folder["checkpoints"] / "best_model.pth")
            dump(classifier, save_folder["checkpoints"] / f'best_classifier.joblib')
            best_score = score

    return best_score

##############################################################################################################################################################

def train(config):

    #########################################################
    # GPU
    #########################################################

    # specify which gpu should be visible
    os.environ["CUDA_VISIBLE_DEVICES"] = config["training"]["gpu"]

    # save the gpu settings
    config["device"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # gradscaler to improve speed performance with mixed precision training
    scaler = GradScaler()
    
    #########################################################
    # Setup
    #########################################################

    # create the folders for saving
    save_folder = folder_setup(config)

    # create the model, optimizer and data loader
    model, optimizer, train_loader, labels_dict = model_setup(config, save_folder)

    # get also a test loader for evaluation on unseen dataset
    test_loader = get_test_loader(config, real_only=True, get_train_loader=True)

    #########################################################
    # Training
    #########################################################

    # keep track of time
    timer = utils.TrainingTimer()

    # init
    best_score = 0

    # for each epoch
    for nbr_epoch in range(config["training"]["epochs"]):

        # train a single epoch
        model = train_one_epoch(model, optimizer, scaler, train_loader, config, save_folder, nbr_epoch)
        
        # reconstruct one batch for each loader for visualization purposes
        recon_one_batch(model, test_loader, config, save_folder, nbr_epoch, split="test")

        # get the latent space for all datasets for the current epoch
        if config["dataset"]["name"].lower() not in ["mpi3d", "mnist", "fonts"]:
            best_score = evaluate(model, labels_dict, test_loader["train"], test_loader, config, save_folder, nbr_epoch, best_score)

    #########################################################
    # Aftermath
    #########################################################     

    # save the last model
    torch.save(model.state_dict(), save_folder["checkpoints"] / "last_model.pth")

    print("=" * 37)
    timer.print_end_time()
    print("=" * 37)

    # reset the stdout with the original one
    # this is necessary when the train function is called several times
    # by another script
    sys.stdout = sys.stdout.end()
    
##############################################################################################################################################################
##############################################################################################################################################################

if __name__ == "__main__":

    # reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    # load the config file
    # config = toml.load("cfg/conv_ae.toml")
    # config = toml.load("cfg/vae.toml")
    # config = toml.load("cfg/factorvae.toml")
    config = toml.load("cfg/extractor_ae.toml")
    
    # start the training using the config file
    train(config)