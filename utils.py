##############################################################################################################################################################
##############################################################################################################################################################
"""
Helperfunctions used by other scripts.
"""
##############################################################################################################################################################
##############################################################################################################################################################

import toml
import time
import torch
import datetime
import dateutil

import numpy as np

from torchvision.utils import save_image as save_grid
from torchvision.utils import make_grid

from matplotlib import pyplot as plt
plt.style.use(['seaborn-white', 'seaborn-paper'])
plt.rc('font', family='serif')

##############################################################################################################################################################
##############################################################################################################################################################

class Tee(object):
    """
    Class to make it possible to print text to the console and also write the 
    output to a file.
    """

    def __init__(self, original_stdout, file):

        # keep the original stdout
        self.original_stdout = original_stdout

        # the file to write to
        self.log_file_handler= open(file, 'w')

        # all the files the print statement should be saved to
        self.files = [self.original_stdout, self.log_file_handler]

    def write(self, obj):

        # for each file
        for f in self.files:

            # write to the file
            f.write(obj)

            # If you want the output to be visible immediately
            f.flush() 

    def flush(self):

        # for each file
        for f in self.files:

            # If you want the output to be visible immediately
            f.flush()

    def end(self):

        # close the file
        self.log_file_handler.close()

        # return the original stdout
        return self.original_stdout

##############################################################################################################################################################

def write_toml_to_file(cfg, file_path):
    """
    Write the parser to a file.
    """

    with open(file_path, 'w') as output_file:
        toml.dump(cfg, output_file)

    print('=' * 57)
    print("Config file saved: ", file_path)
    print('=' * 57)

##############################################################################################################################################################

def append_accuracy(save_folder, score):

    with save_folder["accuracy"].open('a') as file:
        file.write(str(score)+"\n")

def save_accuracy(save_path, score):

    with save_path.open('w') as file:
        file.write(str(score))

##############################################################################################################################################################

class TrainingTimer():
    """
    Keep track of training times.
    """

    def __init__(self):

        # get the current time
        self.start = datetime.datetime.fromtimestamp(time.time())
        self.eval_start = datetime.datetime.fromtimestamp(time.time())

        # print it human readable
        print("Training start: ", self.start)
        print('=' * 57)

    def print_end_time(self):

        # get datetimes for simplicity
        datetime_now = datetime.datetime.fromtimestamp(time.time())

        print("Training finish: ", datetime_now)

        # compute the time different between now and the input
        rd = dateutil.relativedelta.relativedelta(datetime_now, self.start)

        # print the duration in human readable
        print(f"Training duration: {rd.hours} hours, {rd.minutes} minutes, {rd.seconds} seconds")

    
    def print_time_delta(self):

        # get datetimes for simplicity
        datetime_now = datetime.datetime.fromtimestamp(time.time())

        # compute the time different between now and the input
        rd = dateutil.relativedelta.relativedelta(datetime_now, self.eval_start)

        # print the duration in human readable
        print(f"Duration since last evaluation: {rd.hours} hours, {rd.minutes} minutes, {rd.seconds} seconds")
        print('=' * 57)

        # update starting time
        self.eval_start = datetime.datetime.fromtimestamp(time.time())

##############################################################################################################################################################

def plot_progress(input_images, recon_images, save_folder, nbr_epoch, text, max_nbr_samples=16, **kwargs):

    # check which number is smaller and use it to extract the correct amount of images
    nbr_samples = min(input_images.shape[0], max_nbr_samples)

    # collect what to plot
    gather_sample_results = []
    gather_sample_results.append(input_images[:nbr_samples].detach())
    gather_sample_results.append(recon_images[:nbr_samples].detach())

    # make it a torch tensor
    gather_sample_results = torch.cat(gather_sample_results, dim=0)

    # create the filename
    save_name = save_folder["images"] / (text + "_" + str(nbr_epoch) + ".pdf")

    # save images as grid
    # each element in the batch has its own column
    save_grid(gather_sample_results, save_name, nrow=nbr_samples, **kwargs)

##############################################################################################################################################################

def plot_recon_and_lable(input_images, labels, recon_images, predictions, save_folder, nbr_epoch, text, max_nbr_samples=16, **kwargs):

    # check which number is smaller and use it to extract the correct amount of images
    nbr_samples = min(input_images.shape[0], max_nbr_samples)

    predictions = predictions[:nbr_samples]
    labels = labels[:nbr_samples]

    # collect what to plot
    gather_sample_results = []
    gather_sample_results.append(input_images[:nbr_samples].detach())
    gather_sample_results.append(recon_images[:nbr_samples].detach())

    # make it a torch tensor
    gather_sample_results = torch.cat(gather_sample_results, dim=0)

    # create the filename
    save_name = save_folder["images"] / (text + "_" + str(nbr_epoch) + ".png")

    # save images as grid
    # each element in the batch has its own column
    grid = make_grid(gather_sample_results, nrow=nbr_samples, **kwargs).cpu().numpy()

    fig, ax = plt.subplots(figsize=(16*4, 4))
    plt.imshow(np.transpose(grid, (1,2,0)), interpolation='nearest')
    for index, (pred, labels) in enumerate(zip(predictions, labels)):
        ax.text(x=index*130+64, y=-13, s=str(labels), fontsize=11, horizontalalignment="center", verticalalignment="center")
        ax.text(x=index*130+64, y=278, s=str(pred), fontsize=11, horizontalalignment="center", verticalalignment="center", color="green" if pred==labels else "red")
    plt.axis("off")
    fig.savefig(save_name, dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig=fig)

##############################################################################################################################################################

def stringify(x):
    # get the labels from each dimension and combine them to a string
    return "_".join(str(y) for y in x)
    
##############################################################################################################################################################