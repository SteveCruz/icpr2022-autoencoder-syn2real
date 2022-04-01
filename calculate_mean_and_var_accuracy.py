##############################################################################################################################################################
##############################################################################################################################################################
"""
This script computes the mean, standard deviation and max performance of the models for different runs.

You will have to have evaluated the models previously either using eval_ae.py or eval_classifier.py

You will need to manually insert the experiments which you want to use.
Replace the label and insert the folder names (needs to be located inside the results folder) of the following part of the code.

experiments.append({
        "label":"Resnet-50-II-E-TAE (Ours)",
        "data": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ]
    })

"""
#####################################################################################################################################################
#####################################################################################################################################################

import numpy as np
from pathlib import Path
from collections import defaultdict

#####################################################################################################################################################
#####################################################################################################################################################

def get_data_from_log(experiment, dataset):

    # get and set the folders to the experiment
    result_folder = Path("results") / experiment 
    log_file = result_folder / "logs" / f"{dataset}.accuracy"

    # open the accuracy log file and read the line
    with log_file.open(mode="r") as file:
        accuracy = file.read()

    # make sure all the values are float
    # also make sure we have accuracies in percentage
    accuracy = float(accuracy)*100

    return accuracy

#####################################################################################################################################################

def get_data_for_all_experiments(experiments):

    # for each block
    for block in experiments:

        # add a dict, which will contain the data read from the log files
        block["results"] = defaultdict(list)

        # for each experiment inside this block
        for experiment in block["data"]:

            if experiment == "":
                continue

            # for the datasets we want to consider
            for dataset in ["ticam", "sviro"]:

                # read the data and append it to the current block's list
                block["results"][dataset].append(get_data_from_log(experiment, dataset))

    return experiments

#####################################################################################################################################################

def calculate_accuracies(experiments):

    # for each block containint results for several runs
    for block in experiments:

        # for the datasets we want to consider
        for dataset in ["ticam", "sviro"]:

            # make sure its a numpy array
            dataset_accuracies = np.array(block["results"][dataset])

            # get the mean and std
            maxs = np.max(dataset_accuracies)
            means = np.mean(dataset_accuracies)
            stds = np.std(dataset_accuracies)

            print(f"{block['label']} - {dataset}: \tMean max {means:.1f}% Std max - {stds:.1f} - Best max {maxs:.1f}%")

#####################################################################################################################################################

if __name__ == "__main__":

    # contains all the blocks of experiments to plot
    experiments = []

    experiments.append({
        "label":"Resnet-50-E-AE (Ours)",
        "data": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ]
    })

        
    # get the data for all experiments from the log files
    experiments = get_data_for_all_experiments(experiments)

    # calculate the mean performance
    calculate_accuracies(experiments)
