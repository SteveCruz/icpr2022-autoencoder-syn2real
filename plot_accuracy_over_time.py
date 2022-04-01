##############################################################################################################################################################
##############################################################################################################################################################
"""
During training, the accuracy is saved to a log file for each checkpoint step (frequency is defined in the config file).
This files uses the accuracies of all evaluations, for different training runs and plots the results to obtain the figure from the paper.

You will need to manually insert the experiments which you want to use in the plot.
Replace the label and insert the folder names (needs to be located inside the results folder) of the following part of the code.
You can have several of such plots: they will be plotted and compared in the same figure.

experiments.append({
        "label":"Resnet-50-II-E-TAE (Ours)",
        "style" : "-",
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
##############################################################################################################################################################
##############################################################################################################################################################

import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
plt.style.use(['seaborn-white', 'seaborn-paper'])
plt.rc('font', family='serif')
plt.rc('axes', labelsize=8)
plt.rc('font', size=8)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)
plt.rc('figure', figsize=[9, 4.5])

##############################################################################################################################################################
##############################################################################################################################################################

def get_data_from_log(experiment):

    # get and set the folders to the experiment
    result_folder = Path("results") / experiment 
    log_file = result_folder / "logs" / "accuracy.log"

    # open the accuracy log file and read all lines
    with log_file.open(mode="r") as file:
        accuracies = file.read().splitlines()

    # make sure all the values are float
    # also make sure we have accuracies in percentage
    accuracies = [float(x)*100 for x in accuracies]

    # make sure its a numpy array
    accuracies = np.array(accuracies)

    return accuracies

##############################################################################################################################################################

def get_data_for_all_experiments(experiments):

    # for each block
    for block in experiments:

        # add a list, which will contain the data read from the log files
        block["results"] = []

        # for each experiment inside this block
        for experiment in block["data"]:

            if experiment == "":
                continue

            # read the data and append it to the current block's list
            block["results"].append(get_data_from_log(experiment))

    return experiments

##############################################################################################################################################################

def plot_result(experiments, limit_to_nbr_epochs):

    # create the figure
    plt.figure()

    # for each block containint results for several runs
    for block in experiments:

        # limit the data to the number of epochs specified
        block["results"] = [x[0:limit_to_nbr_epochs] for x in block["results"]]

        # the epochs will be the indices
        epochs = list(range(1, len(block["results"][0])+1))

        # make sure its a numpy array
        block["results"] = np.array(block["results"])

        # get the mean and std per epoch
        means = np.mean(block["results"], axis=0)
        stds = np.std(block["results"], axis=0)

        # plot the means
        plt.plot(epochs, means, label=block["label"], linewidth=2, linestyle=block["style"])

        # plot the error band around the mean with std
        plt.fill_between(epochs, means-stds, means+stds, alpha=0.25)

        # also compute the best performances
        best = np.max(block["results"], axis=1)
        mean_best = np.mean(best)
        std_best  = np.std(best)
        best_best  = np.max(best)
        print(f"Performance - {block['label']}: Mean max {mean_best:.1f}% Std max - {std_best:.1f} - Best max {best_best:.1f}%")

        # make sure that we do not have duplicates in best
        # in case that we used twice the same seed for training
        if len(best) != len(set(best)):
            print(best)
            raise Exception("The values are not unique!")

    # labelling and so on
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy - Mean and std [%]")
    plt.grid(axis="y", color="0.9", linestyle='-', linewidth=1)
    plt.legend()
    plt.box(False)

    # plot it
    plt.savefig("result_plot_accuracy_over_time.pdf", bbox_inches='tight')
    print("Figure has been saved to the main folder: result_plot_accuracy_over_time.pdf")

##############################################################################################################################################################

if __name__ == "__main__":

    # contains all the blocks of experiments to plot
    experiments = []
    experiments.append({
        "label":"Resnet-50-II-E-TAE (Ours)",
        "style" : "-",
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

    # limit the plot to a specified number of epochs
    limit_to_nbr_epochs = 250

    # get the data for all experiments from the log files
    experiments = get_data_for_all_experiments(experiments)

    # plot the result for all the experiments
    plot_result(experiments, limit_to_nbr_epochs)

