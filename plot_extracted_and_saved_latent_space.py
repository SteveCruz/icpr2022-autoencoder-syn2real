##############################################################################################################################################################
##############################################################################################################################################################
"""
This script uses cuml, which needs to be installed using conda (they do not provide pip install): https://rapids.ai/start.html
To run this script, we need to activate conda.
Then simply run this script as usual

This script projects a previously extracted latent space and plots the results.
"""
##############################################################################################################################################################
##############################################################################################################################################################

import numpy as np
from pathlib import Path
from cuml import UMAP, TSNE, PCA

import matplotlib.pyplot as plt
plt.style.use(['seaborn-white', 'seaborn-paper'])
plt.rc('font', family='serif')
plt.rc('axes', labelsize=8)
plt.rc('font', size=8)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)
plt.rc('figure', figsize=[4.5, 4.5])

# reproducibility
SEED = 42
np.random.seed(SEED)

##############################################################################################################################################################        
##############################################################################################################################################################        

def plot_train_against(experiment, vs, which_model):

    print(f"Plot against {vs}: ", experiment)

    # ---------------------------------------------------------------------
    # Load the data
    # ---------------------------------------------------------------------

    # get and set the folders to the experiment
    result_folder = Path("results") / experiment 
    latent_folder = result_folder / "data" / "latent" 
    save_folder = result_folder / "images" / "latent" / f"train_vs_{vs}"
    save_folder.mkdir(exist_ok=True, parents=True)

    # load the mu from the numpy file
    mu = {
        "train": np.load(latent_folder / f"train_{which_model}_mu.npy"),
        vs: np.load(latent_folder / f"{vs}_{which_model}_mu.npy"),
    }

    # combine training and test mus
    together_mu = np.concatenate((mu["train"], mu[vs]))

    # ---------------------------------------------------------------------
    # Embedding 
    # ---------------------------------------------------------------------

    # init umap
    # hash_input = consistent behavior between calling model.fit_transform(X) and calling model.fit(X).transform(X)
    # use a random seed, though UMAP cannot be completely deterministic
    reducers = {
        "tsne2": TSNE(n_components=2, random_state=SEED),
    }

    # for each projection method
    for name, reducer in reducers.items():

        # project the data into the two dimensional space
        together_embedding = reducer.fit_transform(together_mu) 

        # split the data again in test and train
        train_embedding = together_embedding[0:mu["train"].shape[0]]
        vs_embedding = together_embedding[mu["train"].shape[0]:]

        # ---------------------------------------------------------------------
        # Plot 
        # ---------------------------------------------------------------------

        # create a figure
        fig, ax = plt.subplots(1,1, dpi=300)

        # scatterplot 
        ax.scatter(train_embedding[:, 0], train_embedding[:, 1], alpha=0.95, s=17, edgecolor='white', linewidth=0.5, rasterized=True)
        vs_sc = ax.scatter(vs_embedding[:, 0], vs_embedding[:, 1], alpha=0.95, s=17, edgecolor='white', linewidth=0.5, rasterized=True, marker="X")

        # adjust the space between the marker and the text
        # and the vertical offset of the marker to center with text
        plt.grid()

        # remove the ticks from the plot as they are meaningless
        ax.axis('scaled')
        plt.grid(True, color="0.9", linestyle='-', linewidth=1)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.box(False)
    
        filename = f'{name}_train_vs_{vs}_mu'
        fig.savefig(save_folder / (filename + ".pdf"), dpi=fig.dpi, bbox_inches='tight')

        if "pca" in name:
            print(f"{name} - Explained variance ratio by PCA: {100*np.array(reducer.explained_variance_ratio_).sum():.2f}% - {reducer.explained_variance_ratio_}")

##############################################################################################################################################################        

if __name__ == '__main__':

    # list of all the experiments to test
    experiments = [

    ]

    # best or last model
    which_model = "last"
    # which_model = "best"

    # the vehicle for which we want to use the latent space representation
    vehicles = ["mpi3d"]

    # for each experiment
    for experiment in experiments:
        for vehicle in vehicles:
            plot_train_against(experiment, vs=vehicle, which_model=which_model)
   
