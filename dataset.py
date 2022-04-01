##############################################################################################################################################################
##############################################################################################################################################################
"""
Dataloader definitions for all the datasets used in our paper.
The datasets need to be downloaded manually and placed inside a same folder.
Specify your folder location in the following line

# directory containing all the datasets
ROOT_DATA_DIR = Path("")
"""
####################################################################################################################################################
####################################################################################################################################################

import cv2
import random
import numpy as np
import albumentations as album
from PIL import Image
from pathlib import Path
from skimage import exposure

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST as TMNIST

from matplotlib import pyplot as plt
plt.style.use(['seaborn-white', 'seaborn-paper'])
plt.rc('font', family='serif')

####################################################################################################################################################
####################################################################################################################################################

# directory containing all the datasets
ROOT_DATA_DIR = Path("")

####################################################################################################################################################

class BaseDatasetCar(Dataset):
    """
    Base class for all dataset classes.
    """
    def __init__(self, root_dir, car, split, use_triplet, make_scene_impossible, make_instance_impossible, get_all=False):

        # path to the main folder
        self.root_dir =  Path(root_dir)

        # which car are we using?
        self.car = car

        # train or test split
        self.split = split

        # should we use the triplet loss
        self.use_triplet = use_triplet

        # normal or impossible reconstruction loss?
        self.make_scene_impossible = make_scene_impossible
        self.make_instance_impossible = make_instance_impossible

        # get all images or just 1 per scenery?
        self.get_all = get_all

        # pre-process the data if necessary
        self._pre_process_dataset()

        # load the data into the memory
        self._get_data()
        self.string_labels_to_int()

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        # number of images to use
        return len(self.images)

    def _get_data(self):

        # get all folders with the sceneries
        if self.car.lower() == "all":
            self.folders = sorted(list(self.root_dir.glob("*/pp_*_128/*")))
        else:
            self.folders = sorted(list(self.root_dir.glob(f"{self.car}/pp_*_128/*")))

        # placeholder for all images and labels
        self.images = []
        self.labels = []

        # for each folder
        for idx, folder in enumerate(self.folders):

            # get classification labels for each seat from folder name
            classif_labels = self._get_classif_label(folder)

            # each scene will be an array of images
            if not self.get_all:
                self.images.append([])

            # get all the images for this scene
            files = sorted(list(folder.glob("*.png")))

            # for each file
            for file in files:
        
                # open the image specified by the path
                # make sure it is a grayscale image
                img = Image.open(file).convert("L")

                # append the image to the placeholder
                if not self.get_all:
                    self.images[idx].append(img)
                else:
                    self.images.append(img)
            
            # append label to placeholder
            if not self.get_all:
                self.labels.append(classif_labels)
            else:
                for _ in range(len(files)):
                    self.labels.append(classif_labels)

    def _get_classif_label(self, file_path):

        # get the filename only of the path
        name = file_path.stem

        # split at GT 
        gts = name.split("GT")[-1]
        
        # split the elements at _
        # first element is empty string, remove it
        clean_gts = gts.split("_")[1:]

        # convert the strings to ints
        clean_gts = [int(x) for x in clean_gts]

        # convert sviro labels to compare with other datasets
        for index, value in enumerate(clean_gts):
            # everyday objects to background
            if value == 4:
                clean_gts[index] = 0

        return clean_gts

    def string_labels_to_int(self):

        # create an empty dict
        self.string_labels_to_integer_dict = dict()

        # count the number of unique classes
        counter = 0

        # for each label in the dataset
        for x in self.labels:

            # get the 3 labels and make it a string
            x = str(x[0]) + "_" + str(x[1]) + "_" + str(x[2])

            # if this label has not been seen
            if x not in self.string_labels_to_integer_dict:

                # add it to the dict using the counter as integer value
                self.string_labels_to_integer_dict[x] = counter

                # increase the counter
                counter += 1

            # also take into account the flipped version
            x = x[::-1]

            # if this label has not been seen
            if x not in self.string_labels_to_integer_dict:

                # add it to the dict using the counter as integer value
                self.string_labels_to_integer_dict[x] = counter

                # increase the counter
                counter += 1

    def _pre_process_dataset(self):

        # get all the subfolders inside the dataset folder
        data_folder_variations = self.root_dir.glob("*")

        # for each variation
        for folder in data_folder_variations:

            # for each split
            for pre_processed_split in ["pp_train_128", "pp_test_128"]:

                # create the path
                path_to_preprocessed_split = folder / pre_processed_split
                path_to_vanilla_split = folder / pre_processed_split.split("_")[1]

                # if no pre-processing for these settings exists, then create them
                if not path_to_preprocessed_split.exists():

                    print("-" * 37)
                    print(f"Pre-process and save data for folder: {folder} and split: {pre_processed_split} and downscale size: 128 ...")
                    
                    self.pre_process_and_save_data(path_to_preprocessed_split, path_to_vanilla_split)
                    
                    print("Pre-processing and saving finished.")
                    print("-" * 37)

    def pre_process_and_save_data(self, path_to_preprocessed_split, path_to_vanilla_split):
        """
        To speed up training, it is beneficial to do the rudementary pre-processing once and save the data.
        """

        # create the folders to save the pre-processed data
        path_to_preprocessed_split.mkdir()

        # get all the files in all the subfolders 
        files = list(path_to_vanilla_split.glob(f"**/*.png"))

        # for each image 
        for curr_file in files:

            # open the image specified by the path
            img = Image.open(curr_file).convert("L")

            # center crop the image using the smaller size (i.e. width or height)
            # to define the new size of the image (basically we remove only either the width or height)
            img = TF.center_crop(img, np.min(img.size))

            # then resize the image to the one we want to use for training
            img = TF.resize(img, 128)

            # create the folder for the experiment
            save_folder = path_to_preprocessed_split / curr_file.parent.stem
            save_folder.mkdir(exist_ok=True)

            # save the processed image
            img.save(save_folder / curr_file.name)

    @staticmethod
    def differ_by_one(s1, s2):
        ok = False

        for c1, c2 in zip(s1, s2):
            if c1 != c2:
                if ok:
                    return False
                else:
                    ok = True
                    
        return ok    


    def _get_positive(self, rand_indices, positive_label, positive_images, flipped):

        # get all the potential candidates which have the same label 
        masked = [idx for idx, x in enumerate(self.labels) if x==positive_label] 

        # if there is no other image with the same label
        if not masked:
            
            new_rand_indices = random.sample(range(0,len(positive_images)), 2)
            positive_input_image = positive_images[new_rand_indices[0]]
            positive_output_image = positive_images[new_rand_indices[1]] if self.make_scene_impossible else positive_images[new_rand_indices[0]]
            positive_input_image =  TF.to_tensor(positive_input_image)
            positive_output_image =  TF.to_tensor(positive_output_image)
            if flipped:
                positive_input_image = TF.hflip(positive_input_image)
                positive_output_image = TF.hflip(positive_output_image)

        else:
            # choose one index randomly from the masked subset
            index = np.random.choice(masked)

            positive_input_image = self.images[index][rand_indices[0]]
            positive_output_image = self.images[index][rand_indices[1]] if self.make_scene_impossible else self.images[index][rand_indices[0]]
            positive_input_image =  TF.to_tensor(positive_input_image)
            positive_output_image =  TF.to_tensor(positive_output_image)

        return positive_input_image, positive_output_image


    def _get_negative(self, rand_indices, positive_label):

        # the labels should only differ by one, e.g. 3_0_2 -> 3_1_2
        # moreover, we do not want completely empty seat, i.e. 0_0_0, because thats too easy
        masked = [idx for idx, x in enumerate(self.labels) if self.differ_by_one(x, positive_label) and not all(v==0 for v in x)] 

        # choose one index randomly from the masked subset
        index = np.random.choice(masked)

        negative_input_image = self.images[index][rand_indices[0]]
        negative_output_image = self.images[index][rand_indices[1]] if self.make_scene_impossible else self.images[index][rand_indices[0]]
        negative_input_image =  TF.to_tensor(negative_input_image)
        negative_output_image =  TF.to_tensor(negative_output_image)

        return negative_input_image, negative_output_image, self.labels[index]

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        images = self.images[index]
        label = self.labels[index]

        if self.get_all:

            # make it a tensor
            input_image = TF.to_tensor(images)

            # randomly decide to flip the image
            flipped = random.choice([True, False])

            # flip the label if we flipped the image
            if flipped:
                input_image = TF.hflip(input_image)
                label = label[::-1]

            return {"image":input_image, "target":input_image, "gt_left": label[0], "gt_middle": label[1], "gt_right": label[2]}

        # randomly selected
        # .) the input images 
        # .) the output images 
        rand_indices = random.sample(range(0,len(images)), 2)

        # get the image to be used as input
        input_image = images[rand_indices[0]]

        # get the image to be used for the reconstruction error
        output_image = images[rand_indices[1]] if self.make_scene_impossible else images[rand_indices[0]]

        # make sure its a tensor
        input_image =  TF.to_tensor(input_image)
        output_image =  TF.to_tensor(output_image)

        # randomly decide to flip the image
        flipped = random.choice([True, False])

        # flip the label if we flipped the image
        if flipped:
            input_image = TF.hflip(input_image)
            output_image = TF.hflip(output_image)
            label = label[::-1]

        if self.make_instance_impossible:
            _, output_image = self._get_positive(rand_indices, label, images, flipped)

        # if we use the triplet loss, then we need some additional images
        if self.use_triplet:

            # get positive and negative samples w.r.t. anchor label
            if self.make_instance_impossible:
                _, output_image = self._get_positive(rand_indices, label, images, flipped)
                positive_input_image, _ = self._get_positive(rand_indices, label, images, flipped)
                _, positive_output_image = self._get_positive(rand_indices, label, images, flipped)
                negative_input_image, _, negative_label = self._get_negative(rand_indices, label)
                _, negative_output_image = self._get_positive(rand_indices, negative_label, images, flipped)
            else:
                positive_input_image, positive_output_image = self._get_positive(rand_indices, label, images, flipped)
                negative_input_image, negative_output_image, _ = self._get_negative(rand_indices, label)

            return {"image":input_image, "target":output_image, "positive":positive_input_image, "positive_target":positive_output_image, "negative":negative_input_image, "negative_target":negative_output_image, "gt_left": label[0], "gt_middle": label[1], "gt_right": label[2]}

        else:

            return {"image":input_image, "target":output_image, "gt_left": label[0], "gt_middle": label[1], "gt_right": label[2]}

####################################################################################################################################################

class TICAM(BaseDatasetCar):
    """
    https://vizta-tof.kl.dfki.de/

    
    You only need the IR images.
    Make sure to have a folder structure as follows:

    TICaM 
    ├── train
    │   ├── cs00
    │   ├── cs01
    │   ├── cs02
    │   ├── cs03
    │   └── cs04
    └── test
        ├── cs01
        ├── cs02
        ├── cs03
        ├── cs04
        └── cs05
    """
    def __init__(self, use_triplet):

        # path to the main folder
        root_dir = ROOT_DATA_DIR / "TICaM" 

        # call the init function of the parent class
        super().__init__(root_dir=root_dir, car="all", split="all", use_triplet=use_triplet, make_scene_impossible=False, make_instance_impossible=False)

    def _get_data(self):

        # get all png images
        self.files = sorted(list(self.root_dir.glob("pp_*_128/**/*.png")))

        # placeholder for all images and labels
        self.images = []
        self.labels = []

        # for each file
        for counter, file in enumerate(self.files):

            # get classification labels for each seat from file name
            classif_labels = self._get_classif_label(file)

            # each scene will be an array of images
            self.images.append([])
        
            # open the image specified by the path
            img = Image.open(file).convert("L")

            # append the image to the placeholder
            self.images[counter].append(img)

             # append label to placeholder
            self.labels.append(classif_labels)

    def _get_classif_label(self, file_path):

        # get the filename only of the path
        name = file_path.stem

        # split the elements at _
        file_name_parts = name.split("_")

        # get the two entries responsible for the classification of the scenery
        # add 0 label in the middle to be consistent with the other datasets
        # left, middle (not existing), right
        gts = [file_name_parts[-6], 0, file_name_parts[3]]

        # convert the string definitions into integers
        for idx, value in enumerate(gts):
            if value == 0:
                continue
            # adult passenger
            elif "p" in value:
                gts[idx] = 3
            # everyday objects
            elif "o" in value:
                gts[idx] = 0
            # infant seat
            elif value in ["s05", "s15", "s06", "s16"]:
                gts[idx] = 1
            # child seat
            elif value in ["s03", "s13", "s04", "s14"]:
                gts[idx] = 2
            elif value in ["s01", "s11", "s02", "s12"]:
                if file_name_parts[-3] == "g00":
                    gts[idx] = 2
                elif file_name_parts[-3] == "g01" or file_name_parts[-3] == "g11" or file_name_parts[-3] == "g10":
                    gts[idx] = 1
        
        # convert the strings to ints
        gts = [int(x) for x in gts]

        return gts

    def _pre_process_dataset(self):

        # for each split
        for pre_processed_split in ["pp_train_128", "pp_test_128"]:

            # create the path
            path_to_preprocessed_split = self.root_dir / pre_processed_split
            path_to_vanilla_split = self.root_dir / pre_processed_split.split("_")[1]

            # if no pre-processing for these settings exists, then create them
            if not path_to_preprocessed_split.exists():

                print("-" * 37)
                print(f"Pre-process and save data for folder: {self.root_dir} and split: {pre_processed_split} and downscale size: 128 ...")
                
                self.pre_process_and_save_data(path_to_preprocessed_split, path_to_vanilla_split)
                
                print("Pre-processing and saving finished.")
                print("-" * 37)

    def pre_process_and_save_data(self, path_to_preprocessed_split, path_to_vanilla_split):
        """
        To speed up training, it is beneficial to do the rudementary pre-processing once and save the data.
        """

        # create the folders to save the pre-processed data
        path_to_preprocessed_split.mkdir()

        # get all the files in all the subfolders 
        files = list(path_to_vanilla_split.glob(f"**/*.png"))

        # for each image 
        for curr_file in files:

            # open the image specified by the path
            img = cv2.imread(str(curr_file), -1) 

            # histogram equalization
            img = exposure.equalize_hist(img)

            # make it a tensor
            img = torch.from_numpy(img)

            # center crop the image using the smaller size (i.e. width or height)
            # to define the new size of the image (basically we remove only either the width or height)
            # img = TF.center_crop(img, 300)
            img = TF.crop(img, top=120, left=(512-300)//2, height=300, width=300)

            # then resize the image to the one we want to use for training
            img = TF.resize(img.unsqueeze(0), 128)

            # make it a pil again
            img = TF.to_pil_image(img)

            # create the folder for the experiment
            save_folder = path_to_preprocessed_split / curr_file.parent.stem
            save_folder.mkdir(exist_ok=True)

            # save the processed image
            img.save(save_folder / curr_file.name)
        
    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index][0]
        label = self.labels[index]

        # transform it for pytorch (normalized and transposed)
        image =  TF.to_tensor(image)

        return {"image":image, "target":image, "gt_left": label[0], "gt_middle": label[1], "gt_right": label[2]}

####################################################################################################################################################

class SVIRO(BaseDatasetCar):
    """
    https://sviro.kl.dfki.de

    You only need the grayscale images for the whole scene.
    Make sure to have a folder structure as follows:
    
    SVIRO 
    ├── aclass
    │   ├── train
    │   │   └──── grayscale_wholeImage
    │   └── test
    │       └──── grayscale_wholeImage
    ⋮
    ⋮
    ⋮
    └── zoe
        ├── train
        │   └──── grayscale_wholeImage
        └── test
            └──── grayscale_wholeImage
    """
    def __init__(self, car, use_triplet):

        # path to the main folder
        root_dir = ROOT_DATA_DIR / "SVIRO"  

        # call the init function of the parent class
        super().__init__(root_dir=root_dir, car=car, split="all", use_triplet=use_triplet, make_scene_impossible=False, make_instance_impossible=False)
        
    def _get_data(self):

        # get all the png files, i.e. experiments
        if self.car.lower() == "all":
            self.files = sorted(list(self.root_dir.glob("*/train/grayscale_wholeImage_pp_640_128/*.png")))
        else:
            self.files = sorted(list(self.root_dir.glob(f"{self.car}/train/grayscale_wholeImage_pp_640_128/*.png")))

        # placeholder for all images and labels
        self.images = []
        self.labels = []

        # for each file
        for file in self.files:
        
            # get classification labels for each seat from folder name
            classif_labels = self._get_classif_label(file)
            if 5 in classif_labels or 6 in classif_labels:
                continue

            # open the image specified by the path
            # make sure it is a grayscale image
            img = Image.open(file).convert("L")

            # each scene will be an array of images
            # append the image to the placeholder
            self.images.append([img])
        
            # append label to placeholder
            self.labels.append(classif_labels)

    def _pre_process_dataset(self):

        # get all the subfolders inside the dataset folder
        data_folder_variations = self.root_dir.glob("*/*")

        # for each variation
        for folder in data_folder_variations:

            # create the path
            path_to_preprocessed_split = folder / "grayscale_wholeImage_pp_640_128"
            path_to_vanilla_split = folder / "grayscale_wholeImage"

            # if no pre-processing for these settings exists, then create them
            if not path_to_preprocessed_split.exists():

                print("-" * 37)
                print(f"Pre-process and save data for folder: {folder} and downscale size: 128 ...")
                
                self.pre_process_and_save_data(path_to_preprocessed_split, path_to_vanilla_split)
                
                print("Pre-processing and saving finished.")
                print("-" * 37)

    def pre_process_and_save_data(self, path_to_preprocessed_split, path_to_vanilla_split):
        """
        To speed up training, it is beneficial to do the rudementary pre-processing once and save the data.
        """

        # create the folders to save the pre-processed data
        path_to_preprocessed_split.mkdir()

        # get all the files in all the subfolders 
        files = list(path_to_vanilla_split.glob("*.png"))

        # for each image 
        for curr_file in files:

            # open the image specified by the path
            img = Image.open(curr_file).convert("L")

            # center crop the image using the smaller size (i.e. width or height)
            # to define the new size of the image (basically we remove only either the width or height)
            img = TF.center_crop(img, np.min(img.size))

            # then resize the image to the one we want to use for training
            img = TF.resize(img, 128)

            # create the path to the file
            save_path = path_to_preprocessed_split / curr_file.name

            # save the processed image
            img.save(save_path)

    def _get_negative(self, positive_label):

        # get all the potential candidates from the real images which have the same label as the synthetic one
        masked = [idx for idx, x in enumerate(self.labels) if self.differ_by_one(x, positive_label)] 

        # choose one index randomly from the masked subset
        index = np.random.choice(masked)
        
        input_image = self.images[index][0]
        input_image =  TF.to_tensor(input_image)

        return input_image, self.labels[index] 

    def _get_positive(self, positive_label):

        # get all the potential candidates from the real images which have the same label as the synthetic one
        masked = [idx for idx, x in enumerate(self.labels) if x==positive_label] 

        # choose one index randomly from the masked subset
        index = np.random.choice(masked)

        input_image = self.images[index][0]
        input_image =  TF.to_tensor(input_image)

        return input_image


    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.images[index][0]
        label = self.labels[index]

        # transform it for pytorch (normalized and transposed)
        image =  TF.to_tensor(image)
    
        # randomly decide to flip the image
        flipped = random.choice([True, False])

        # if we use the triplet loss, then we need some additional images
        if self.use_triplet:

            positive_image = self._get_positive(label)
            negative_image, _ = self._get_negative(label)

            # flip the label if we flipped the image
            if flipped:
                label = label[::-1]
                image = TF.hflip(image)
                positive_image = TF.hflip(positive_image)
                negative_image = TF.hflip(negative_image)

            return {"image":image, "target":image, "positive":positive_image, "positive_target":positive_image, "negative":negative_image, "negative_target":negative_image, "gt_left": label[0], "gt_middle": label[1], "gt_right": label[2]}

        else:

            # flip the label if we flipped the image
            if flipped:
                label = label[::-1]
                image = TF.hflip(image)

            return {"image":image, "target":image, "gt_left": label[0], "gt_middle": label[1], "gt_right": label[2]}

####################################################################################################################################################

class Seats_And_People_Nocar(BaseDatasetCar):
    """
    Download dataset from the anonymous link provided in the paper or in the README.md.

    Make sure to have a folder structure as follows:

    SEATS_AND_PEOPLE_NOCAR 
    └── cayenne
        ├── pp_train_128
        └── pp_test_128
    """
    

    def __init__(self, car, use_triplet,  make_scene_impossible, make_instance_impossible, get_all):

        # path to the main folder
        root_dir = ROOT_DATA_DIR / "SEATS_AND_PEOPLE_NOCAR"

        # call the init function of the parent class
        super().__init__(root_dir=root_dir, car=car, split="all", use_triplet=use_triplet, make_scene_impossible=make_scene_impossible, make_instance_impossible=make_instance_impossible, get_all=get_all)
        
####################################################################################################################################################

class SVIROIllumination(BaseDatasetCar):
    """
    https://sviro.kl.dfki.de
    
    Make sure to have a folder structure as follows:

    SVIRO-Illumination 
    ├── cayenne
    │   ├── train
    │   └── test
    ├── kodiaq
    │   ├── train
    │   └── test
    └── kona
        ├── train
        └── test
    """
    def __init__(self, car, use_triplet,  make_scene_impossible, make_instance_impossible, get_all):

        # path to the main folder
        root_dir = ROOT_DATA_DIR / "SVIRO-Illumination"

        # call the init function of the parent class
        super().__init__(root_dir=root_dir, car=car, split="all", use_triplet=use_triplet, make_scene_impossible=make_scene_impossible, make_instance_impossible=make_instance_impossible, get_all=get_all)

####################################################################################################################################################

class MPI3D(Dataset):
    """
    https://github.com/rr-learning/disentanglement_dataset

    Datasets: toy, realistic, real

    # shape             (1036800, 64, 64, 3)
    # reshape           (6,6,2,3,3,40,40,64,64,3)

    # object_color  	white=0, green=1, red=2, blue=3, brown=4, olive=5
    # object_shape 	    cone=0, cube=1, cylinder=2, hexagonal=3, pyramid=4, sphere=5
    # object_size 	    small=0, large=1
    # camera_height 	top=0, center=1, bottom=2
    # background_color 	purple=0, sea green=1, salmon=2
    # horizontal_axis 	0,...,39
    # vertical_axis 	0,...,39

    Make sure to have a folder structure as follows:

    MPI3D 
    ├── realistic.npz
    ├── toy.npz
    └── real.npz
    """
    def __init__(self, which_dataset, which_factor, use_triplet, should_augment):

        # path to the main folder
        self.root_dir = ROOT_DATA_DIR / "MPI3D"

        # which dataset is being used
        self.which_dataset = which_dataset

        # select which dataset to use
        self.data_dir = self.root_dir / f"{self.which_dataset}.npz"
     
        # which factors to use from the dataset
        self.which_factor = which_factor

        # do we need to sample triplets?
        self.use_triplet = use_triplet
        if self.use_triplet:
            raise ValueError("The triplet loss has not been implemented for MPI3D yet.")

        # load the data as images to the memory
        self._get_data()
        self.string_labels_to_int()

        self.should_augment = should_augment

        # we only need these transformation for training
        if self.should_augment:
            if self.should_augment:
                self.augment = album.Compose([
                    album.GaussNoise(),    
                    album.ISONoise(),  
                ], p=1.0)

    def _get_data(self):

        # load the numpy array from the file
        self.images = np.load(self.data_dir)['images']
        self.images = self.images.reshape(6,6,2,3,3,40,40,64,64,3)
        self.reshaped_images_size = self.images.shape

        # load the factors we need
        if self.which_factor == "reduced":
            self.images = self.images[:,:,1,:,:,:,:,:,:,:] # get only large and camera center images, background_color sea green
            self.reshaped_images_size = self.images.shape
            
        self.images = self.images.reshape(-1, 64, 64, 3)

    def string_labels_to_int(self):

        # empty dict
        self.string_labels_to_integer_dict = dict()

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image
        input_image = self.images[index]
        output_image =  TF.to_tensor(input_image)

        # augment input image if needed
        if self.should_augment:
            input_image = self.augment(image=np.array(input_image))['image']
        
        # make sure its a tensor
        input_image =  TF.to_tensor(input_image)

        return {"image":input_image, "target":output_image}

####################################################################################################################################################

class MNIST(TMNIST):

    string_labels_to_integer_dict = dict()

    def __init__(self, which_factor, use_triplet, make_instance_impossible):

        # path to the main folder
        root_dir = Path(f"/data/local_data/workingdir_g02/sds/data/")

        # train or test split
        self.split = which_factor
        self.is_train = True if self.split.lower() == "train" else False

        self.use_triplet = use_triplet

        # normal or impossible reconstruction loss?
        self.make_instance_impossible = make_instance_impossible

        # call the init function of the parent class
        super().__init__(root=root_dir, train=self.is_train, download=True)

    def _get_classif_str(self, label):
        return int(label)

    def _get_positive(self, positive_label):

        while True:
            index = random.randint(0, len(self.targets)-1)
            if int(self.targets[index]) == positive_label:
                image = self.data[index]
                image = Image.fromarray(image.numpy(), mode='L')
                image = TF.resize(image, [128, 128])
                image = TF.to_tensor(image)

                return image

    def _get_negative(self, positive_label):

        while True:
            index = random.randint(0, len(self.targets)-1)
            if int(self.targets[index]) != positive_label:
                image = self.data[index]
                image = Image.fromarray(image.numpy(), mode='L')
                image = TF.resize(image, [128, 128])
                image = TF.to_tensor(image)

                return image, self.targets[index]

    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = self.data[index]
        label = int(self.targets[index])

        # doing this so that it is consistent with all other datasets to return a PIL Image
        image = Image.fromarray(image.numpy(), mode='L')

        # transform it for pytorch (normalized and transposed)
        image = TF.resize(image, [128, 128])
        image = TF.to_tensor(image)

        if self.make_instance_impossible:
            output_image = self._get_positive(label)
        else:
            output_image = image.clone()

        if self.use_triplet:

            positive_input_image = self._get_positive(label)
            negative_input_image, negative_label = self._get_negative(label)

            if self.make_instance_impossible:
                positive_output_image = self._get_positive(label)
                negative_output_image = self._get_positive(negative_label)
            else:
                positive_output_image = positive_input_image.clone()
                negative_output_image = negative_input_image.clone()

            return {"image":image, "target":output_image, "positive":positive_input_image, "positive_target":positive_output_image, "negative":negative_input_image, "negative_target":negative_output_image, "gt": label}

        else:

            return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

class FONTS(Dataset):

    string_labels_to_integer_dict = dict()

    def __init__(self):

        # path to the main folder
        root_dir = Path(f"/data/local_data/workingdir_g02/sds/data/") / "REAL_FONTS" / "digits"

        self.images = list(root_dir.glob("*/*.png"))

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        # number of images to use
        return len(self.images)


    def __getitem__(self, index):
        """
        Return an element from the dataset based on the index.

        Parameters:
            index -- an integer for data indexing
        """

        # get the image and labels
        image = Image.open(self.images[index]).convert("L")
        label = int(self.images[index].parent.name)

        # transform it for pytorch (normalized and transposed)
        image = TF.resize(image, [128, 128])
        image = TF.to_tensor(image)

        output_image = image.clone()

        return {"image":image, "target":output_image, "gt":label}

####################################################################################################################################################

def wif(id):
    """
    np.random generates the same random numbers for each data batch
    https://github.com/pytorch/pytorch/issues/5059
    """
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))

####################################################################################################################################################

def create_dataset(which_dataset, which_factor, use_triplet, should_augment, make_scene_impossible=False, make_instance_impossible=False, batch_size=64, shuffle=True, get_all=False, print_dataset=True):

    # create the dataset
    if which_dataset.lower() == "ticam":
        dataset = TICAM(use_triplet=use_triplet)
    elif which_dataset.lower() == "sviro":
        dataset = SVIRO(car=which_factor, use_triplet=use_triplet)
    elif which_dataset.lower() == "sviro_illumination":
        dataset = SVIROIllumination(car=which_factor, use_triplet=use_triplet, make_scene_impossible=make_scene_impossible, make_instance_impossible=make_instance_impossible, get_all=get_all)
    elif which_dataset.lower() == "seats_and_people_nocar":
        dataset = Seats_And_People_Nocar(car="cayenne", use_triplet=use_triplet, make_scene_impossible=make_scene_impossible, make_instance_impossible=make_instance_impossible, get_all=get_all)
    elif which_dataset.lower() == "mpi3d":
        dataset = MPI3D(which_dataset=which_factor, which_factor="reduced", use_triplet=use_triplet, should_augment=should_augment)
    elif which_dataset.lower() == "mnist":
        dataset = MNIST(which_factor=which_factor, use_triplet=use_triplet, make_instance_impossible=make_instance_impossible)
    elif which_dataset.lower() == "fonts":
        dataset = FONTS()
    else:
        raise ValueError(f"The dataset {which_dataset} does not exist.")

    if len(dataset) == 0:
        raise ValueError("The length of the dataset is zero. There is probably a problem with the folder structure for the dataset you want to consider. Have you downloaded the dataset and used the correct folder name and folder tree structure?")

    # create loader for the defined dataset
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4, 
        pin_memory=True,
        worker_init_fn=wif
    )

    if print_dataset:
        print("=" * 37)
        print("Dataset used: \t", dataset)
        print("Samples: \t", len(dataset))
        print("=" * 37)

    return train_loader

####################################################################################################################################################
