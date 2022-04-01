##############################################################################################################################################################
##############################################################################################################################################################
"""
Autoencoder model definitions are located inside this script.
"""
##############################################################################################################################################################
##############################################################################################################################################################

import torch
from torch import nn
import torchvision
from pytorch_msssim import SSIM

##############################################################################################################################################################
##############################################################################################################################################################

def get_activation_function(model_config):

    # from the torch.nn module, get the activation function specified 
    # in the config. make sure the naming is correctly spelled 
    # according to the torch name
    if hasattr(nn, model_config["activation"]):
        activation = getattr(nn, model_config["activation"])
        return activation()
    else:
        raise ValueError("Activation function does not exist.")

##############################################################################################################################################################

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        # As an example, we print the shape here.
        print(x.shape)
        return x

##############################################################################################################################################################

def create_ae_model(model_config, data_config, training_config):

    # define the autoencoder model
    if model_config["type"] == "extractor":
        return ExtractorAE(model_config, data_config, training_config)
    elif model_config["type"] == "conv":
        return ConvAE(model_config, data_config, training_config)
    elif model_config["type"] == "vae":
        return VAE(model_config, data_config, training_config)
    elif model_config["type"] == "factorvae":
        return FactorVAE(model_config, data_config, training_config)

##############################################################################################################################################################

class BaseAE(nn.Module):
    def __init__(self, model_config, data_config, training_config, print_model=True):
        super().__init__()

        # some definitions
        self.type =  model_config["type"]
        self.variation =  model_config["variation"]
        self.nbr_input_channels = 3 if data_config["name"] in ["mpi3d"] else 1
        self.reshape_channels = 4 if data_config["name"] in ["mpi3d"] else 8
        self.dimension_before_fc = 64*self.reshape_channels*self.reshape_channels 

        # latent space dimension
        self.latent_dim = model_config["dimension"]

        # define the activation function to use
        self.activation = get_activation_function(model_config)      

        # define the loss function to use
        self.which_loss = training_config["loss"]
        self._init_loss(data_config, print_model)

    def _define_decoder(self):

        # the decoder is the same for all models
        decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2),
            self.activation,
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, padding=1, stride=2),
            self.activation,
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            self.activation,
            nn.ConvTranspose2d(in_channels=32, out_channels=self.nbr_input_channels, kernel_size=4, padding=1, stride=2),
            nn.Sigmoid()
        )

        return decoder

    def print_model(self):

        print("=" * 57)
        print("The Encoder is defined as: ")
        print("=" * 57)
        print(self)
        print("=" * 57)
        print("Parameters of the model to learn:")
        print("=" * 57)
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                print(name)
        print('=' * 57)

    def _init_loss(self, data_config, print_model):

        # define the loss function to use

        if self.which_loss == "MSE":
            self.criterion = nn.MSELoss(reduction="sum")

        elif self.which_loss == "L1":     
            self.criterion = nn.L1Loss(reduction="sum")

        elif self.which_loss == "BCE":     
            self.criterion = nn.BCELoss(reduction="sum")

        elif self.which_loss == "SSIM":     
            self.criterion = SSIM(data_range=1.0, size_average=True, channel=1 if data_config["name"].lower() in ["ticam", "sviro", "seats_and_people_nocar", "sviro_illumination", "mnist", "fonts"] else 3)
        
        else:
            raise ValueError("Loss definition does not exist.")

        # init triplet loss
        self._triplet_loss_fn = nn.TripletMarginLoss(margin=0.2, p=2.0, swap=True, reduction='mean')

        if print_model:
            print("Loss function used: ", self.criterion)
            print('=' * 57)

    def _get_extractor(self, which_model, which_layer):

        # get a fixed number of blocks from the pre-trained network
        # make sure it is in eval mode
        if which_model == "vgg11":
            setup = {
                -13: {"dimension_before_fc": 64*22*22, "in_channels": 256},
                -11: {"dimension_before_fc": 64*22*22, "in_channels": 256},
                -8: {"dimension_before_fc": 64*8*8, "in_channels": 512},
                -6: {"dimension_before_fc": 64*8*8, "in_channels": 512},
                -3: {"dimension_before_fc": 256*4*4, "in_channels": 512},
                -1: {"dimension_before_fc": 256*4*4, "in_channels": 512},
            }
            blocks = list(torchvision.models.vgg11(pretrained=True).eval().features.children())[0:which_layer]

        elif which_model == "resnet50":
            setup = {
                -5: {"dimension_before_fc": 64*22*22, "in_channels": 256},
                -4: {"dimension_before_fc": 64*8*8, "in_channels": 512},
                -3: {"dimension_before_fc": 256*4*4, "in_channels": 1024},
            }
            blocks = list(torchvision.models.resnet50(pretrained=True).eval().children())[0:which_layer]
        
        elif which_model == "densenet121":
            setup = {
                -7: {"dimension_before_fc": 64*22*22, "in_channels": 256},
                -6: {"dimension_before_fc": 64*8*8, "in_channels": 128},
                -5: {"dimension_before_fc": 64*8*8, "in_channels": 512},
                -4: {"dimension_before_fc": 256*4*4, "in_channels": 256},
                -3: {"dimension_before_fc": 256*4*4, "in_channels": 1024},
            }
            blocks = list(torchvision.models.densenet121(pretrained=True).eval().features.children())[0:which_layer]

        # make the blocks iterable for inference
        extractor = torch.nn.Sequential(*blocks)

        # get the number of input channels for the layers after the extractor
        in_channels = setup[which_layer]["in_channels"]

        # get the number of dimensions as input to the fc
        dimension_before_fc = setup[which_layer]["dimension_before_fc"] 

        return extractor, dimension_before_fc, in_channels  

    def triplet_loss(self, anchor, positive, negative) : 

        loss = self._triplet_loss_fn(anchor=anchor, positive=positive, negative=negative)

        return loss

    def loss(self, prediction, target):

        if self.which_loss == "SSIM":   
            return 1-self.criterion(prediction, target)
        else:
            return self.criterion(prediction, target)

    def forward(self, x, **kwargs):

        mu = self.encode(x)
        z = self.decode(mu, **kwargs)

        return {"xhat":z, "mu": mu}

##############################################################################################################################################################

class ConvAE(BaseAE):
    def __init__(self, model_config, data_config, training_config, print_model=True):
        # call the init function of the parent class
        super().__init__(model_config, data_config, training_config, print_model)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.nbr_input_channels, out_channels=32, kernel_size=4, padding=1, stride=2),
            self.activation,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            self.activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=2),
            self.activation,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2),
            self.activation,
            nn.Flatten(),
            nn.Linear(in_features=self.dimension_before_fc, out_features=256, bias=True),
            self.activation,
            nn.Linear(in_features=256, out_features=self.latent_dim, bias=True)
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=256, bias=True),
            self.activation,
            nn.Linear(in_features=256, out_features=self.dimension_before_fc, bias=True),
            self.activation,
        )

        self.decoder_conv = self._define_decoder()

    def encode(self, x):

        mu = self.encoder(x)

        return mu

    def decode(self, mu):

        z = self.decoder_fc(mu)
        z = z.reshape(-1, 64, self.reshape_channels, self.reshape_channels)
        z = self.decoder_conv(z)

        return z

##############################################################################################################################################################

class ExtractorAE(BaseAE):
    def __init__(self, model_config, data_config, training_config, print_model=True):
        # call the init function of the parent class
        super().__init__(model_config, data_config, training_config, print_model)

        # get the feature extractor and some related parameters
        self.extractor, self.dimension_before_fc, self.in_channels = self._get_extractor(which_model=model_config["extractor"], which_layer=model_config["layer"])

        # for each of the extracted vgg blocks
        # we do not want to update the weights
        for param in self.extractor.parameters():
            param.requires_grad = False
            
        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=self.in_channels, out_channels=256, kernel_size=4, padding=0, stride=1),
        )

        self.encoder = nn.Sequential(
            self.activation,
            nn.Flatten(),
            nn.Linear(in_features=self.dimension_before_fc, out_features=256, bias=True),
            self.activation,
            nn.Linear(in_features=256, out_features=self.latent_dim, bias=True)
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=256, bias=True),
            self.activation,
            nn.Linear(in_features=256, out_features=64*self.reshape_channels*self.reshape_channels, bias=True),
            self.activation,
        )

        self.decoder_conv = self._define_decoder()

        # interpolate if images to small, i.e. not 224x224 as the images used for pre-training
        self.transform = torch.nn.functional.interpolate

    def encode(self, x):

        # if the input does not have 3 channels, i.e. grayscale
        # repeat the image along channel dimension
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)

        # resize input
        if x.shape[2] != 224:
            x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)

        # extract features by pre-trained network
        x = self.extractor(x)
        
        # encode the extracted features
        x = self.conv(x)
        x = self.encoder(x)

        return x

    def decode(self, mu):

        # decode the extracted features
        z = self.decoder_fc(mu)
        z = z.reshape(-1, 64, self.reshape_channels, self.reshape_channels)
        z = self.decoder_conv(z)

        return z

##############################################################################################################################################################

class VAE(BaseAE):
    def __init__(self, model_config, data_config, training_config, print_model=True):
        # call the init function of the parent class
        super().__init__(model_config, data_config, training_config, print_model)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.nbr_input_channels, out_channels=32, kernel_size=4, padding=1, stride=2),
            self.activation,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2),
            self.activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=2),
            self.activation,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2),
            self.activation,
            nn.Flatten(),
            nn.Linear(in_features=self.dimension_before_fc, out_features=256, bias=True),
            self.activation,
        )

        self.encoder_fc21 = nn.Linear(in_features=256, out_features=self.latent_dim, bias=True)
        self.encoder_fc22 = nn.Linear(in_features=256, out_features=self.latent_dim, bias=True)

        self.decoder_fc = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=256, bias=True),
            self.activation,
            nn.Linear(in_features=256, out_features=self.dimension_before_fc, bias=True),
            self.activation,
        )
        
        self.decoder_conv = self._define_decoder()

    def kl_divergence_loss(self, mu, logvar):

        # KL divergence between gaussian prior and sampled vector
        # we need to sum over the latent dimension
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)

        # take the mean over the batch dimension
        kl_loss = kl_loss.mean(dim=0)

        return kl_loss

    def reparametrize(self, mu, logvar):
        
        # reparametrization rick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def encode(self, x):

        x = self.encoder(x)
        mu = self.encoder_fc21(x)
        logvar = self.encoder_fc21(x)

        return mu, logvar

    def decode(self, z):

        z = self.decoder_fc(z)
        z = z.reshape(-1, 64, self.reshape_channels, self.reshape_channels)
        z = self.decoder_conv(z)

        return z

    def sample(self, mu, logvar):

        # if we want to reparametrize and we are in training mode
        if self.training:
            z = self.reparametrize(mu, logvar)
        else:
            z = mu

        return z

    def forward(self, x):

        # encode the input image
        mu, logvar = self.encode(x)

        # sample the latent vector
        z = self.sample(mu, logvar)

        # decode the sampled latent vector
        xhat = self.decode(z)
        
        return {"xhat": xhat, "mu": mu, "logvar": logvar, "z":z}

##############################################################################################################################################################

class FactorVAE(VAE):
    """
    Inspired and code taken from: https://github.com/AntixK/PyTorch-VAE/blob/master/models/fvae.py
    """
    def __init__(self, model_config, data_config, training_config, print_model=True):
        # call the init function of the parent class
        super().__init__(model_config, data_config, training_config, print_model)

        self.discriminator = nn.Sequential(
            nn.Linear(self.latent_dim, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 2)
        )

    def permute_latent(self, z):
        """
        Permutes each of the latent codes in the batch
        """

        # batch and latent space dimension
        B, D = z.size()

        # Returns a shuffled inds for each latent code in the batch
        inds = torch.cat([(D *i) + torch.randperm(D) for i in range(B)])

        return z.view(-1)[inds].view(B, D)

##############################################################################################################################################################


