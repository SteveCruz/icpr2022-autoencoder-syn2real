##############################################################################################################################################################
##############################################################################################################################################################
"""
Defines the predefined classification models using torchvision.
The script is generic such that we can easily use different predefined models.
You can also use pre-trained weights provided by torchvision.
Finally, the number of layers to be frozed is generic as well.
"""
##############################################################################################################################################################
##############################################################################################################################################################

import torch
import torch.nn as nn
from torchvision import models

##############################################################################################################################################################
##############################################################################################################################################################

def set_parameter_requires_grad(model, freeze_n_layers):
    """
    :param model:
    The model we want to use.

    Taken from:
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """

    # for each parameter in the model
    for idx_layer, param in enumerate(model.parameters()):

        # set the tracking of the gradient to false
        param.requires_grad = False

        # if we have fixed the number of layers we wanted, then
        # we can leave the function
        if idx_layer == freeze_n_layers-1:
            return

##############################################################################################################################################################


def initialize_model(model_name, num_classes, use_pretrained=True, freeze_n_layers=False):
    """
    Initialize these variables which will be set in this if statement.
    Each of these variables is model specific.

    :param model_name:
    The name of the model we want to load.

    :param num_classes:
    The number of classes of our datasets.

    :param use_pretrained:
    Should we load the weights pre-traiend on Imagenet?

    Taken from:
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """

    # define the variable to contain the model
    model = None

    # several models have different input sizes.
    # save the correct one and output it.
    input_size = 0

    if model_name == "resnet18":
        """
        Resnet18
        """
        model = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "resnet34":
        """
        Resnet34
        """
        model = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    # --------------------------------------------------------------------------

    elif model_name == "resnet50":
        """
        Resnet50
        """
        model = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    # --------------------------------------------------------------------------

    elif model_name == "resnet101":
        """
        Resnet101
        """
        model = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    # --------------------------------------------------------------------------

    elif model_name == "resnet152":
        """
        Resnet152
        """
        model = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "alexnet":
        """
        Alexnet
        """
        model = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "vgg11":
        """
        VGG11
        """
        model = models.vgg11(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224    
    # --------------------------------------------------------------------------

    elif model_name == "vgg11_bn":
        """
        VGG11_bn
        """
        model = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    # --------------------------------------------------------------------------

    elif model_name == "vgg13_bn":
        """
        VGG13_bn
        """
        model = models.vgg13_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    # --------------------------------------------------------------------------

    elif model_name == "vgg16_bn":
        """
        VGG16_bn
        """
        model = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    # --------------------------------------------------------------------------

    elif model_name == "vgg19_bn":
        """
        VGG19_bn
        """
        model = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "squeezenet1_0":
        """
        Squeezenet 1.0
        """
        model = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
        input_size = 224
    # --------------------------------------------------------------------------

    elif model_name == "squeezenet1_1":
        """
        Squeezenet 1.1
        """
        model = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "mobilenet_v2":
        """
        Mobilenet v2
        """
        model = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "densenet121":
        """
        Densenet 121
        """
        model = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "densenet161":
        """
        Densenet 161
        """
        model = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "densenet169":
        """
        Densenet 169
        """
        model = models.densenet169(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "densenet201":
        """
        Densenet 201
        """
        model = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "inception":
        """
        Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    # --------------------------------------------------------------------------

    elif model_name == "resnext50":
        """
        ResNext-50 32x4d
        """
        model = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "resnext101":
        """
        ResNext-101 32x8d
        """
        model = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "shufflenet_v2_x0_5":
        """
        ShuffleNetV2 with 0.5x output channels
        """
        model = models.shufflenet_v2_x0_5(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "shufflenet_v2_x1_0":
        """
        ShuffleNetV2 with 1.0x output channels
        """
        model = models.shufflenet_v2_x1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "shufflenet_v2_x1_5":
        """
        ShuffleNetV2 with 1.5x output channels
        """
        model = models.shufflenet_v2_x1_5(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # --------------------------------------------------------------------------

    elif model_name == "shufflenet_v2_x2_0":
        """
        ShuffleNetV2 with 2.0x output channels
        """
        model = models.shufflenet_v2_x2_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model, freeze_n_layers)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    # --------------------------------------------------------------------------

    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size

##############################################################################################################################################################

class PretrainedClassifier(nn.Module):
    def __init__(self, model_config, nbr_classes, print_model=False):
        super().__init__()

        # number of predictions by the classifier
        self.nbr_of_classif_outputs = nbr_classes 
      
        # Get the pre-trained model
        # freeze the specified number of first layers
        # adapt the model so that it outputs the correct number of classes
        self.pre_trained_model, _ = initialize_model(
            model_name = model_config["architecture"],
            num_classes = self.nbr_of_classif_outputs,
            use_pretrained = model_config["use_pretrained"],
            freeze_n_layers = model_config["freeze_n_first_layers"]
        )

        # print the model
        if print_model:
            self._print_model()

        # interpolate if images to small, i.e. not 224x224 as the images used for pre-training
        self.transform = torch.nn.functional.interpolate

    def _print_model(self):

        print("=" * 57)
        print("The Encoder is defined as: ")
        print("=" * 57)
        print(self.pre_trained_model)
        print("=" * 57)
        print("Parameters of the model to learn:")
        print("=" * 57)
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                print(name)
        print('=' * 57)

    def forward(self, x):

        # if the input does not have 3 channels, i.e. grayscale
        # repeat the image along channel dimension
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)

        # resize input
        if x.shape[2] != 224:
            x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)

        x = self.pre_trained_model(x)

        return x

##############################################################################################################################################################

def classification_loss(outputs, targets):

    # by default, the mean is taken along the batch dimension
    # The input is expected to contain raw, unnormalized scores for each class.
    # expects a class index in the range [0,C−1] as the target
    criterion = nn.CrossEntropyLoss()

    single_label_loss = criterion(outputs, targets)

    return single_label_loss

##############################################################################################################################################################