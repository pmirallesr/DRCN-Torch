""" The 3 modules that make up a DRCN network: A labeller head,
an autoencoder head, and a common encoder body for both """
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision


class Encoder(nn.Module):
    """Encoder common to Autoencoder and labeller"""

    def __init__(self, input_channels, dropout_chance=0.5, dense_layer_neurons=300):
        """Initialize DomainRegressor."""
        super(Encoder, self).__init__()

        # Size Parameters

        conv1_filters = 100
        conv1_kernel_size = 5
        max_pool1_size = (2, 2)

        conv2_filters = 150
        conv2_kernel_size = 5
        max_pool2_size = (2, 2)

        conv3_filters = 200
        conv3_kernel_size = 3

        # Placeholder ranges
        fc4_output_dim = dense_layer_neurons
        fc5_output_dim = dense_layer_neurons

        # Convolutional Layers Size Calculations
        conv1_input_channels = input_channels
        conv2_input_channels = conv1_filters
        conv3_input_channels = conv2_filters

        # Convolutional Layers
        self.conv1 = nn.Conv2d(
            conv1_input_channels, conv1_filters, conv1_kernel_size, padding=2
        )
        self.maxPool2D1 = nn.MaxPool2d(max_pool1_size)
        self.conv2 = nn.Conv2d(
            conv2_input_channels, conv2_filters, conv2_kernel_size, padding=2
        )
        self.maxPool2D2 = nn.MaxPool2d(max_pool2_size)
        self.conv3 = nn.Conv2d(
            conv3_input_channels, conv3_filters, conv3_kernel_size, padding=1
        )

        # 8 is the final h x w dimension of the tensors when passed to the dense layer
        fc4_input_dim = conv3_filters * 8 * 8
        fc5_input_dim = fc4_output_dim

        # Fully connected Layers
        self.fc4 = nn.Linear(fc4_input_dim, fc4_output_dim)
        self.dropout4 = nn.Dropout2d(p=dropout_chance)

        self.fc5 = nn.Linear(fc5_input_dim, fc5_output_dim)
        self.dropout5 = nn.Dropout2d(p=dropout_chance)

    def forward(self, x):
        """Forward pass X and return probabilities of source and domain."""
        x = F.relu(self.conv1(x.float()))
        x = self.maxPool2D1(x)

        x = F.relu(self.conv2(x))
        x = self.maxPool2D2(x)

        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)
        return x


class Labeller(nn.Module):
    """ The labeller part of the network is constituted by
    the common Encoder plus a labelling fully connected layer"""

    def __init__(self, encoder, n_classes=10):
        """
        Args:
            encoder: The encoder used in this DRCN network
            n_classes: The number of output labels
        """
        super(Labeller, self).__init__()
        self.encoder = encoder
        # As many in features as the previous layer's out features
        self.fcOUT = nn.Linear(self.encoder.fc5.out_features, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.fcOUT(x)


class Autoencoder(nn.Module):
    """The autoencoder is constituted by the Encoder common to
    the labeller and itself, and a decoder part that is a mirror
    image of the Encoder

    Layers 6 and 7 are FC layers, layers 8 through 10 are (de)convolutional layers

    """

    def __init__(self, encoder):
        """
        Initialize DomainRegressor.
        Args
            encoder: The encoder used in this DRCN network
        """
        super(Autoencoder, self).__init__()

        self.encoder = encoder

        # Layers
        self.fc6 = nn.Linear(
            self.encoder.fc5.out_features, self.encoder.fc5.in_features
        )
        self.fc7 = nn.Linear(
            self.encoder.fc4.out_features, self.encoder.fc4.in_features
        )

        # Layer 8 is an extra layer in the author's github implementation wrt the paper.
        self.deconv8 = nn.Conv2d(
            self.encoder.conv3.out_channels,
            self.encoder.conv3.out_channels,
            self.encoder.conv3.kernel_size,
            padding=1,
        )
        self.deconv9 = nn.Conv2d(
            self.encoder.conv3.out_channels,
            self.encoder.conv3.in_channels,
            self.encoder.conv3.kernel_size,
            padding=1,
        )
        self.upsample9 = nn.Upsample(scale_factor=2, mode="nearest")
        self.deconv10 = nn.Conv2d(
            self.encoder.conv2.out_channels,
            self.encoder.conv2.in_channels,
            self.encoder.conv2.kernel_size,
            padding=2,
        )
        self.upsample10 = nn.Upsample(scale_factor=2, mode="nearest")
        self.deconv11 = nn.Conv2d(
            self.encoder.conv1.out_channels,
            self.encoder.conv1.in_channels,
            self.encoder.conv1.kernel_size,
            padding=2,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        x = torch.reshape(x, (x.shape[0], self.deconv8.in_channels, 8, 8))
        x = F.relu(self.deconv8(x))
        x = F.relu(self.deconv9(x))
        x = self.upsample9(x)
        x = F.relu(self.deconv10(x))
        x = self.upsample10(x)
        x = self.deconv11(x)

        return x
