from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .autoencoder import EncoderCNN, DecoderCNN


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        # To extract image features you can use the EncoderCNN from the VAE
        # section or implement something new.
        # You can then use either an affine layer or another conv layer to
        # flatten the features.
        # ====== YOUR CODE: ======
        count_pool = 0
        #K = [64, 128, 256, 512]
        K = [250, 500, 750, 1000]
        modules = []
        for in_c, out_c in zip([self.in_size[0]] + K, K):
            #modules += [nn.Conv2d(in_c, out_c, 3, padding=1), 
            #            nn.BatchNorm2d(out_c),
            #            nn.ReLU(),
            #            nn.MaxPool2d(2)]
            modules += [nn.Conv2d(in_c, out_c, 4, padding=1, stride=2), 
                        nn.BatchNorm2d(out_c),
                        nn.ReLU()]
            count_pool += 1
        self.cnn = nn.Sequential(*modules)
        
        h, w = self.in_size[1:]
        ds_factor = 2 ** count_pool
        h_ds, w_ds = h // ds_factor, w // ds_factor
        
        classifier_modules = [nn.Linear(h_ds * w_ds * K[-1], h_ds * w_ds // 4),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(h_ds * w_ds // 4, 1)]
        self.classifier = nn.Sequential(*classifier_modules)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (aka logits, not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        # No need to apply sigmoid to obtain probability - we'll combine it
        # with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        x_feat = self.cnn(x)
        x_feat = x_feat.view(x_feat.size(0), -1)
        y = self.classifier(x_feat)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        # To combine image features you can use the DecoderCNN from the VAE
        # section or implement something new.
        # You can assume a fixed image size.
        # ====== YOUR CODE: ======
        K = [250, 500, 750, 1000]
        #K = [512, 256, 128, 64]
        modules = []
        for in_c, out_c in zip([self.z_dim] + K, K + [out_channels]):
            modules += [nn.ConvTranspose2d(in_c, out_c, featuremap_size, padding=1 if in_c != self.z_dim else 0, stride=2), 
                        nn.ReLU(),        
                        nn.BatchNorm2d(out_c)]
#        first_layer = False
#        for in_channel, out_channel in zip([self.z_dim] + K, K + [out_channels]):
#            if not first_layer:
#                first_layer = True
#                padding = 0
#            else:
#                padding = 1

#            modules += 
#            [nn.ConvTranspose2d(in_channel, out_channel, featuremap_size, 2, padding, bias=False), nn.Tanh()]\
#                    if out_channel == out_channels \
#                    else [nn.ConvTranspose2d(in_channel, out_channel, featuremap_size, 2, padding, bias=False),
#                      nn.ReLU(), nn.BatchNorm2d(out_channel)]״״״
        self.generator = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should have
        gradients or not.
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        # Generate n latent space samples and return their reconstructions.
        # Don't use a loop.
        # ====== YOUR CODE: ======
        torch.autograd.set_grad_enabled(with_grad)
        z = torch.randn([n, self.z_dim], device=device, requires_grad=with_grad)
        samples = self.forward(z)
        torch.autograd.set_grad_enabled(True)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        # Don't forget to make sure the output instances have the same scale
        # as the original (real) images.
        # ====== YOUR CODE: ======
        z = torch.unsqueeze(z, dim=2)
        z = torch.unsqueeze(z, dim=3)
        x = self.generator(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO: Implement the discriminator loss.
    # See torch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    positive = torch.full(y_data.size(), data_label, device=y_data.device)
    negative = 1 - positive
    a, b = -label_noise / 2, label_noise / 2
    pos_noisy = positive + torch.distributions.uniform.Uniform(a, b).sample(positive.size()).to(positive.device)
    neg_noisy = negative + torch.distributions.uniform.Uniform(a, b).sample(negative.size()).to(negative.device)
    loss_data = F.binary_cross_entropy_with_logits(y_data, pos_noisy)
    loss_generated = F.binary_cross_entropy_with_logits(y_generated, neg_noisy)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    # TODO: Implement the Generator loss.
    # Think about what you need to compare the input to, in order to
    # formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    loss = F.binary_cross_entropy_with_logits(y_generated, torch.full(y_generated.size(), data_label, device=y_generated.device))
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    # 1. Show the discriminator real and generated data
    # 2. Calculate discriminator loss
    # 3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()
    real_batch = x_data
    fake_batch = gen_model.sample(x_data.shape[0], with_grad=True)
    real_res = dsc_model(real_batch)
    fake_res = dsc_model(fake_batch.detach())
    dsc_loss = dsc_loss_fn(real_res, fake_res)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    # 1. Show the discriminator generated data
    # 2. Calculate generator loss
    # 3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()
    res = dsc_model(fake_batch)
    gen_loss = gen_loss_fn(res)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()

