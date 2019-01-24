r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq, temperature = 'Once upon a time,', .05
    # ========================
    return start_seq, temperature


part1_q1 = r"""
There is no reason to learn from a very long sequence of characters. For eample the 1000th character is not related to the 10th in the corpus.
The characters are related to the sentence in wich they are part of, and this means that only the last few characters are relevant to the prediction.

"""

part1_q2 = r"""
The generated text seems to have memory that is longer then the squence length because of the hidden leyer that is passed between the samples in the batch and between batches. It learns the connection between the characters and have longer memory then the size of one batch.
"""

part1_q3 = r"""
The order of batches is represents the form of the text. Because of the relations that we disscused in q2 we must keep the order of the batches as it is in the text.

"""

part1_q4 = r"""
We can see that high temperture results with a distribution that is closer to uniform over the dictionary and a lower temp results in a distribution wich is closer to 1 for the character with highest probability.
When training we want to make sure to test all possible options and train accordingly so we use higher temprature and when sampling we want the outcome to represent the data learned as closely as possible so we will use lower temp. 

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['h_dim'] = 1024
    hypers['z_dim'] = 2
    hypers['x_sigma2'] = 5
    hypers['learn_rate'] = 5e-4
    hypers['betas'] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
The x_sigma2 hyperparameter is in charge of the relation between the data loss and the KL divergence loss in the calculation of the loss funtion.
For high values the weight of the data loss will be small and the KL div loss will be high, and vice versa.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size']=32
    hypers['z_dim']=10
    hypers['data_label']=1
    hypers['label_noise']=0.3
    hypers['discriminator_optimizer']['type']='Adam'
    hypers['discriminator_optimizer']['weight_decay']=0.002
    hypers['discriminator_optimizer']['betas']=(0.5,0.999)
    hypers['discriminator_optimizer']['lr']=0.0002
    hypers['generator_optimizer']['type']='Adam'
    hypers['generator_optimizer']['weight_decay']=0.002
    hypers['generator_optimizer']['betas']=(0.4,0.999)
    hypers['generator_optimizer']['lr']=0.0002
    # ========================
    return hypers


part3_q1 = r"""
We maintain the gradient when we sample from the generator in the batch training function.
The reason is that when we are training we want the gradient so that we can optimize the result of the generator. In all other occasions we do not maintain the gradient so that we will not change its value and ruine the training.

"""

part3_q2 = r"""
We can't decide to stop training based on the generator loss being bellow a certain threshold because the loos of the generator and the loss of the descriminator are connected. We can imagine a situation where the loss of the generator is very low (so we would think to stop training) but in the next batch the descriminator will sudenlly improve and find new differences between the real and fake images therefore the loss of the generator will go back up. If we get into a situation where the loss of the descriminator is constant but the loos of the generator keeps improving then we are in a situation where the descriminator can no longer tell the difference between the real and fake images but the generator keeps making the images better and better in comparison to the real ones.

"""

part3_q3 = r"""
The main difference between the results from the VAE and from the GAN is that the with the VAE we get a clear face with almost no background but with good details and with the GAN we get a better overall image in terms of the background and the face in it but with less fine details.
We think that the it is happaning because of the way the models are built, optimized and trained.
The VAE tries to extract the most importent and reoccuring features and convert them to a representation in the latent space, that is why we get a good result for the fine details of the face.
In comparison the GAN tries to optimize the generated image in relation to its ability to distinguish between a real and fake image, therefore it produces images that as a whole look similar to the dataset but the fine details are less importent.

"""

# ==============


