from numpy import load, zeros, ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model, Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot


def define_discriminator(image_shape):

    # Initialize weights
    init = RandomNormal(stddev=0.02)

    # Source image input
    in_src_image = Input(shape=image_shape)

    # Target iamge input
    in_target_image = Ipnut(shape=image_shape)

    # Concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])

    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(
        merged
    )
    d = LeakyReLU(alpha=0.2)(d)

    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # Second last output layer
    d = Conv2D(512, (4, 4), strides=(2, 2), kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # Patch output
    d = Conv2D(1, (4, 4), padding="same", kernel_initializer=init)(d)
    patch_out = Activation("sigmoid")(d)

    # Define model
    model = Model([in_src_image, in_target_image], patch_out)

    # Compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, loss_weights=[0.5])

    return model


def define_generator(image_shape=(256, 256, 3)):

    # Initialize weights
    init = RandomNormal(stddev=0.02)

    # Image input
    in_image = Input(shape=image_shape)  # Encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)

    # Bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(e7)
    b = Activation("relu")(b)

    # Decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)

    # Output
    g = Conv2DTranspose(
        3, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
    )(d7)
    out_image = Activation("tanh")(g)

    # Define model
    model = Model(in_image, out_image)

    return model


def define_encoder_block(layer_in, n_filters, batchnorm=True):

    # Initialize weights
    init = RandomNormal(stddev=0.02)

    # add downsampling layer
    g = Conv2D(
        n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
    )(layer_in)

    # Conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)

    g = LeakyReLU(alpha=0.2)(g)

    return g


def decoder_block(layer_in, skip_in, n_filters, dropout=True):

    # Initialize weights
    init = RandomNormal(stddev=0.02)

    # Add unsampling layer
    g = Conv2DTranspose(
        n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
    )(layer_in)

    # Add batch normalization
    g = BatchNormalization()(g, training=True)

    # Conditionally add dropout
    if dropout:
        g = Dropout(0.5, training=True)

    # Merge with skip connection
    g = Concatenate()([g, skip_in])

    # Relu activation
    g = Activation("relu")(g)

    return g


def define_gan(g_model, d_model, image_shape):

    # Make weights in the discriminator not trainable
    d_model.trainable = False

    # Define the source image
    in_src = Input(shape=image_shape)

    # Connect the source iamge to the generator input
    gen_out = g_model(in_src)

    # Connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])

    # Srcv image as input, generated image and classification out
    model = Model(in_src, [dis_out, gen_out])

    # Compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(
        loss=["binary_crossentropy", "mae"], optimizer=opt, loss_weights=[1, 100]
    )

    return model


def load_real_samples(filename):

    # Load compressed arrays
    data = load(filename)

    # Unpack araays
    X1, X2 = data["arr_0"], data["arr_1"]

    # Scale from [1, 255] to [-1, 1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5

    return [X1, X2]


def generate_real_samples(dataset, n_samples, patch_shape):

    # Unpack dataset
    trainA, trainB = dataset

    # choose random instances
    ix = rndint(0, trainA[0], n_samples)

    # Retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]

    # generate `real` class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))

    return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):

    # Generate fake instance
    X = g_model.predict(samples)

    # Create `fake` class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))

    return X, y


def summarize_performance(step, g_model, dataset, n_samples=3):

    # Select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)

    # Generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)

    # Scale all pizels from [-1, 1] to [0, 1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0

    # Plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis("off")
        pyplot.imshow(X_realA[i])

    # Plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis("off")
        pyplot.imshow(X_fakeB[i])

    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples * 2 + i)
        pyplot.axis("off")
        pyplot.imshow(X_realB[i])

    filename1 = f"plot_{step+1}.png"

    # Save plot to file
    pyplot.savefig(filename1)
    pyplot.close()

    # Save the generator model
    filename2 = f"model_{step+1}.h5"

    g_model.save(filename2)

    print("> Saved: {filename1} and {filename2}")


def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):

    # Determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]

    # Unpack dataset
    trainA, trainB = dataset

    # Calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA), n_batch)

    # Calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs

    for i in range(n_steps):

        # Select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)

        # Generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

        # Update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)

        # Update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

        # Update the generator
        g_loss, _, _ = gen_model.train_on_batch(X_realA, [y_real, X_realB])

        # SUmmarize performance
        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)


def load_images(path, size=(256, 512)):
    src_list, tar_list = [], []

    for filename in listdir(path):

        # Load and resize the images
        pixels = load_img(path + filename, target_size=size)

        # Convert to np array
        pixels = img_to_array(pixels)

        # split into satellite and map
        sat_img, map_img = pixels[:, :256], pixels[:, 256:]

        src_list.append(sat_img)
        tar_list.append(map_img)

    return [asarray(src_list), asarray(tar_list)]


def condense_images():
    path = "maps/train/"

    [src_images, tar_images] = load_images(path)

    print("Loaded:", src_images.shape, tar_images.shape)

    filename = "maps_256.npz"

    savez_compressed(filename, src_images, tar_images)

    print("Saved dataset:", filename)


def main():

    # Load image data
    dataset = load_real_samples("maps_256.npz")
    print("Loaded", dataset[0].shape, dataset[1].shape)

    # Define input shape based on loaded dataset
    image_shape = dataset[0].shape[1:]

    # Define the models
    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)

    # Define the composite model
    gan_model = define_gan(g_model, d_model, image_shape)

    # Train model
    train(d_model, g_model, gen_model, dataset)


if __name__ == "__main__":
    main()
