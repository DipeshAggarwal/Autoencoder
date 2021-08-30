from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

class ConvAutoencoder():
    
    @staticmethod
    def build(width, height, depth, filters=(32, 64), latent_dim=16):
        input_shape = (height, width, depth)
        chan_dim = -1
        
        inputs = Input(shape=input_shape)
        x = inputs
        
        for f in filters:
            x = Conv2D(f, (3, 3), strides=(2, 2), padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chan_dim)(x)
            
        volume_size = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latent_dim)(x)
        
        encoder = Model(inputs, latent, name="encoder")
        
        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(np.prod(volume_size[1:]))(latent_inputs)
        x = Reshape((volume_size[1], volume_size[2], volume_size[3]))(x)
        
        for f in filters[::-1]:
            x = Conv2DTranspose(f, (3, 3), strides=(2, 2), padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chan_dim)(x)
        
        x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
        outputs = Activation("sigmoid")(x)
        
        decoder = Model(latent_inputs, outputs, name="decoder")
        autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
        
        return (encoder, decoder, autoencoder)
        