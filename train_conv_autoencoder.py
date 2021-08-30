import matplotlib
matplotlib.use("Agg")

from core.convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", type=int, default=8, help="# Number of samples to visualize when decoding")
ap.add_argument("-o", "--output", type=str, default="output.png", help="Path to output visualization file")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="Path to output plot file")
args = vars(ap.parse_args())

EPOCHS = 10
BS = 32

print("[INFO] Loading MNIST Dataset...")
(x_train, _), (x_test, _) = mnist.load_data()

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("[INFO] Building Autoencoder...")
encoder, decoder, autoencoder = ConvAutoencoder.build(width=28, height=28, depth=1)
opt = Adam(lr=1e-3)
autoencoder.compile(loss="mse", optimizer=opt)

H = autoencoder.fit(x_train, x_train, validation_data=(x_test, x_test), epochs=EPOCHS, batch_size=BS)

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

print("[INFO] Making predictions...")
decoded = autoencoder.predict(x_test)
outputs = None

for i in range(0, args["samples"]):
    original = (x_test[i] * 255).astype("uint8")
    recon = (decoded[i] * 255).astype("uint8")
    
    output =np.hstack([original, recon])
    
    if outputs is None:
        outputs = output
    else:
        outputs = np.vstack([outputs, output])
        
cv2.imwrite(args["output"], outputs)
