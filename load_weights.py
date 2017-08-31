import tensorflow as tf
import numpy as np

def load_weights(layer_name):
    name = "./numpy_out/"+layer_name+".py"
    data = np.load(name)
    data = data.transpose()
    return data