import os
from tensorflow import keras

model_path = os.path.join(os.getcwd(), 'model_data', 'DenseNet201_Model_10.h5')
densenet_model = keras.models.load_model(model_path)