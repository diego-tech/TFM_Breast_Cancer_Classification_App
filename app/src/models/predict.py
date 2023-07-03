from tensorflow import keras
import numpy as np

def denseNet201_Predict(data, model):
    input_array = np.expand_dims(data, axis=0)

    predictions = model.predict(input_array)

    label_names = ['Evil', 'Benign'] 

    predicted_label = label_names[np.argmax(predictions)]

    return predictions
    