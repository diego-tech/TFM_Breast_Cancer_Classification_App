import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def denseNet201_Predict(data, model):
    result = {}
    input_array = np.expand_dims(data, axis=0)
    
    predictions = model.predict(input_array)

    label_names = ['Evil', 'Benign']

    predicted_label = label_names[0] if predictions > 0.5 else label_names[1]

    result = {
        'prediction_average': predictions.tolist(),
        'predicted_label': predicted_label
    }

    return result


def resnet50_Predict(data, model):
    result = {}
    input_array = np.expand_dims(data, axis=0)

    predictions = model.predict(input_array)

    label_names = ['Evil', 'Benign']

    predicted_label = label_names[0] if predictions > 0.5 else label_names[1]

    result = {
        'prediction_average': predictions.tolist(),
        'predicted_label': predicted_label
    }

    return result


class Model(nn.Module):
    def __init__(self, num_neurons):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, num_neurons, 3)
        self.conv2 = nn.Conv2d(num_neurons, num_neurons*2, 3)
        self.conv3 = nn.Conv2d(num_neurons*2, num_neurons*4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # Ajustar el tamaño de entrada de acuerdo a las salidas de las capas anteriores
        self.fc1 = nn.Linear(num_neurons*8*8, num_neurons)
        self.fc2 = nn.Linear(num_neurons, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Cambiar la forma a un tensor de tamaño batch_size x (num_neurons*4*3*3)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def classicCNN_Predict(data, model_path):
    result = {}

    model = Model(num_neurons=30)

    torch.save(model, model_path)

    loaded_model = torch.load(model_path)

    input_array = np.expand_dims(data, axis=0)

    with torch.no_grad():
        inputs = torch.from_numpy(input_array).permute(
            0, 3, 1, 2).type(torch.FloatTensor)
        predictions = loaded_model(inputs)

    label_names = ['Evil', 'Benign']

    predicted_label = label_names[0] if predictions > 0.5 else label_names[1]

    result = {
        'prediction_average': predictions.numpy().tolist(),
        'predicted_label': predicted_label
    }

    return result