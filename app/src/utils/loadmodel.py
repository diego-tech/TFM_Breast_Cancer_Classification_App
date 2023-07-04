import os
from tensorflow import keras

densenet_model_path = os.path.join(os.getcwd(), 'model_data', 'DenseNet201_Model_30.h5')
densenet_model = keras.models.load_model(densenet_model_path)

resnet50_model_path = os.path.join(os.getcwd(), 'model_data', 'ResNet50__30.h5')
resenet50_model = keras.models.load_model(resnet50_model_path)

classic_cnn_path = os.path.join(os.getcwd(), 'model_data', 'ClassicCNN_0010_20.pth')