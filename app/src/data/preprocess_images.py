import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
import base64
import cv2

def preprocess_image(image_path):
    image = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), cv2.IMREAD_COLOR)

    if image is not None:
        image = cv2.resize(image, (50, 50))

        return image
    
    return None

def processed_classic_cnn_image_function(image_path):
    image = Image.open(image_path)

    # Redimensionar o recortar la imagen a 50x50 píxeles
    image = image.resize((50, 50))
    
    # Convertir la imagen a modo RGB
    image = image.convert('RGB')

    # Convertir la imagen en un array numpy
    image_array = np.array(image)

    # Crear una instancia del escalador MinMaxScaler
    scaler = MinMaxScaler()

    # Reshape para que la imagen sea 2D
    image_2d = image_array.reshape(1, -1)

    # Aplicar el escalado a la imagen
    image_scaled = scaler.fit_transform(image_2d)

    # Reshape de nuevo a las dimensiones originales de la imagen
    image_scaled = image_scaled.reshape(*image_array.shape)

    return image_scaled

def rescale_image(image_path):
    # Open the image using PIL
    image = Image.open(image_path)
    
    # Rescale the image while preserving its aspect ratio and increasing quality
    width, height = image.size
    aspect_ratio = width / height
    new_width = 600
    new_height = int(new_width / aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Encode the resized image as Base64
    buffered = BytesIO()
    resized_image.save(buffered, format="JPEG", quality=95)
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Return the Base64 encoded image
    return encoded_image