import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
import base64

def preprocess_image(image_path):
    # Cargar la imagen utilizando la biblioteca PIL
    image = Image.open(image_path)
    
    # Redimensionar o recortar la imagen a 50x50 píxeles
    image = image.resize((50, 50))
    
    # Convertir la imagen a modo RGB
    image = image.convert('RGB')
    
    # Verificar la forma de la imagen
    if image.size != (50, 50):
        # Recortar la imagen si es necesario
        left = (image.width - 50) // 2
        top = (image.height - 50) // 2
        right = left + 50
        bottom = top + 50
        image = image.crop((left, top, right, bottom))
    
    # Convertir la imagen a un array NumPy
    image_array = np.array(image)
    
    # Normalizar los valores de los píxeles entre 0 y 1
    image_array = image_array / 255.0
    return image_array

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
    new_width = 400
    new_height = int(new_width / aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    
    # Encode the resized image as Base64
    buffered = BytesIO()
    resized_image.save(buffered, format="JPEG", quality=95)
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Return the Base64 encoded image
    return encoded_image