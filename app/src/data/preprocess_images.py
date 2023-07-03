import numpy as np
from PIL import Image

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