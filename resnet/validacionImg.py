import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

ruta_imagen = './Bueno1.jpg'
ruta_modelo ='./modelo_resnet_bueno_malo.keras'
img_size = (100, 100)

def predecir_imagen(ruta_modelo, ruta_imagen):
    # Cargar el modelo
    model = tf.keras.models.load_model(ruta_modelo)
    
    # Cargar la imagen
    img = image.load_img(ruta_imagen, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0  # Escalar la imagen
    img_array = np.expand_dims(img_array, axis=0)  # Añadir batch dimension
    
    # Hacer la predicción
 
    prediction = model.predict(img_array)
    print(prediction)
    classes = ['Malo', 'Bueno']
    print('La predicción es =', classes[np.argmax(prediction)])

# Llamar a la función para predecir
predecir_imagen(ruta_modelo, ruta_imagen)
