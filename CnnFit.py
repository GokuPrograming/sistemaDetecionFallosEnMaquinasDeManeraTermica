# entrenar el modelo con Cnn

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import os
from SerializarDatos import TamañoImagen, Categoria  # Asumo que tienes un archivo GenerarDatos.py

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


neuronas = [32, 64, 128]
densas = [0, 1, 2]
convpoo = [1, 2, 3]
drop = [0]

#importar los datos serializados
imagenes=pickle.load(open("./src/data/imagenes.pickle","rb"))
etiquetas=pickle.load(open("./src/data/etiquetas.pickle","rb"))
#255 es blanco 0 es negro, se normaliza a [0,1], manejando los datos en decimal 0.1, 0.6 etc
print (imagenes)
imagenes = imagenes / 255.0
etiquetas=np.array(etiquetas)
print("imagenes",imagenes)
print("etiquetas", etiquetas)

def entrenar():
    if not os.path.exists("models"):
        os.makedirs("models")
    for neurona in neuronas:
        print('ejecutando a neurosnas',neurona)
        for conv in convpoo:
            print('ejecutando las convpoo', conv)
            for densa in densas:
                print('ejecutando densas',densa)
                for d in drop:
                    print('va el drop',drop)
                    nombreModelo="RedConv-n{}-cl{}-d{}-dropout{}".format(neurona,conv,densa,d)
                    tensorboard=TensorBoard(log_dir='logs/{}'.format(nombreModelo))
                    
                    model=Sequential()
                    model.add(Conv2D(64,(3,3),input_shape=(TamañoImagen,TamañoImagen,1)))
                    model.add(Activation("relu"))
                    model.add(MaxPooling2D(pool_size=(2,2)))
                    if d == 1:
                        print('llego a 1 el drop',d)
                        model.add(Dropout(0.2))
                        
                    for i in range(conv):
                        model.add(Conv2D(64, (3, 3),input_shape=(TamañoImagen,TamañoImagen,1)))
                        model.add(Activation("relu"))
                        model.add(MaxPooling2D(pool_size=(2, 2)))
                    print('el valor de i',i)
                    model.add(Flatten())

                    for i in range(densa):
                        model.add(Dense(neurona))
                        model.add(Activation("relu"))

                    model.add(Dense(1))
                    model.add(Activation('sigmoid'))
                    model.compile(loss="binary_crossentropy",
                                  optimizer="adam",
                                  metrics=['accuracy'])
                    model.fit(imagenes,etiquetas, batch_size=30, epochs=13, validation_split=0.3, callbacks=[tensorboard])
                    model.save("models/{}.keras".format(nombreModelo))  
entrenar()