import os
import tqdm 
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import random as rm 
import numpy as np
import pickle

dataBase_dir="./database/Brocoli"
Categoria=["Bueno","Malo"]
TamañoImagen=100


# guardar la serializacion
output_dir = "./src/data"  
rutaImagenPickle=os.path.join(output_dir, "imagenes.pickle")
rutaEtiquetaPickle=os.path.join(output_dir, "etiquetas.pickle")

def Ganerar_datos_serializados():
    data=[]##arreglo de los datos
     ##analiza la informacion, le asigna un valor de 0 y 1(solo son dos carpetas)
     ##con todo y su directorio de iamgenes y su ruta
    for categoria in Categoria:
        ruta=os.path.join(dataBase_dir,categoria)
        valor=Categoria.index(categoria)
        lista_directorios=os.listdir(ruta)
        # print ("path:" ,ruta,"///"," valor:", valor,"lista_directorios",lista_directorios)
        for i in tqdm(range(len(lista_directorios)),desc=categoria):
            nombreImagen=lista_directorios[i]
            try:
                rutaImagen=os.path.join(ruta,nombreImagen)
                imagen= cv2.imread(rutaImagen,cv2.IMREAD_GRAYSCALE)
                imagen= cv2.resize(imagen,(TamañoImagen,TamañoImagen))
                data.append([imagen,valor])
            except Exception as e:
                 pass
    rm.shuffle(data)
    imagenes_x=[]
    etiqueta_y=[]
    for i in tqdm(range(len(data)),desc="procesamiento"):
        par=data[i]
        imagenes_x.append(par[0])
        etiqueta_y.append(par[1])
    
    imagenes_x=np.array(imagenes_x).reshape(-1,TamañoImagen,TamañoImagen,1)
    
    salida_pickle=open(rutaImagenPickle,"wb")
    pickle.dump(imagenes_x,salida_pickle)
    salida_pickle.close
    print("ya se creo la serializacion de las imagenes")
    
    salida_pickle=open(rutaEtiquetaPickle,"wb")
    pickle.dump(etiqueta_y,salida_pickle)
    salida_pickle.close
    print("ya se creo la serializacion de las etiquetas")
    
        
        
if __name__=="__main__":
    Ganerar_datos_serializados()