version 1.0
fecha->29/09/2024
-------VERSION DE PYTHON--------
se necesita version de python 3.12.5  -> https://www.python.org/downloads/release/python-3125/
-------DEPENDENCIAS Y LIBRERIAS-------
absl-py==2.1.0
astunparse==1.6.3
certifi==2024.7.4
charset-normalizer==3.3.2
colorama==0.4.6
contourpy==1.2.1
cycler==0.12.1
flatbuffers==24.3.25
fonttools==4.53.1
gast==0.6.0
google-pasta==0.2.0
grpcio==1.66.0
h5py==3.11.0
idna==3.8
keras==3.5.0
kiwisolver==1.4.5
libclang==18.1.1
Markdown==3.7
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.9.2
mdurl==0.1.2
ml-dtypes==0.3.2
namex==0.0.8
numpy==1.26.4
opencv-python==4.10.0.84
opencv-python-headless==4.10.0.84
opt-einsum==3.3.0
optree==0.12.1
packaging==24.1
pillow==10.4.0
protobuf==4.25.4
Pygments==2.18.0
pyparsing==3.1.4
python-dateutil==2.9.0.post0
requests==2.32.3
rich==13.8.0
setuptools==73.0.1
six==1.16.0
tensorboard==2.17.1
tensorboard-data-server==0.7.2
tensorflow==2.17.0
tensorflow-intel==2.17.0
termcolor==2.4.0
tqdm==4.66.5
typing_extensions==4.12.2
urllib3==2.2.2
Werkzeug==3.0.4
wheel==0.44.0
wrapt==1.16.0

------ORDEN DE EJECUCION-------------------
1-serializarDatos  -> nos permite generar que cada carpeta , y cada imagen sean serializadas y sean faciles de usar por el modelo
2-Entrenamiento (CnnFit)-> permite entrenar la red neuronal Cnn, con varias capas, en la capa 12, con un total de 200 fotos analizadas empieza a taner problemas y falla al sesgar serializarDatos
3-insertar imagen-> en el archivo es un arreglo de prueba de varias imagenes, que para la cantidad tan pqueña que tenemos, se modificara para las imagenes de prueba

4-falta por trabajar en las demas versiones de las redes neuronales, pero solo seria entrenar los datos serializados, y crear los modelos

