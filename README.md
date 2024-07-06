# Ejemplo de uso de la Api funcional de Tensorflow Keras

En este proyectos se incluye un pequeño tutorial para aprender a crear redes neuronales usando tensorflow-keras siguiendo
el uso de la api funcional.

Asimismo, se lleva a cabo un caso de estudio del precio de compra de una serie de inmuebles teniendo en cuenta 
tanto información descriptiva como imágenes. La idea de este proyecto no es evaluar la capacidad predictiva 
del modelo planteado sino la ilustración de poder crear redes neuronales con más de una entrada haciendo uso de 
la api funcional de Tensorflow-Keras.

Crea un ambiente conda con python 3.10

```
conda create --name tf_vivienda python=3.10
```

E instala los requerimientos del fichero requirements.tx

```
pip install -r requirements.txt
```

Además, crea un kernel específico

```
python -m ipykernel install --user --name tf_vivienda --display-name tf_vivienda
```

## Requirements

```
pandas==2.1.1
numpy==1.23.5
matplotlib==3.9.0
seaborn==0.13.2
tensorflow==2.12.0
scikit-learn==1.2.2
ipykernel==6.29.4
jupyterlab==4.2.1
plotly==5.22.0
opencv-python-4.10.0.84
```
