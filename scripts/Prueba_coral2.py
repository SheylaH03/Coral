import os
import pathlib
import time
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plot

script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'model_edgetpu.tflite')
label_file = os.path.join(script_dir, 'labels.txt')

ruta_perros_prueba = "../conjunto_datos/prueba/dogs/"
ruta_gatos_prueba = "../conjunto_datos/prueba/cats/"

nombre_imagenes_perros = glob.glob(ruta_perros_prueba+"*.jpg")
nombre_imagenes_gatos = glob.glob(ruta_gatos_prueba+"*.jpg")

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

size = common.input_size(interpreter)

T = []

contador_imagenes_p = 0
contador_aciertos_p = 0
for i in nombre_imagenes_perros:

    image = Image.open(i).convert('RGB').resize(size, Image.ANTIALIAS)

    common.set_input(interpreter, image)
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    T += [inference_time]

    classes = classify.get_classes(interpreter, top_k=1)
    print('%.1fms' % (inference_time * 1000))

    labels = dataset.read_label_file(label_file)
    for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score))

    # if prediccion[0][0] > prediccion[0][1]:
    #     animal = "gato"
    # else:
    #     animal = "perro"
    #     contador_aciertos_p = contador_aciertos_p + 1
    # # print(i + "->"+animal)
    # # print(prediccion)
    # contador_imagenes_p = contador_imagenes_p + 1
# promedio = contador_aciertos / contador_imagenes
# print("Promedio de aciertos = "+ str(promedio))


# print("------------------Probando imagenes de gatos--------------------")
contador_imagenes_g = 0
contador_aciertos_g = 0
for i in nombre_imagenes_perros:

    image = Image.open(i).convert('RGB').resize(size, Image.ANTIALIAS)

    common.set_input(interpreter, image)
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    T += [inference_time]

    classes = classify.get_classes(interpreter, top_k=1)
    print('%.1fms' % (inference_time * 1000))

    labels = dataset.read_label_file(label_file)
    for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
    # if prediccion[0][0] > prediccion[0][1]:
    #     animal = "gato"
    # else:
    #     animal = "perro"
    #     contador_aciertos_g = contador_aciertos_g + 1
    # # print(i + "->"+animal)
    # # print(prediccion)
    # contador_imagenes_g = contador_imagenes_g + 1
# promedio = contador_aciertos / contador_imagenes
# print("Promedio de aciertos = "+ str(promedio))

# promedio = (contador_aciertos_g + contador_aciertos_p)/(contador_imagenes_g + contador_imagenes_p)
# print('Promedio de aciertos:', promedio)
print('Suma de tiempos de inferencia:', sum(T[1:]))
print('Media de tiempos de inferencia:', np.mean(T[1:]))
print('Desviacion estandar de tiempos de inferencia:', np.std(T[1:]))



plot.hist(x=T[1:])
plot.title('Histograma')
plot.xlabel('Segundos')
plot.ylabel('Frecuencia')

plot.show() #dibujamos el histograma