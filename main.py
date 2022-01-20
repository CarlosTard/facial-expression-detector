
# modulo de proposito general

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import model
import evaluate
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from cortacaras import carga_caras

""" data tree
    +data
    +----train
         +---class1
         +---class2
            ...
    +----test 
         +---class1
         +---class2
            ...
"""

def load_images1(path, fit_data=False):
    #Aplicamos reescalado, y un flip horizontal aleatorio, para aprovechar las simetrías de las caras
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(  
        rescale=1./255, #Hay que ver si el rescale mejora algo
        rotation_range = 0,
        horizontal_flip=fit_data, # Hay que ver si horizontal flip mejora el overfitting
        validation_split=0.2 if fit_data else None
        ) 
    if fit_data:
        train_generator = datagen.flow_from_directory( 
            path,
            shuffle=fit_data, 
            target_size=model.target_size,
            color_mode=model.color_mode,
            class_mode=model.class_mode,
            classes=model.classes,
            subset='training')
        val_generator = datagen.flow_from_directory( 
            path,
            shuffle=fit_data, 
            target_size=model.target_size,
            color_mode=model.color_mode,
            class_mode=model.class_mode,
            classes=model.classes,
            subset='validation')
        return train_generator, val_generator
        
    else:
        train_generator = datagen.flow_from_directory( 
            path,
            shuffle=fit_data, 
            target_size=model.target_size,
            color_mode=model.color_mode,
            class_mode=model.class_mode,
            classes=model.classes)
    
        return train_generator
    
def show_img(images, lab):
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(lab)):
        lon = math.ceil(math.sqrt(len(lab)))
        ax = plt.subplot(lon,lon, i + 1)
        plt.imshow(np.array(tf.math.scalar_mul(255, images[i])).astype("uint8"))
        plt.title(f"Cara {i}: " + model.classes[lab[i]], y=-0.1 - 0.15*(len(lab) > 1))
        plt.axis("off")
    fig.subplots_adjust(hspace=.5)
    plt.show()


parser = argparse.ArgumentParser("python3 main.py", description="")
model_name = parser.add_argument("model", help="Model name")
load = parser.add_argument("-l", action='store_true', help="Load Model from file ")
load = parser.add_argument("-g", action='store_true', help="Train with gpu ")
eva = parser.add_argument("-e", const='data/test',nargs='?', help="Evaluate model. Optionally provide test data path")

fit = parser.add_argument("-f", const='data/train',nargs='?', help="Fit model. Optionally provide train data path. Always saves fitted model")
pipe = parser.add_argument("-p", const='fotos_prueba',nargs='?', help="Test final pipeline")

def pipeline(path):
    path = path + "/"
    with tf.device('/CPU:0'):
        caras = carga_caras(path)
    model_input = tf.stack(caras)
    if len(model_input) == 0:
        print("No hay caras en el directorio " + path)
        exit()
    print(f"Modelo {args.model} emitiendo predicciones")
     
    prob = mod.predict(model_input)
    for cont, i in enumerate(prob):
        print(f"-----CARA {cont}-----")
        for cont2, j in enumerate(i):
            clase = model.classes[cont2]
            print(clase + " "*(11-len(clase)) + "%.4f %%" % (j*100))
    pred = tf.argmax(prob,axis=-1)
    show_img(caras, pred)


if __name__ == "__main__":
    
    args = parser.parse_args()
    # Para entrenar con gpu
    if args.g:
        physical_devices = tf.config.list_physical_devices('GPU') 
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if args.l:
        mod = model.load_model(args.model)
    elif args.f is not None:
        train_generator, val_generator = load_images1(args.f, True)
        mod = model.compile_fit(train_generator, val_generator, args.model)
    if args.p is not None:
        pipeline(args.p)
    elif args.e is not None:
        train_generator = load_images1(args.e)
        evaluate.evaluar(mod, train_generator)
