## Modelo keras secuencial. El módulo tambien se encarga de guardar y cargar el modelo una vez entrenado
import tensorflow as tf
from tensorflow.keras import callbacks

saved_models = 'saved_models/'
target_size = (48, 48)
class_mode = 'categorical' # Como representar labels:' "categorical", "binary", "sparse", "input",
max_epochs = 500 
patience = 8
color_mode='rgb' #Puede ser "grayscale", "rgb", "rgba"
classes=['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
num_classes=len(classes)
use_multiprocessing=False
workers=1

#### IMPORTANTE ###
""" 
Lo que sabemos hasta ahora de las capas. (Solo usamos los parámetros principales de cada capa)
-Convolucional: extrae patrones locales
                Conv2D(filters, kernel_size, strides=(1, 1), activation=None)
                
-MaxPooling2D: actúa como función de activación de las capas convolucionales(introduce no-linealidad), extrayendo patrones locales.
             además reduce el tamaño de muestra
             MaxPooling2D(pool_size=(2, 2)) pool_size indica el tamaño de la matriz 2x2 sobre la que se calcula el maximo
             
-AveragePooling2D: tiene la misma función que maxpooling, pero calcula la media en cada matriz de dimensiones pool_size

-Flatten: transforma el input en un array unidimensional (lo aplasta). Se usa para poder pasar el output de las capas convolucionales-maxpooling
          como input a la siguiente parte del modelo:las capas densamente conectadas. Si se hace al principio es devastador, pues estropea
          la estructura bidimensional de las imágenes, e imposibilita encontrar patrones.
          
-Dense: capas de neuronas densamente conectadas. Toman como input la salida de las capas convolucionales(aplastadas en un vector), y se usan
        para combinar los patrones locales hallados por las capas convolucionales
"""
    
def model1(): ## Primer modelo de prueba. Los resultados son nefastos debido al flatten

    """
    Model: "model1"

    Trainable params: 69,317

    Entrenando modelo model1
     
    Epoch 5/5
    898/898 [==============================] - 6s 7ms/step - loss: 1.9382 - accuracy: 0.1675
    """
    ### loss: 1.8727 - accuracy: 0.2471
    model = tf.keras.models.Sequential(name='model1', layers=[
        tf.keras.layers.Flatten(input_shape=(48, 48,3)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='relu')])
    return model
    
def model1b():
    model = tf.keras.models.Sequential(name='model1b', layers=[
        tf.keras.layers.Flatten(input_shape=(48, 48,3)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='relu'),
        tf.keras.layers.Activation("softmax")])
    return model
    
def model2(): ## Primer modelo con capas convolucionales, se nota mejoría
    """
    Model: "model2"
    Trainable params: 50,679

    Entrenando modelo model2
    Epoch 5/5
    898/898 [==============================] - 13s 14ms/step - loss: 1.2516 - accuracy: 0.5312
    """
    model = tf.keras.models.Sequential(name='model2', layers=[
        tf.keras.layers.Conv2D(48, 3, input_shape=(48, 48,3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes)])
    return model
    
def model2b(): # El mismo modelo que antes, pero con funciones de activacion en las capas convolucionales. Parece que influyen negativamente
                # Se puede deber a que las capas maxpooling funcionan como funcion de activacion, y a la vez extraen datos locales, mientras que
                # si añadimos funcion de activación, estas funcionalidades se desfiguran?
    """
    Model: "model2b"
    Trainable params: 50,679
    Epoch 5/5
    898/898 [==============================] - 12s 14ms/step - loss: 1.3406 - accuracy: 0.4529
    """
    ###  loss: 1.3805 - accuracy: 0.4440
    model = tf.keras.models.Sequential(name='model2b', layers=[
        tf.keras.layers.Conv2D(48, 3, input_shape=(48, 48,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes)])
    return model
    
def model2c(): ## El model2 inicial pero la ultima activacion es sofmax, que nos proporciona las probabilidades de pertenecer a cada clase y 
                # posibilita el uso de metricas como recall y precision
    """
    Model: "model2"
    Trainable params: 50,679
    Epoch 5/5
    898/898 [==============================] - 22s 24ms/step - loss: 1.2040 - accuracy: 0.5460 - recall: 0.3353 - precision: 0.7586
    """
    #Para poder usar metricas por clases, necesitamos que el output esté en [0,1]^7
    ### loss: 1.3260 - accuracy: 0.5020 - recall: 0.2987 - precision: 0.7016
    model = tf.keras.models.Sequential(name='model2c', layers=[
        tf.keras.layers.Conv2D(48, 3, input_shape=(48, 48,3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes),
        tf.keras.layers.Activation("softmax")])
    return model
    
def model3(): ## Proof of concept: añadimos una capa convolucional más, y parece mejorar ligeramente los resultados. 
    """
    Model: "model3"
    Trainable params: 65,703
    Epoch 5/5
    898/898 [==============================] - 12s 14ms/step - loss: 1.1774 - accuracy: 0.5636
    """
    model = tf.keras.models.Sequential(name='model3', layers=[
        tf.keras.layers.Conv2D(48, 3, input_shape=(48, 48,3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes)])
    return model
    
    
def model2d(): ## Cambio radical. Probamos con más capas convolucionales, algunas de ellas seguidas(sin capa maxpooling entre medias), usando relu 
                # como activacion. Ademas añadimos dos capas densas finales. Debido a las capas extra, necesita mas pasos de entrenamiento(10)
    """
        Model: "model2d"
    Trainable params: 114,637

    Epoch 10/10
    898/898 [==============================] - 13s 15ms/step - loss: 0.8547 - accuracy: 0.6784 - recall: 0.5595 - precision: 0.7959
    """
    ### Evaluacion con datos de testing. Parece haber problemas con la clase fearful y sad
    """
    Métricas agregadas:
    {'loss': 1.275061011314392, 'categorical_accuracy': 0.5275843143463135}
    
    Métricas por clases:
    angry:           Precision: 0.4372384937238494 Recall: 0.4363256784968685 
    disgusted:       Precision: 0.5211267605633803 Recall: 0.3333333333333333 
    fearful:         Precision: 0.324032403240324 Recall: 0.3515625 
    happy:           Precision: 0.7179104477611941 Recall: 0.8134160090191658 
    neutral:         Precision: 0.5174746335963923 Recall: 0.3722627737226277 
    sad:             Precision: 0.38936781609195403 Recall: 0.43464314354450684 
    surprised:       Precision: 0.7030625832223701 Recall: 0.6353790613718412 
    """
    model = tf.keras.models.Sequential(name='model2d', layers=[
        tf.keras.layers.Conv2D(48, 3, input_shape=(48, 48,3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3, activation='relu'),
        tf.keras.layers.Conv2D(48, 3, activation='relu'),
        tf.keras.layers.Conv2D(48, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation="softmax")])
    return model
    
def model2e(): # Añadimos aun mas capas. Parece que hemos overfiteado el modelo(20 epochs)
    """
    Model: "model2e"

    Trainable params: 141,805

    Epoch 20/20
    898/898 [==============================] - 24s 27ms/step - loss: 0.5862 - accuracy: 0.7842 - recall: 0.7220 - precision: 0.8531
    """
    ########### Evaluacion con datos de testing. Comentarios: parece haber problemas con la clase fearful y sad
    """
    Métricas agregadas:
    {'loss': 1.6575945615768433, 'accuracy': 0.514906644821167, 'recall': 0.4595987796783447, 'precision': 0.5765466690063477}
    
    Métricas por clases:
    angry:           Precision: 0.42433537832310836 Recall: 0.4331941544885177 
    disgusted:       Precision: 0.5555555555555556 Recall: 0.2702702702702703 
    fearful:         Precision: 0.3304812834224599 Recall: 0.3017578125 
    happy:           Precision: 0.7165745856353591 Recall: 0.7311161217587373 
    neutral:         Precision: 0.4656786271450858 Recall: 0.48418491484184917 
    sad:             Precision: 0.3761398176291793 Recall: 0.39695268644747395 
    surprised:       Precision: 0.688667496886675 Recall: 0.6654632972322503
    """
    model = tf.keras.models.Sequential(name='model2e', layers=[
        tf.keras.layers.Conv2D(48, 3, input_shape=(48, 48,3) , activation='relu'),
        tf.keras.layers.Conv2D(48, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3, activation='relu'),
        tf.keras.layers.Conv2D(48, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3, activation='relu'),
        tf.keras.layers.Conv2D(48, 3, activation='relu'),
        tf.keras.layers.Conv2D(48, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation="softmax")])
    return model
    
def model2f(): # Probando con filtros crecientes
    """
    Model: "model2f"

    Trainable params: 404,773
    Epoch 17/100
    718/718 [==============================] - 17s 24ms/step - loss: 0.5761 - categorical_accuracy: 0.7882 - val_loss: 1.4005 - val_categorical_accuracy: 0.5532
    """
    ## Parece mejorar. Se nota mejoría en clase sad
    """
    Métricas agregadas:
    {'loss': 1.2344497442245483, 'categorical_accuracy': 0.5661744475364685}
    
    Métricas por clases:
    angry:           Precision: 0.5147783251231527 Recall: 0.4363256784968685 
    disgusted:       Precision: 0.5416666666666666 Recall: 0.23423423423423423 
    fearful:         Precision: 0.40512048192771083 Recall: 0.2626953125 
    happy:           Precision: 0.704052780395853 Recall: 0.8421645997745209 
    neutral:         Precision: 0.5380952380952381 Recall: 0.4582319545823195 
    sad:             Precision: 0.42083082480433476 Recall: 0.5605453087409783 
    surprised:       Precision: 0.7222898903775883 Recall: 0.7135980746089049
    """
    model = tf.keras.models.Sequential(name='model2f', layers=[
        tf.keras.layers.Conv2D(12, 3, input_shape=(48, 48,3) , activation='relu'),
        tf.keras.layers.Conv2D(24, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(2*48, 3, activation='relu'),
        tf.keras.layers.Conv2D(4*48, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation="softmax")])
    return model

def model2g(): #mejora un poco el accuracy, a costa de aumentar el numero de parametros
    """
    Model: "model2g"

    Trainable params: 3,165,535
    Epoch 20/500
    718/718 [==============================] - 34s 47ms/step - loss: 0.2791 - categorical_accuracy: 0.9006 - val_loss: 2.0661 - val_categorical_accuracy: 0.5626
    
    Métricas agregadas:
    {'loss': 1.3319982290267944, 'categorical_accuracy': 0.5812203884124756}
    
    Métricas por clases:
    angry:           Precision: 0.5365853658536586 Recall: 0.4363256784968685 
    disgusted:       Precision: 0.7291666666666666 Recall: 0.3153153153153153 
    fearful:         Precision: 0.4309462915601023 Recall: 0.3291015625 
    happy:           Precision: 0.7210753720595295 Recall: 0.846674182638106 
    neutral:         Precision: 0.5028694404591105 Recall: 0.5685320356853204 
    sad:             Precision: 0.4550185873605948 Recall: 0.4907778668805132 
    surprised:       Precision: 0.7590361445783133 Recall: 0.6823104693140795 
    """
    model = tf.keras.models.Sequential(name='model2g', layers=[
        tf.keras.layers.Conv2D(12, 3, input_shape=(48, 48,3) , activation='relu'),
        tf.keras.layers.Conv2D(24, 1, activation='relu'),
        tf.keras.layers.Conv2D(48, 2),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(2*48, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(4*48, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(8*48, 1, activation='relu'),
        tf.keras.layers.Conv2D(16*48, 3),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(288, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation="softmax")])
    return model
    
def model4(): # Probamos disminuyendo la linealidad(con funciones de activacion y inicializador personalizado)
    """
    Model: "model4"
    Trainable params: 2,239,147
    
    Métricas agregadas:
    {'loss': 1.1096537113189697, 'categorical_accuracy': 0.5895792841911316}
    
    Métricas por clases:
    angry:           Precision: 0.5324137931034483 Recall: 0.40292275574112735 
    disgusted:       Precision: 0.6071428571428571 Recall: 0.3063063063063063 
    fearful:         Precision: 0.5 Recall: 0.2333984375 
    happy:           Precision: 0.7560851926977687 Recall: 0.8404735062006764 
    neutral:         Precision: 0.4787985865724382 Recall: 0.6593673965936739 
    sad:             Precision: 0.4585400425230333 Recall: 0.5188452285485164 
    surprised:       Precision: 0.7422434367541766 Recall: 0.7484957882069796
    """
    """
    Epoch 16/500
    718/718 [==============================] - 31s 43ms/step - loss: 0.4586 - categorical_accuracy: 0.8333 - val_loss: 1.4496 - val_categorical_accuracy: 0.5800
    """

    model = tf.keras.models.Sequential(name='model4', layers=[
        tf.keras.layers.Conv2D(12, 2, input_shape=(48, 48,3) , activation='relu'),
        tf.keras.layers.Conv2D(24, 1, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Conv2D(48, 2, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(2*48, 2, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(4*48, 2, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(8*48, 1, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Conv2D(16*48, 2, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(288, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation="softmax")])
    return model 

### Como hemos comprobado en la fase de entrenamiento, los anteriores modelos pueden ser entrenados para 
### tener accuracy ideal(sobre cjto train), simplemente aumentando epochs. Sin embargo,sufren de overfitting

def model5():## Incorpora ideas de tamaños de filtro incrementales, entrenamiento con early stopping, funciones de activacion en todas las capas convolucionales, 
            ## y añade dropouts para reducir el overfitting
    """
    Model: "model5"
    Trainable params: 2,239,147
    Epoch 34/500
    718/718 [==============================] - 33s 47ms/step - loss: 0.7972 - categorical_accuracy: 0.7001 - val_loss: 1.0988 - val_categorical_accuracy: 0.6079

    Métricas agregadas:
    {'loss': 1.0753401517868042, 'categorical_accuracy': 0.6074115633964539}
    
    Métricas por clases:
    angry:           Precision: 0.4866920152091255 Recall: 0.534446764091858 
    disgusted:       Precision: 0.5517241379310345 Recall: 0.43243243243243246 
    fearful:         Precision: 0.5192660550458715 Recall: 0.2763671875 
    happy:           Precision: 0.776188042922841 Recall: 0.8562570462232244 
    neutral:         Precision: 0.559533721898418 Recall: 0.5450121654501217 
    sad:             Precision: 0.48849693251533743 Recall: 0.5108259823576584 
    surprised:       Precision: 0.6676356589147286 Recall: 0.8291215403128761 
    """
    model = tf.keras.models.Sequential(name='model5', layers=[
        tf.keras.layers.Conv2D(12, 2, input_shape=(48, 48,3) , activation='relu'),
        tf.keras.layers.Conv2D(24, 1, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Conv2D(48, 2, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(2*48, 2, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(4*48, 2, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(8*48, 1, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Conv2D(16*48, 2, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(288, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation="softmax")])
    return model 


def compile_fit(train_generator, val_generator, model_name):
    try:
        #elige el modelo de nombre model_name, que coincide con el nombre de la función
        model = globals()[model_name]() 
    except KeyError:
        print(f"Modelo {model_name} no existe")
        return
    model.compile(
        optimizer='adam',
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]#,tf.keras.metrics.Recall(),tf.keras.metrics.Precision()]
        )
    print(model.summary())
    #fit realiza shuffle por defecto.
    print(f"Entrenando modelo {model_name}")
    earlystopping = callbacks.EarlyStopping(monitor ="val_categorical_accuracy",  
                                        mode ="max", patience = patience,  
                                        restore_best_weights = True)
    model.fit(train_generator, validation_data=val_generator,
             epochs=max_epochs, use_multiprocessing=use_multiprocessing,
              workers=workers, callbacks=[earlystopping])
    model.save(saved_models + model_name)
    print(f"Modelo {model_name} guardado en {saved_models + model_name}")
    return model

def load_model(model_name):
    return tf.keras.models.load_model(saved_models + model_name)
    
def predict(mod, model_input ):
    prob = mod.predict(model_input)
    pred = tf.argmax(prob,axis=-1)
    return pred.numpy()
