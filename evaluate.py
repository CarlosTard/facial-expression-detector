import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import model
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score

#Módulo que sea capaz de proporcionar, dado un conjunto de imágenes, el porcentaje de acierto, tanto de forma desagregada por clases como de forma agregada.

def evaluar(mod, datagen):

    predictions = mod.predict(datagen)
    y_true = datagen.classes
    y_pred=tf.argmax(predictions,1)
    array = tf.cast(tf.math.confusion_matrix(y_true, y_pred),tf.int32).numpy()
    df_cm = pd.DataFrame(array, index = model.classes,
                  columns = model.classes)
    plt.figure(figsize = (10,7))
    ax = plt.axes()
    sn.heatmap(df_cm, annot=True, fmt='d', ax =ax)
    print("Métricas agregadas:")
    print(mod.evaluate(datagen, verbose=0, return_dict=True))
    print()
    print("Métricas por clases:")
    for i in range(0, model.num_classes):
        print(model.classes[i] + ":" + " "*(16-len(model.classes[i])), end='')
        p = precision_score(y_true, y_pred, labels=[i], average='macro')
        print(f"Precision: {p} ", end='')
        r = recall_score(y_true, y_pred, labels=[i], average='micro')
        print(f"Recall: {r} ")
    ax.set_title("Confusion matrix " + mod.name)
    plt.show()
        
