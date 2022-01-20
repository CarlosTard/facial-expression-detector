
# face detection with mtcnn on a photograph
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import glob
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import load_img
import tensorflow as tf
import numpy as np

def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

def carga_caras (path):

	# create the detector, using default weights
	detector = MTCNN()
	caras = []
	for fmt in ["*jpg","*png"]:
		path_to_file_list = glob.glob(path + fmt )
		file_list = [i.split('/')[-1] for i in path_to_file_list]
		for archivo in file_list:
			print(f"Detecting faces in file: {archivo}")
			pixels = rgba2rgb(pyplot.imread(path+archivo))
			filas, columnas, rgb =  pixels.shape
			faces = detector.detect_faces(pixels)
			print(f"{len(faces)} faces detected")
			for face in faces:
				x1, y1, width, height = face['box']
				x2, y2 = x1 + width, y1 + height
				x1 = max (0,x1)
				y1 = max (0,y1)
				x2 = min (x2, columnas)
				y2 = min (y2, filas)
				width = x2-x1
				height = y2-y1
				img = pixels[y1:y2, x1:x2]
				tensor = tf.convert_to_tensor(img.reshape(1, height, width, 3))
				cara_tf = tf.reshape(tf.image.resize(tensor, [48,48]), [48,48,3])
				cara_tf = tf.math.scalar_mul(1/255, cara_tf)
				caras.append(cara_tf)
	return caras

		
