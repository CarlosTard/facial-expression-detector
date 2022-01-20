# Facial expression detector
This is a multiclass clasification model that detects the facial expressions of a given picture. 
To detect and crop the faces, it uses MTCNN. Then, with a tensorflow CNN, this algorithm classifies individual faces into one of the following classes:
 'angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised'

The MTCNN model is pretrained, and we use it without fine-tunning to this task, in order to maximize generalization.

The CNN tensorflow model is trained using EarlyStopping, in order to stop training when the val_categorical_accuracy has stopped improving.

