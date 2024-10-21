
import numpy as np
import cv2
from tensorflow.keras.models import load_model


def predict_image(type_prediction): 
  
  model = get_model(type_prediction)

  img = pre_process_image()  

  prediction = model.predict(img)

  predicted_labels = np.argmax(prediction, axis=1)

  unique_labels=[]
  if type_prediction=="validation":
        unique_labels=['eye', 'random']
  else:
        unique_labels=['absence', 'grave', 'leve', 'moderado']

  predicted_class_names = [unique_labels[label] for label in predicted_labels]

  return adjust_response_for_multiclass(type_prediction,predicted_class_names[0]) 

def pre_process_image():
  img_width, img_height = 128, 128

  img = cv2.imread("reiceved_image.png")
  img = cv2.resize(img, (img_width, img_height))
  img = img / 255.0   
  img = np.expand_dims(img, axis=0)  

  return img

def get_model(type_prediction):

    if type_prediction=="multiclass" or type_prediction=="binary":
        model = load_model('multiclass_model.h5')
        return model
    if type_prediction=="validation":
        model=load_model("eye_validation_model.h5")
        return model


def get_labels_for_response(type_prediction):
    unique_labels=[]
    if type_prediction=="validation":
        unique_labels=['eye', 'randon']
    else:
        unique_labels=['absence', 'grave', 'leve', 'moderado']
    return unique_labels

def adjust_response_for_multiclass(type_prediction,response):
    if type_prediction=="validation":
        return response
    if (type_prediction=="multiclass" or type_prediction=="binary" ) and response=="absence":
        return "saudável"
    if type_prediction=="binary":
        print("aqio")
        return "presença de retinopatia diabética" 
        
