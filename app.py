from flask import Flask
#from redis import Redis
from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
import joblib
import numpy as np
import os
#dependences from load image 
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
#import cv2
import zipfile
import json
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import urllib.request
import zipfile


APP = Flask(__name__)
API = Api(APP)
app = Flask("_name_")
CORS(app)

@app.route('/ret', methods = ['GET'])
def returnascii():
        inputchr = str(request.args['nome'])
        type_prediction=str(request.args['tipopredicao'])
        stringfinal=inputchr.replace(" ","+")
        stringfinal=stringfinal.replace("data:image/png;base64,","")
        strinfinal2=str.encode(stringfinal)
        data={}
        data['img'] = strinfinal2
        im = Image.open(BytesIO(base64.b64decode(data['img'])))
        image_result = open('deer_decode2.png', 'wb') # create a writable image and write the decoding result
        image_result.write(base64.b64decode(data['img']))
        model = VGG19(weights='imagenet', include_top=False)
        img = image.load_img('deer_decode2.png', target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        vgg19_feature_listteste=[]
        vgg19_feature = model.predict(img_data)
        vgg19_feature_np = np.array(vgg19_feature)
        vgg19_feature_listteste.append(vgg19_feature_np.flatten())

        #validation of image
        model_validation_zip = zipfile.ZipFile('models/SVM_validador.zip') 
        #load binary model
        model_binary_zip = zipfile.ZipFile('models/SVM_binario.zip') 
        #load multilcass model
        model_multiclass_zip = zipfile.ZipFile('models/SVM_multiclasse.zip') 

        clf_validation =joblib.load(model_validation_zip.open('SVM_validador.mdl'))
        clf_binary =joblib.load(model_binary_zip.open('SVM_binario.mdl'))
        clf_multiclass =joblib.load(model_multiclass_zip.open('SVM_multiclasse.mdl'))

        result = ""
        if(clf_validation.predict(vgg19_feature_listteste)=="olho"):
                if(type_prediction=='binario'):
                        pred= clf_binary.predict(vgg19_feature_listteste)
                        if( pred[0]=="absence"):
                                result="Ausência"
                        else:
                                result="Presença"
                               
                if(type_prediction=='multiclasse'):
                        pred= clf_multiclass.predict(vgg19_feature_listteste)
                        if( pred[0]=="absence"):
                                result="Ausência"
                              
                        if( pred[0]=="leve"):
                                result="Leve"
                                
                        if( pred[0]=="grave"):
                                result="Grave"
                               
                        else:
                                result="Moderado"
                                
        else:
                result="invalido"
        people = [{'prediction': result}]  
        return jsonify(people[0]) 


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)