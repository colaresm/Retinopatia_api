from flask import Flask
from redis import Redis
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
import json
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64



APP = Flask(__name__)
API = Api(APP)
app = Flask("_name_")
CORS(app)

@app.route('/ret', methods = ['GET'])
def returnascii():
        inputchr = str(request.args['nome'])
        stringfinal=inputchr.replace(" ","+")
        stringfinal=stringfinal.replace("data:image/png;base64,","")
        strinfinal2=str.encode(stringfinal)
        data={}
        data['img'] = strinfinal2
        im = Image.open(BytesIO(base64.b64decode(data['img'])))
        image_result = open('deer_decode.png', 'wb') # create a writable image and write the decoding result
        image_result.write(base64.b64decode(data['img']))
      #  print(type(inputchr))
        model = VGG19(weights='imagenet', include_top=False)
        img = image.load_img('deer_decode.png', target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        vgg19_feature_listteste=[]
        vgg19_feature = model.predict(img_data)
        vgg19_feature_np = np.array(vgg19_feature)
        vgg19_feature_listteste.append(vgg19_feature_np.flatten())
        crc =joblib.load('SVM_multiclasse.mdl')
        pred= crc.predict(vgg19_feature_listteste)
      
        print(pred)
        result = ""
        
        if( pred[0]=="leve"):
                result="Leve"
        if( pred[0]=="moderado"):
                result="Moderado"
        if( pred[0]=="grave"):
                result="Grave"
        else:
                result="Ausência"
        people = [{'prediction': result}]
        
        return jsonify(people[0]) 


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
