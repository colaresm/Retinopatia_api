from flask import Flask
from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
import base64
from core import predict_image


APP = Flask(__name__)
API = Api(APP)
app = Flask("_name_")
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_in_base64 = data.get('nome')   
        type_prediction = data.get('tipopredicao')
        
        image_in_base64=image_in_base64.replace(" ","+")
        image_in_base64=image_in_base64.replace("data:image/png;base64,","")
        image_in_base64=str.encode(image_in_base64)

        image_result = open('reiceved_image.png', 'wb') 
        image_result.write(base64.b64decode(image_in_base64))

        result=""
        
        validation=predict_image("validation")

        if validation=="eye":
              result=predict_image(type_prediction)
        else:
              result=="randon"  

        data = {'type_prediction':type_prediction,'prediction': result}
       
        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/hello', methods=['GET'])
def hello():
    return "Hello!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)