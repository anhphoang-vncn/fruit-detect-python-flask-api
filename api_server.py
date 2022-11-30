from PIL import Image
from io import BytesIO
import base64
import numpy as np
import cv2
from keras.applications.mobilenet import  MobileNet
from keras.layers import Dense, Dropout 
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

n_class = 10

def pil_image_to_base64(pil_image): #input str - output img
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue())

def base64_to_pil_image(base64_img): #input img - output bytes
    return Image.open(BytesIO(base64.b64decode(base64_img)))

def get_model():
    # Tạo base model
    base_model = MobileNet(include_top=False, weights="imagenet", input_shape=(224,224,3))
    # Tạo model chính
    x = base_model.output
    # Add some new Fully connected layers to
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(512, activation='relu')(x)
    outs = Dense(n_class, activation='softmax')(x)
    # Đóng băng các layer của base_model
    for layer in base_model.layers:
        layer.trainable = False
    model = Model(inputs=base_model.inputs, outputs= outs)
    return model

model = get_model()
model.load_weights('./models/best.hdf5') 

# Khoi tao Flask Server BE
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# @app.route("/api/classifier/<string:fruit_b64>", methods=['POST', 'GET'])
@app.route("/api/classifier", methods=['POST'])
@cross_origin(origins='*')
def detect():
    fruit_b64 = request.json['fruit_b64']
    fruit_pil = base64_to_pil_image(fruit_b64)
    fruit_cv2 = np.array(fruit_pil)
    # resize
    image = cv2.resize(fruit_cv2, (224, 224))
    #Normalize
    image = image/127.5
    # expand_dims
    image = tf.expand_dims(image, axis=0)
    # classifi
    predicted_class = model.predict(image)
    output = str(np.argmax(predicted_class))
    return jsonify({
            "api_name":"AnhFang fruit detect API",
            "fruit_detect": output,
            })

@app.route("/api/test", methods=['POST', 'GET'])
@cross_origin(origins='*')
def testFuntion():
    return jsonify([{
            "api_name":"AnhFang fruit detect API",
            "fruit_detect": "test ok",
            }])

# Start BE
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888')
