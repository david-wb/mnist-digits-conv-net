import os
from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
import imageio 
import numpy as np
from flask import jsonify

model = tf.keras.models.load_model('mnist_cnn.h5')
model._make_predict_function()

def predict(image_path):
    image = imageio.imread(image_path).astype('float32') / 255
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, -1)
    pred = model.predict(image)[0].argmax()
    return pred

UPLOAD_FOLDER = './image_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = set(['png'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            pred = predict(filepath)
            res = {
                'file': filename,
                'prediction': int(pred)
            }
            return jsonify(res)
    return '''
    <!doctype html>
    <title>Inference Server</title>
    <h1>Upload a MNIST image in PNG format (Use ./test.png for an example)</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''