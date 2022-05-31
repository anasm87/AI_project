from flask import Flask, request, jsonify
app = Flask(__name__)


# importing sys
import sys
  
# adding Folder_2 to the system path
sys.path.insert(0, './../src')
from inference import *

ALLOWED_EXTENSIONS = {'png','jpeg','jpg'}
def allowed_file(filename):
    return '.' in filename and filename.split('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename=="":
            return jsonify({'error':'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error':'formate is not suported'})
        
        
        img_byte = file.read() 
        prediction,probability = predict_image(img_byte)
        data = {'prediction':prediction,'probability':str(probability)}
        return jsonify(data)
    


        
 

    return jsonify({'results':1})


if(__name__ == "__main__"):
    app.run()