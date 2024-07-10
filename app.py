from flask import Flask, request, render_template, url_for, jsonify
from PIL import Image
import numpy as np
import cv2
from classify import classify_image

app = Flask(__name__)

font_names = [
    "IBM Plex Sans Arabic",
    "Lemonada",
    "Marhey",
    "Scheherazade New"
]


@app.route('/')
def index():

    return render_template('index.html', appName="YMY Classification")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        image_file = request.files['fileup']
        
        # Open the image using PIL
        img = Image.open(image_file)
        
        # Convert the image to a numpy array
        img_array = np.array(img)
        
        # Pass the image array to the classify_image function
        result = classify_image(img_array)

        return render_template('index.html', prediction=str(result) + "-" + font_names[int(result)], appName="YMY Classification")
    else:
        return render_template('index.html',appName="YMY Classification")

@app.route('/predictApi', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image_file = request.files['image']

    # Open the image using PIL
    img = Image.open(image_file)

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Pass the image array to the classify_image function
    result = classify_image(img_array)

    return str(result)

if __name__ == '__main__':
    app.run(debug=True)
