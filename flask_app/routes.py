from typing import Any

from flask import Flask, request, jsonify
import torch
from PIL import Image

from vit import ViT
from vit.utils import _process_image

app = Flask(__name__)
model = ViT.from_pretrained(
    'notaryanramani/ViT-cifar10', # loads the latest model from main branch
    revision = 'main', # use branch name here to change the model version
) 


# dummy endpoint
@app.route('/api/dummy', methods=['POST'])
def predict():
    """
        Dummy endpoint to test the API.
    """
    prediction = torch.randint(0, 10, (1,)).item()
    output = {
        'prediction': prediction
    }
    return jsonify(output)

# endpoint to predict class
@app.route('/api/predict', methods=['POST'])
def predict():
    """ 
        Endpoint to predict the class of an image.
    """
    image = request.files['image']
    image = _process_image(image)
    prediction = model.predict(image)
    output = {
        'prediction': prediction.tolist()
    }
    return jsonify(output)


# endpoint to predict probabilities
@app.route('/api/predict_proba', methods=['POST'])
def probabilities():
    """
        Endpoint to predict the probabilities of each class.
    """
    image = request.files['image']
    image = _process_image(image)
    probabilities = model.predict_proba(image)
    output = {
        'probabilities': probabilities.tolist() 
    }
    return jsonify(output)