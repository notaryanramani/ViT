from flask import Flask, request, jsonify
from vit import ViT
import torch
from typing import Any
from PIL import Image

app = Flask(__name__)
model = ViT.from_pretrained('notaryanramani/ViT-cifar10')

def _process_image(image: Any) -> torch.Tensor:
    """
        Process the image to the correct shape and size.

        Args:
            image: Any
                Image to be processed.
            
        Returns:
            image: torch.Tensor
                Processed image.
    """
    image = Image.open(image).convert('RGB')
    image = image.resize(model.img_size)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    return image

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