from flask import Flask, request, jsonify
from vit import VIT
import torch

app = Flask(__name__)
model = VIT.from_pretrained('notaryanramani/ViT-cifar10')

# dummy endpoint
@app.route('/api/dummy', methods=['POST'])
def predict():
    prediction = torch.randint(0, 10, (1,)).item()
    return jsonify(prediction)
