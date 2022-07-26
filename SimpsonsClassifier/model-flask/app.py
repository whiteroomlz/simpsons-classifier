import base64
import io
import json
import torch
import torchvision.transforms as transforms

from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)
simpsons_class_index = json.load(open('simpsons.json'))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load('model_scripted.pt')
model.to(device)
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes).to(device)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    return predicted_idx, simpsons_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img_bytes = base64.decodebytes(request.data)
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()
