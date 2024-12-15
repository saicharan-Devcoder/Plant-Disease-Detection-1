import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import torchvision.transforms as transforms
import pandas as pd
import random


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def prediction(image_path):
    # Open the image file
    image = Image.open(image_path)
    
    # Convert the image to RGB (this ensures no alpha channel is present)
    image = image.convert("RGB")
    
    # Resize the image to (224, 224)
    image = image.resize((224, 224))
    
    # Define transformation: Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
    ])
    
    # Apply the transformation to the image
    input_data = transform(image)
    
    # Add a batch dimension: From [3, 224, 224] to [1, 3, 224, 224]
    input_data = input_data.unsqueeze(0)  # Equivalent to .view(1, 3, 224, 224)
    
    # Ensure the input is a float tensor
    input_data = input_data.float()
    
    # Make prediction
    output = model(input_data)
    
    # Detach from the graph and convert to numpy
    output = output.detach().numpy()
    
    # Get the index of the maximum output value (class prediction)
    index = np.argmax(output)
    
    return index

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
     uploaded_file = request.files.get('uploaded_file')
     captured_file = request.files.get('captured')
     print('request.files',request.files)

    if (captured_file and captured_file.filename != ''):
        file = captured_file  # Use captured file if it exists and has a filename
    elif (uploaded_file and uploaded_file.filename != ''):
        file = uploaded_file  # Use uploaded file if it exists and has a filename
    else:
        return "No file uploaded or captured!", 400

     # Save the file
    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    print('......file.....',file)
    random_number = random.randint(1000, 9999)

    file_path = os.path.join(upload_folder, f"{file.filename}_{random_number}")
    print('......file_path.....',file_path)
    file.save(file_path)
    pred = prediction(file_path)
    title = disease_info['disease_name'][pred]
    description =disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    image_url = disease_info['image_url'][pred]
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]
    return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)