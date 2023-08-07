import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import src.components.data_ingestion as DI
from src.components.model_trainer import modelTrain

app = Flask(__name__)
content_dir = DI.default_content_dir
style_dir = DI.default_style_dir

@app.route('/')
def index():
    # Get content and style image filenames from src/components/data directory
    content_images = [f for f in os.listdir(content_dir) if f.endswith('.jpg' or '.png' or '.jpeg')]
    style_images = [f for f in os.listdir(style_dir) if f.endswith('.jpg' or '.png' or '.jpeg')]

    return render_template('index.html', content_images=content_images, style_images=style_images)

@app.route('/transfer', methods=['POST'])
def transfer_style():
    # Retrieve user input from the form
    content_image = request.form['contentImage']
    style_image = request.form['styleImage']
    epochs = int(request.form['epochs'])
    learning_rate = float(request.form['learningRate'])
    alpha = float(request.form['alpha'])
    beta = float(request.form['beta'])

    # Construct the file paths for the content and style images
    content_image_path = os.path.join(content_dir, content_image)
    style_image_path = os.path.join(style_dir, style_image)

    # Perform style transfer
    test = modelTrain(content_image_path, style_image_path)
    generated_image = test.train(epochs=epochs, lr=learning_rate, alpha=alpha, beta=beta)

    # Convert the generated image to base64 and pass it to the template
    buffer = BytesIO()
    plt.imshow(generated_image)
    plt.axis('off')
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return render_template('result.html', img_data=img_str)

if __name__ == "__main__":
    app.run(debug=True)
