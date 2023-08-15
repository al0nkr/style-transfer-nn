import os
from flask import Flask, render_template, request
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import src.components.data_ingestion as DI
from src.components.model_trainer import modelTrain
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static', 'uploads'))
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
    epochs = int(request.form['epochs'])
    learning_rate = float(request.form['learningRate'])
    alpha = float(request.form['alpha'])
    beta = float(request.form['beta'])

    selected_source = request.form.get("imageSource")
    content_image = request.form.get('contentImage')
    style_image = request.form.get('styleImage')

    if selected_source == 'default':
        content_image_path = os.path.join(content_dir, content_image)
        style_image_path = os.path.join(style_dir, style_image)

    elif selected_source == 'custom_image':
        custom_content = request.files.get('customContentImage')
        content_image_filename = secure_filename(custom_content.filename)
        content_image_path = os.path.join(app.config['UPLOAD_FOLDER'], content_image_filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        print("Content Image Path:", content_image_path)
        custom_content.save(content_image_path)

        style_image_path = os.path.join(style_dir, style_image)

    elif selected_source == 'custom_style':
        custom_style = request.files.get('customStyleImage')
        style_image_filename = secure_filename(custom_style.filename)

        style_image_path = os.path.join(app.config['UPLOAD_FOLDER'], style_image_filename)
        print("Style Image Path:", style_image_path)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        custom_style.save(style_image_path)

        content_image_path = os.path.join(content_dir, content_image)
    
    elif selected_source == 'custom':
        custom_content = request.files.get('customContentImage')
        content_image_filename = secure_filename(custom_content.filename)
        content_image_path = os.path.join(app.config['UPLOAD_FOLDER'], content_image_filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        print("Content Image Path:", content_image_path)
        custom_content.save(content_image_path)

        custom_style = request.files.get('customStyleImage')
        style_image_filename = secure_filename(custom_style.filename)
        style_image_path = os.path.join(app.config['UPLOAD_FOLDER'], style_image_filename)
        print("Style Image Path:", style_image_path)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        custom_style.save(style_image_path)

    
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
