from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import zipfile
from model import detect_and_classify  # Updated function for detection & classification

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'zip'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            results = {}
            if filename.lower().endswith('.zip'):
                # Extract zip and process each image
                extracted_folder = os.path.join(app.config['UPLOAD_FOLDER'], filename[:-4])
                os.makedirs(extracted_folder, exist_ok=True)

                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extracted_folder)

                for extracted_file in os.listdir(extracted_folder):
                    if allowed_file(extracted_file) and not extracted_file.lower().endswith('.zip'):
                        image_path = os.path.join(extracted_folder, extracted_file)
                        results[extracted_file] = detect_and_classify(image_path)
            else:
                # Process single image
                results[filename] = detect_and_classify(filepath)

            return jsonify(results)
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    return render_template('upload.html')

@app.route('/sharks')
def sharks():
    shark_data = [
        {'name': 'Great White Shark', 'info': 'One of the largest predatory fish in the world.'},
        {'name': 'Hammerhead Shark', 'info': 'Known for its distinctive head shape.'},
    ]
    return render_template('sharks.html', shark_data=shark_data)

@app.route('/output.jpg')
def get_output_image():
    return send_file("output.jpg", mimetype="image/jpeg")

if __name__ == '__main__':
    app.run(debug=True)
