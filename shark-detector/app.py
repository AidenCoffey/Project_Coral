from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import zipfile
from model import predict_shark  # Assume your model function is defined here

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
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(app.config['UPLOAD_FOLDER'])
                # Loop through extracted files
                for extracted_file in os.listdir(app.config['UPLOAD_FOLDER']):
                    if allowed_file(extracted_file) and not extracted_file.lower().endswith('.zip'):
                        image_path = os.path.join(app.config['UPLOAD_FOLDER'], extracted_file)
                        results[extracted_file] = predict_shark(image_path)
            else:
                # Process single image
                results[filename] = predict_shark(filepath)

            return jsonify(results)
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    return render_template('upload.html')

@app.route('/sharks')
def sharks():
    # You can load shark information from a database or static content
    shark_data = [
        {'name': 'Great White Shark', 'info': 'One of the largest predatory fish in the world.'},
        {'name': 'Hammerhead Shark', 'info': 'Known for its distinctive head shape.'},
        # Add more entries as needed
    ]
    return render_template('sharks.html', shark_data=shark_data)

if __name__ == '__main__':
    app.run(debug=True)
