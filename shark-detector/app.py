from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import zipfile
import uuid  # For unique file names
from model import detect_and_classify  # Import your ML function

app = Flask(__name__)

# Folder Configurations
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'zip'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        results = []

        if filename.lower().endswith('.zip'):
            # Extract zip and process each image
            extracted_folder = os.path.join(UPLOAD_FOLDER, filename[:-4])
            os.makedirs(extracted_folder, exist_ok=True)

            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extracted_folder)

            for extracted_file in os.listdir(extracted_folder):
                if allowed_file(extracted_file) and not extracted_file.lower().endswith('.zip'):
                    image_path = os.path.join(extracted_folder, extracted_file)
                    
                    # Process image
                    result = detect_and_classify(image_path)
                    
                    # Save output image with a unique name
                    output_filename = f"{uuid.uuid4()}.jpg"
                    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
                    os.rename(result["output_image"], output_path)  # Move processed image
                    
                    result["output_image"] = output_filename  # Store new file name
                    results.append(result)
        else:
            # Process single image
            result = detect_and_classify(filepath)

            # Save output image with a unique name
            output_filename = f"{uuid.uuid4()}.jpg"
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)
            os.rename(result["output_image"], output_path)  # Move processed image
            
            result["output_image"] = output_filename  # Store new file name
            results.append(result)

        return jsonify(results)
    return render_template('upload.html')


@app.route('/sharks')
def sharks():
    shark_data = [
        {'name': 'Great White Shark', 'info': 'One of the largest predatory fish in the world.'},
        {'name': 'Hammerhead Shark', 'info': 'Known for its distinctive head shape.'},
    ]
    return render_template('sharks.html', shark_data=shark_data)

@app.route('/processed/<filename>')
def get_processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
