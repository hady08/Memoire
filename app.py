from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import pytesseract
import os
import Levenshtein

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'

# Charger le modèle de détection de carte d'identité
model = load_model("modele_cni.keras")

# Définir le chemin vers l'exécutable Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Pour Windows
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Pour macOS/Linux

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    alpha = 1.5  # Facteur de contraste
    beta = 0     # Valeur de luminosité
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    thresh = cv2.adaptiveThreshold(adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def extract_text(image_path):
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is None:
        return "Erreur lors du pré-traitement de l'image."
    
    text = pytesseract.image_to_string(preprocessed_image, lang='eng')
    return text

def is_match(expected_list, actual, max_distance=2):
    """Compare list of expected strings and return True if any matches are found within max_distance."""
    for expected in expected_list:
        distance = Levenshtein.distance(expected.lower(), actual.lower())
        if distance <= max_distance:
            return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence_score = np.max(predictions) * 100

        if predicted_class == 0:  # La classe 0 représente les cartes d'identité
            return jsonify({
                'result': 'Carte d\'identité détectée.',
                'confidence': f'{confidence_score:.2f}%'
            })
        else:
            return jsonify({
                'result': 'Ce n\'est pas une carte d\'identité.',
                'confidence': f'{confidence_score:.2f}%'
            })

@app.route('/read', methods=['POST'])
def read():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        text = extract_text(file_path)
        return jsonify({
            'result': text
        })

@app.route('/compare', methods=['POST'])
def compare():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'})

    if 'nom' not in request.form or 'prenom' not in request.form or 'date_naissance' not in request.form:
        return jsonify({'error': 'Missing data for comparison'})

    nom = request.form['nom']
    prenom = request.form['prenom']
    date_naissance = request.form['date_naissance']

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        text = extract_text(file_path)

        noms = nom.split()
        prenoms = prenom.split()

        nom_found = any(is_match(noms, word) for word in text.split())
        prenom_found = any(is_match(prenoms, word) for word in text.split())
        date_naissance_found = any(is_match([date_naissance], word) for word in text.split())

        return jsonify({
            'nom_found': nom_found,
            'prenom_found': prenom_found,
            'date_naissance_found': date_naissance_found,
            'text': text
        })

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
