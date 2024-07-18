from flask import Flask, render_template, request, redirect, url_for
import os
from model import TrashClassifier

app = Flask(__name__)
UPLOAD_FOLDER = 'static/img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Instantiate the TrashClassifier
trash_classifier = TrashClassifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        print("There is a file there")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        image = trash_classifier.preprocess_image(file_path, (224, 224))
        prediction = trash_classifier.predict(image)
        return render_template('result.html', image_url=file.filename, prediction=prediction[0])
    return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)
