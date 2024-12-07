from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from imagesearch import ImageSearcher
import os
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

image_searcher = ImageSearcher(
    embeddings_path='image_embeddings.pickle',
    image_folder='coco_images_resized'
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query_type = request.form.get('query_type', 'text').strip()
    text_query = request.form.get('text_query', '').strip()
    weight = float(request.form.get('weight', 0.5))

    # For PCA, we only consider it if query_type == 'image'
    use_pca = False
    k = 50  # default value or max PCA components
    if query_type == 'image':
        use_pca = (request.form.get('use_pca') == 'on')
        # If PCA is used, get the value of k
        if use_pca:
            k = int(request.form.get('pca_k', '50'))
            # Ensure k does not exceed the precomputed PCA dimensions
            k = max(1, min(k, image_searcher.max_pca_components))

    img_file = request.files.get('image_query', None)

    img_embedding = None
    text_embedding = None

    if query_type == 'image':
        if img_file and img_file.filename != '':
            filename = secure_filename(img_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(filepath)
            img_embedding = image_searcher.encode_image(filepath, use_pca=use_pca, k=k)
        else:
            return jsonify({"error": "No image query provided."}), 400
        query_embedding = img_embedding

    elif query_type == 'text':
        if text_query:
            text_embedding = image_searcher.encode_text(text_query)
        else:
            return jsonify({"error": "No text query provided."}), 400
        query_embedding = text_embedding

    elif query_type == 'hybrid':
        # For hybrid, we ignore PCA since requirement states PCA is only for image queries
        if not text_query and (not img_file or img_file.filename == ''):
            return jsonify({"error": "Hybrid query selected, but no text or image provided."}), 400

        if text_query:
            text_embedding = image_searcher.encode_text(text_query)
        else:
            return jsonify({"error": "Hybrid query requires a text query."}), 400

        if img_file and img_file.filename != '':
            filename = secure_filename(img_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(filepath)
            # No PCA for hybrid queries
            img_embedding = image_searcher.encode_image(filepath, use_pca=False)
        else:
            return jsonify({"error": "Hybrid query requires an image query."}), 400

        combined = weight * text_embedding + (1 - weight) * img_embedding
        combined = combined / np.linalg.norm(combined)
        query_embedding = combined

    else:
        return jsonify({"error": "Invalid query type."}), 400

    results = image_searcher.search(query_embedding, top_k=5, use_pca=use_pca, k=k)
    return jsonify(results)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('coco_images_resized', filename)

if __name__ == '__main__':
    app.run(debug=True)
