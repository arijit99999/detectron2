from flask import Flask, request, render_template
import os
import cv2
import io
import pickle
import base64
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# Initialize Flask app
app = Flask(__name__, template_folder='demo/')

# Load Detectron2 configuration
cfg = get_cfg()
cfg.merge_from_file(os.path.join('output_models', 'config.yaml'))  # Corrected line
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
cfg.MODEL.DEVICE = 'cpu'
predictor = DefaultPredictor(cfg)

# Set up upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/pred', methods=['POST'])
def marks():
    # Save the uploaded file
    file = request.files['image']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Load image using OpenCV
    image = cv2.imread(file_path)
    
    # Make prediction
    outputs = predictor(image)

    # Visualize results
    file_path=os.path.join(cfg.OUTPUT_DIR, "meta_data.pkl")
    with open(file_path,'rb') as file_obj:
     meta_data=pickle.load(file_obj)

    v = Visualizer(image[:, :, ::-1], metadata=meta_data, scale=.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Convert output image to PIL format
    segmented_image = Image.fromarray(out.get_image()[:, :, ::-1])
    buffered = io.BytesIO()
    segmented_image.save(buffered, format="JPEG")
    buffered.seek(0)

    # Encode the image to base64
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Render the result template
    return render_template('res.html', x=img_str)

if __name__ == '__main__':
    app.run(debug=True)
