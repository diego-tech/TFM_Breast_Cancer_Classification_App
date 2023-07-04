from flask import Flask, render_template, request, jsonify
import os
import warnings
from datetime import datetime

# Custom Imports
from app.src.data.preprocess_images import *
from app.src.models.predict import *
from app.src.utils.loadmodel import *

# Remove warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# When Running this app on the Local machine, default port to 8000
port = int(os.getenv("PORT", 8080))

# Context Processors


@app.context_processor
def date_now():
    return {
        'proyect_name': 'Invasive Ductal Carcinoma Classification',
        'now': datetime.utcnow()
    }


@app.route("/", methods=["GET", "POST"])
def root():
    if request.method == 'POST':
        if 'predictImage' in request.files:
            image = request.files['predictImage']
            processed_image = preprocess_image(image)

            # Classic CNN Prediction
            processed_classic_cnn_image = processed_classic_cnn_image_function(
                image)
            classic_cnn_prediction = classicCNN_Predict(
                data=processed_classic_cnn_image, model_path=classic_cnn_path)

            classic_cnn_result = {
                'prediction': classic_cnn_prediction
            }

            # ResNet50 Prediction
            resnet50_predition = resnet50_Predict(
                data=processed_image, model=resenet50_model)
            resnet50_result = {
                'prediction': resnet50_predition
            }

            # DenseNet201 Prediction
            densenet201_prediction = denseNet201_Predict(
                data=processed_image, model=densenet_model)
            densenet201_result = {
                'prediction': densenet201_prediction
            }

            result = {
                'classic_cnn': classic_cnn_result,
                'resnet50': resnet50_result,
                'densenet201': densenet201_result
            }

            return jsonify(
                {
                    'result': result,
                    'image': rescale_image(image_path=image)
                }
            )

    return render_template(
        'index.html'
    )


# main
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=True)
