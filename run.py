from flask import Flask, render_template, request, jsonify
import os
import warnings
from datetime import datetime

# Custom Imports
from app.src.data.preprocess_images import preprocess_image
from app.src.models.predict import denseNet201_Predict
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

            prediction = denseNet201_Predict(processed_image, densenet_model)

            prediction_list = prediction.tolist()

            print(prediction_list)

            return jsonify({'prediction': prediction_list})

    return render_template(
        'index.html'
    )


# main
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=True)
