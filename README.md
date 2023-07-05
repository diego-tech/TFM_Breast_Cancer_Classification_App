# TFM Breast Cancer Classification App

## Abstract
Breast cancer is positioned as the most prevalent type of malignant tumor in the female population. It is of utmost importance to recognize the relevance of early detection of this disease through the implementation of methods such as mammography and other techniques, since their application has a direct impact on the patient's prognosis.
In this sense, advances in the field of artificial intelligence have demonstrated their ability to contribute to the early detection of this type of tumor, focusing particularly on invasive ductal carcinoma (IDC), which is the most common form of breast cancer. This type of cancer originates in the lining of the lactiferous ducts of the breast and spreads beyond these ducts to other breast tissues.
By using convolutional neural networks in IDC imaging, greater accuracy and sensitivity in identifying potential tumors is achieved. These networks are able to thoroughly analyze the details of images from scans and mammograms, highlighting suspicious regions and providing a more reliable assessment.

## Components
* Álvaro Ladrón de Guevara Garcés
* Ángel Martín Heras
* Diego Muñoz Herranz
## Environment Setup

### Windows
1. Download and install Python from the official website: https://www.python.org/downloads/
2. Open the Command Prompt
3. Navigate to the project's root folder using the `cd` command
4. Create a virtual environment by running the following command:
```py
python -m venv venv
```
5. Activate the virtual environment with the following command:
```bash
venv\Script\activate
```
6. Install the dependencies listed in the requirements.txt file using the following command:
```py
pip install -r requirements.txt
```
7. Run the Flask program with the following command:
```py
python run.py
```

### macOS

1. Open the Terminal
2. Navigate to the project's root folder using the `cd` command
3. Create a virtual environment by running the following command:
```py
python3 -m venv venv
```
4. Activate the virtual environment with the following command:
```bash
source venv/bin/activate
```
5. Install the dependencies listed in the requirements.txt file using the following command:
```py
pip install -r requirements.txt
```
6. Run the Flask program with the following command:
```py
python3 run.py
```

## Usage
You can find the model files in the following link (https://1drv.ms/u/s!ApTJouCVOUYZlJYYG2ODo6jwgZYLiA?e=2fttC5)*, once you have downloaded the zip file, be sure to unzip the file in the `\model_data` folder so that it is structured as follows:
```
├── model_data
    ├── ClassicCNN_0010_20.pth
    ├── ResNet50__30.h5
    └── weights.best_30.hdf5
```

*_If you are not allowed to do so, please contact us._

Once the program has started, you can enter your image to be classified and by clicking on the predict button, the prediction for the models with the selected image will appear.

## Annexes
This website is related to the repository (https://github.com/diego-tech/TFM_Breast_Cancer_Classification) which hosts the code of the models.