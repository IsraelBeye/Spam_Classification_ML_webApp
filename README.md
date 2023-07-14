# Machine Learning Spam Classification Web App

This repository contains a web application for classifying SMS messages as spam or not using machine learning. The application is built using Flask and deployed on a web server.

## Table of Contents
- [Description](#description)
- [Usage](#usage)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Description

The web app uses a machine learning model trained on a dataset of SMS messages to classify incoming text messages as spam or not. It leverages the power of Natural Language Processing (NLP) and TF-IDF vectorization to convert the text into numerical features and make predictions. The application is built using the Flask framework, which allows for easy development and deployment of web applications.

## Files in the Repository

- `spam_classification.py`: Python script to read the `SPAM_text.csv` file, preprocess the data, train various machine learning models, and save the best model and vectorizer for prediction.
- `app.py`: Flask web application script that loads the trained model and vectorizer, and provides the API endpoints for prediction.
- `templates/index.html`: HTML template file for the home page of the web app.
- `templates/prediction.html`: HTML template file for displaying the prediction results.

## Usage

To use the web app, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask application using `python app.py`.
4. Access the web app through your web browser at `http://localhost:5000`.
5. Enter the text message you want to classify in the provided input box.
6. Click the "Classify" button to get the prediction result.

## Installation

To install and run the application locally, follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/IsraelBeye/Spam_Classification_ML_webApp.git
1. Navigate to the project directory:
cd spam-classification-web-app

2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate
3. Install the required dependencies:
pip install -r requirements.txt

4. Run the spam_classification.py script to train the model and save the best model and vectorizer:
   python spam_classification.py
5. Start the Flask application:

python app.py

6.Access the web app through your web browser at http://localhost:5000.

Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for more details.



