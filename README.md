# Speech Recognition and Chatbot Application

(Note: The following project was a collaborative effort between the dignitaries from the Statue of Equality and my team, which included Sathya Vemulapalli, Pranav Kompally, and myself, Gautam Galada.)
This application utilizes Flask, speech recognition, and a chatbot to enable speech-based interaction with users. The chatbot is trained using AIML (Artificial Intelligence Markup Language) and leverages the transformers library for text classification. The application can recognize speech input, process it using the chatbot, and provide a response or perform a specific action based on the user's query.

## Prerequisites

- Python 3.x
- Flask
- SpeechRecognition
- aiml
- nltk
- tflearn
- tensorflow
- transformers

## Installation

1. Clone the repository or download the code files.
2. Install the required Python packages using the following command:

```bash
pip install flask SpeechRecognition aiml nltk tflearn tensorflow transformers
```

# Chatbot with Speech Recognition

This repository contains a chatbot application that utilizes speech recognition to process user input and generate responses. The application is built using Flask, AIML, and various natural language processing libraries.

## Usage

1. Ensure that you have a directory named "brain" containing AIML files for training the chatbot.
2. Run the application using the following command:

```bash
python your_script_name.py
```

3. Access the application by visiting http://localhost:5000 in your web browser.
4. Click on the microphone icon and ask a question or provide a query.
5. The application will process the speech input, use the chatbot to generate a response, and display the response on the web page.
6. The application may also play a specific video based on the query.

## Code Overview

- The Flask web framework is used to create the application.
- The `app` Flask object is created, and routes are defined to handle HTTP requests.
- The `my_form` route renders an HTML form template (`form.html`) that contains a microphone button for speech input.
- The `my_form_post` route handles the form submission and processes the speech input.
- The `micro` function uses the SpeechRecognition library to convert the speech input into text.
- The text is then passed to the chatbot for processing using the `response` function.
- The chatbot generates a response based on the input text.
- The `display_video` function determines the appropriate video to play based on the chatbot's response.
- The response and video filename are rendered in the `form.html` template.

## Additional Notes

- The application utilizes AIML files in the "brain" directory to train the chatbot. Ensure that you have the necessary AIML files for the desired chatbot behavior.
- The `intents.json` file is used to define intents and patterns for the chatbot training process.
- The `Hate-speech-CNERG/bert-base-uncased-hatexplain` model is used for hate speech classification.
- The `tflearn` library is used to define and load a trained model for text classification.
- The application utilizes the `transformers` library for tokenization and sequence classification using the BERT model.
- The `nltk` library is used for word tokenization and stemming.
- The `speech_recognition` library provides speech-to-text functionality.
- The application plays specific videos based on predefined conditions.
- The application runs on the local Flask development server.

## Acknowledgements

- This code was inspired by various speech recognition and chatbot tutorials and examples available online.
- The AIML files and training data for the chatbot are not included in this code but can be created based on specific requirements.

## Disclaimer

- This code serves as an example and may require additional modifications to suit your specific use case.
- The chatbot's behavior and responses are dependent on the training data and AIML files used.
- Ensure that you comply with the terms and conditions of the libraries and models used in this code.
- Use the application responsibly and ensure that it adheres to applicable laws and regulations.

