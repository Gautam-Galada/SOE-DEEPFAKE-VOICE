from flask import Flask, request, render_template
import speech_recognition as sr
import aiml
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import nltk
nltk.download("punkt")
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


# Importing Libraries needed for Tensorflow processing
import tensorflow as tf   #version 1.13.2
import numpy as np
import tflearn            #version 0.3.2
import random
import json

with open("intents.json") as json_data:
    intents = json.load(json_data)

kernel = aiml.Kernel()

tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
classifier = pipeline('text-classification', model=model, tokenizer = tokenizer)

words = []
documents = []
classes = []
ignore = ["?"]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])


words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

net = tflearn.input_data(shape=[None, 36])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 5, activation="softmax")
net = tflearn.regression(net)

#Defining Model and setting up tensorboard
model = tflearn.DNN(net)

model.load('./models/model.tflearn')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words= [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s"% w)
    return(np.array(bag))

context = {}
ERROR_THRESHOLD = 0.25
def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        return random.choice(i['responses'])
            results.pop(0)



global rec_text
rec_text=''
app = Flask(__name__)
r = sr.Recognizer()


def display_video(omh):
    text = str(omh)
    print(type(text))
    text_1 = classifier(text)
    for t in text_1:
        g = t['score']
        print(t['score'])
    if g >= 0.54:
        try:
            if text == "SOE-1":
                path = "A1.mp4"
            elif text == "SOE-2":
                path = "A2.mp4"
            elif text ==  "SOE-3":
                path = "A3.mp4"
            else :
                path = "A4.mp4"
        except:
                path = "A5.mp4"
    else:
        path = "A6.mp4"

    return path


def micro(audio_text):
    print("Text: "+r.recognize_google(audio_text))
    rec_text = r.recognize_google(audio_text, language='en-US').lower()

    return rec_text


@app.route('/')
def my_form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    with sr.Microphone() as source:
        print("Ask the Acharya")
        #r.adjust_for_ambient_noise(source,duration=5)
        audio_text = r.listen(source, phrase_time_limit=5)
        #print(audio_text)
        print("Question recorded, thanks!")

    omh = micro(audio_text)
    repl = str(response(omh))

    print(repl)
    filename = display_video(repl)
    #print(repl)
    #rec_text = str(omh)
    return render_template('form.html',text1=repl, filename=filename)


if __name__ == "__main__":
    app.run()

