from django.shortcuts import render,HttpResponse
from django.core.mail import send_mail
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# Create your views here.
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import PhishingDataset
from .forms import PhishingForm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import GRU, Dense
import pandas as pd
import numpy as np
# Load the phishing dataset
def load_dataset():
    dataset = PhishingDataset.objects.all()
    data = pd.DataFrame(list(dataset.values()))
    return data

# Preprocess the data
def preprocess_data(data):
    X = data['url']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, y_train, y_test

# Train the RNN-GRU model
def train_rnn_gru(X_train, y_train):
    model = Sequential()
    model.add(GRU(64, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return model

# Make predictions
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# Evaluate the model
def evaluate_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Phishing detection view
def Phishing(request):
    if request.method == 'POST':
        form = PhishingForm(request.POST)
        if form.is_valid():
            url = form.cleaned_data['url']
            dataset = load_dataset()
            X_train, X_test, y_train, y_test = preprocess_data(dataset)
            model = train_rnn_gru(X_train, y_train)
            predictions = make_predictions(model, X_test)
            accuracy = evaluate_model(y_test, predictions)
            result = model.predict([url])
            if result[0] > 0.5:
                return HttpResponse('Phishing URL detected!')
            else:
                return HttpResponse('Legitimate URL detected!')
    else:
        form = PhishingForm()
    return render(request, 'phish.html', {'form':form})
def about(request):
    return render(request,"about.html")
def contact(request):
    if(request.method == "POST"):
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')
        send_mail(
            'Contact Form Submission',
            f'Name: {name}\nEmail: {email}\nMessage: {message}',
            'tarani06sure@gmail.com',
            ['tarani06sure@gmail.com'],
            fail_silently=False,
        )

        return HttpResponse('Thank you for contacting us!')
    else:
        return render(request, 'contact.html')
def index(request):
    return render(request,"index.html")
def About(request):
    return render(request,"Abt.html")
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# Load the dataset
tweets_data = pd.read_csv('dataset2.csv')

# Define the machine learning models
models = {
    'Gaussian Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Adaboost Classifier': AdaBoostClassifier(),
    'Random Forest Classifier': RandomForestClassifier()
}

# Define a function to train and evaluate each model
def train_and_evaluate(model):
    X = tweets_data.drop('label', axis=1)
    y = tweets_data['label']
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    matrix = confusion_matrix(y, y_pred)
    return accuracy, report, matrix

# Define a view to train and evaluate all models
def train_all_models(request):
    results = {}
    for name, model in models.items():
        accuracy, report, matrix = train_and_evaluate(model)
        results[name] = {
            'accuracy': accuracy,
            'report': report,
            'matrix': matrix
        }
    return JsonResponse(results)

# Define a view to predict whether a tweet is cyberbullying or not
def predict_tweet(request):
    tweet_text = request.GET.get('tweet_text')
    X = pd.DataFrame({'text': [tweet_text]})
    predictions = {}
    for name, model in models.items():
        y_pred = model.predict(X)
        predictions[name] = y_pred[0]
    return JsonResponse(predictions)
def home(request):
    return render(request,"home.html")

def Cntact(request):
    if(request.method == "POST"):
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')
        send_mail(
            'Contact Form Submission',
            f'Name: {name}\nEmail: {email}\nMessage: {message}',
            'tarani06sure@gmail.com',
            ['tarani06sure@gmail.com'],
            fail_silently=False,
        )

        return HttpResponse('Thank you for contacting us!')
    else:
        return render(request, 'Cntact.html')