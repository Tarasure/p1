from django.shortcuts import render
# Create your views here.
def Phishing(request):
    if(request.method == "POST"):
        inpt = request.POST

        if("predict" in request.POST):
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from tensorflow import keras
            from keras import layers

            # Load and preprocess data
            data = pd.read_csv('dataset.csv')
            X = data.drop(columns=['index', 'Target'])
            y = data['Target']

            # Split data into training and validation sets
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.75)

            # Define input shape
            input_shape = [X_train.shape[1], 1]  # Assuming univariate time series

            # Reshape data for RNN input
            X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
            X_valid = np.reshape(X_valid.values, (X_valid.shape[0], X_valid.shape[1], 1))

            # Define the RNN-GRU model
            model = keras.Sequential([
                layers.GRU(256, return_sequences=True, input_shape=input_shape),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.GRU(128),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')
            ])

            # Compile the model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['binary_accuracy', keras.metrics.Precision(), keras.metrics.Recall()],
            )

            # Define early stopping callback
            early_stopping = keras.callbacks.EarlyStopping(
                patience=20,
                min_delta=0.01,
                restore_best_weights=True,
            )

            # Train the model
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_valid, y_valid),
                batch_size=128,
                epochs=200,
                callbacks=[early_stopping],
            )

            return render(request, "phish.html", context = {"Result": result})
    return render(request, "phish.html")
