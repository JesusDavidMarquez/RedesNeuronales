import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

data_dir = 'anglesIA(19)'
w = 35

#Se carga el modelo ya entrenado y el escalador
model = load_model('modeloPredecirPersonas.h5')
scaler = joblib.load('scaler.gz')

#Función de predicción
def predict_person(sample_path):
    persons = os.listdir(data_dir)
    scaler = joblib.load('scaler.gz')
    df_sample = pd.read_csv(sample_path)
    X_sample = df_sample.values[:w]
    X_sample = scaler.transform(X_sample.reshape(-1, X_sample.shape[-1])).reshape(1, w, -1)
    predictions = model.predict(X_sample)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    if confidence < 0.8:
        return "Persona desconocida"
    else:
        return persons[predicted_class]

#Se hacen pruebas
sample_path = data_dir+'/AndresMendoza/AndresMendoza5_combined.csv'
result = predict_person(sample_path)
print("La persona es:", result)
sample_path = data_dir+'/NoeMarcelino/NoeMarcelino5_combined.csv'
result = predict_person(sample_path)
print("La persona es:", result)
sample_path = data_dir+'/Moki/Moki5_combined.csv'
result = predict_person(sample_path)
print("La persona es:", result)
sample_path = 'PersonasDesconocidas/Yuliana(19)/Yuliana4_combined.csv'
result = predict_person(sample_path)
print("La persona es:", result)
