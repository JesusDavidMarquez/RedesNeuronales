import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

numCarpeta='19'
data_dir = 'anglesIA('+numCarpeta+')'
persons = os.listdir(data_dir)
w = 35
samples_per_person = 4

#Se cargan datos
def load_data():
    X, y = [], []
    for person_id, person in enumerate(persons):
        person_folder = os.path.join(data_dir, person)
        files = sorted(os.listdir(person_folder))[:samples_per_person]
        for file in files:
            df = pd.read_csv(os.path.join(person_folder, file))
            for start in range(len(df) - w):
                end = start + w
                X.append(df.iloc[start:end].values)
                y.append(person_id)
    return np.array(X), np.array(y)

X, y = load_data()

#Se normalizan los datos
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
joblib.dump(scaler, 'scaler.gz')

#Se codifican las etiquetas
y = to_categorical(y, num_classes=len(persons))

#Se dividen los datos en conjunto de prueba y de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Se crea el modelo LSTM
model = Sequential([
    LSTM(50, input_shape=(w, X.shape[2]), return_sequences=True),
    LSTM(50),
    Dense(len(persons), activation='softmax')
])

#Se compila el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

#Se entrena el modelo
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

#Se guarda el modelo
model.save('modeloPredecirPersonas.h5')

#Función de predicción
def predict_person(sample_path):
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
sample_path = 'anglesIA('+numCarpeta+')'+'/AndresMendoza/AndresMendoza5_combined.csv'
result = predict_person(sample_path)
print("La persona es:", result)
sample_path = 'anglesIA('+numCarpeta+')'+'/NoeMarcelino/NoeMarcelino5_combined.csv'
result = predict_person(sample_path)
print("La persona es:", result)
sample_path = 'anglesIA('+numCarpeta+')'+'/Moki/Moki5_combined.csv'
result = predict_person(sample_path)
print("La persona es:", result)
sample_path = 'PersonasDesconocidas/Yuliana('+numCarpeta+')/Yuliana4_combined.csv'
result = predict_person(sample_path)
print("La persona es:", result)



