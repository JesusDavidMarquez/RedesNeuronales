import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('datos_frames(normalizado).csv')
df['Nombre imagen'] = df['Nombre imagen'].apply(lambda x: x + '.png')

#Se configura el generador de imágenes
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

#Se generan los datos
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='img_algas',
    x_col='Nombre imagen',
    y_col='Biomasa g/L',
    subset='training',
    class_mode='raw',
    target_size=(150, 150),
    batch_size=32
)

valid_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory='img_algas',
    x_col='Nombre imagen',
    y_col='Biomasa g/L',
    subset='validation',
    class_mode='raw',
    target_size=(150, 150),
    batch_size=32,
    shuffle=False
)

#Se construye el modelo CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

#Se compila el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

#Se entrena el modelo
model.fit(train_generator, validation_data=valid_generator, epochs=10)

#Se realizan predicciones en el conjunto de validación
valid_generator.reset()
predicted_biomass = model.predict(valid_generator, steps=len(valid_generator))

#Se recogen las etiquetas reales de Biomasa g/L
all_real_labels = valid_generator.labels

#Se normalización los valores predichos y reales
min_val = min(min(all_real_labels), min(predicted_biomass))
max_val = max(max(all_real_labels), max(predicted_biomass))

normalized_real_labels = (all_real_labels - min_val) / (max_val - min_val)
normalized_predicted_biomass = (predicted_biomass.flatten() - min_val) / (max_val - min_val)

#Gráfico de la Biomasa g/L real vs la Biomasa g/L predicha
plt.figure(figsize=(10, 6))
plt.scatter(range(len(normalized_real_labels)), normalized_real_labels, color='blue', label='Biomasa g/L Real')
plt.scatter(range(len(normalized_predicted_biomass)), normalized_predicted_biomass, color='red', label='Biomasa g/L Predicha', alpha=0.5)
plt.title('Biomasa g/L Real vs. Biomasa g/L Predicha')
plt.xlabel('Número de Muestra')
plt.ylabel('Biomasa g/L')
plt.ylim(0, 1)
plt.legend()
plt.show()




