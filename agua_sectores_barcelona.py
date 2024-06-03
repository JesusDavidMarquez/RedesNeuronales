import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

data = pd.read_csv('demanda_agua(normalizado).csv')

#Se selecciona el sector 5 de Barcelona
output_column = 'sector5'
y = data[output_column]

#Función para crear las secuencias
def create_sequences(y, w):
    ys = []
    for i in range(len(y) - w):
        ys.append(y.iloc[i:(i + w + 1)].values)
    return np.array(ys)

#Tamaño de la ventana de tiempo
w = 15
y_seq = create_sequences(y, w)

#Ajuste para tener 'X' y 'y' desde y_seq ('X' son todos menos el último de cada secuencia, 'y' es el último)
X_seq = y_seq[:, :-1]
y_seq = y_seq[:, -1]

#Se dividen los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)

#Se crea el modelo secuencial con una capa LSTM
model = Sequential([
    LSTM(100, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dense(300, activation='relu'),
    Dense(200, activation='relu'),
    Dense(1)
])

#Se configura el optimizador Adam
optimizer = Adam(learning_rate=0.001)

#Se compila el modelo
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

#Se entrena el modelo
history = model.fit(X_train, y_train, epochs=500, validation_split=0.2, batch_size=32)

#Se hacen predicciones con el conjunto de prueba
y_pred = model.predict(X_test)

#Se asegura que y_test y y_pred tienen la misma longitud y forma
y_test = y_test.flatten()
y_pred = y_pred.flatten()

#Gráfico de la diferencia entre los valores reales y los predichos
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Valores Reales')
plt.plot(y_pred, label='Valores Predichos', linestyle='--')
plt.title('Comparación de Valores Reales vs Valores Predichos')
plt.xlabel('Número de muestra')
plt.ylabel('Demanda de agua')
plt.ylim(0, 1)
plt.legend()
plt.show()



