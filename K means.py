import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

"""
Imaginemos que estamos trabajando con un conjunto de datos de 
clientes de una tienda en línea, donde cada punto representa 
la actividad de compra de un cliente. Los dos atributos que 
consideraremos son el "monto total gastado" y la "frecuencia 
de las compras" durante el último año.

Utilizaremos estos dos atributos para agrupar a los clientes 
en segmentos que podrían representar diferentes tipos de 
comportamientos de compra, como clientes habituales, ocasionales, 
grandes gastadores, etc.
"""

#Se generan datos con 'make_blobs' donde cada centro representará un tipo de comportamiento de compra
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.5, random_state=42)

#Las columnas representan 'Total Gastado' y 'Frecuencia de Compras'
#Se escala para representar el gasto en dólares
X[:, 0] = (X[:, 0] - X[:, 0].min()) * 50   
#Se escala para representar la cantidad de compras en el año
X[:, 1] = np.abs(X[:, 1]) * 5              

#Se visualizan los datos simulados de clientes
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolor='k')
plt.xlabel('Total Gastado ($)')
plt.ylabel('Frecuencia de Compras (veces/año)')
plt.title("Distribución Simulada de la Actividad de Clientes")
plt.grid(True)
plt.show()

#Se aplica el algoritmo k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

#Se visualizan los grupos formados
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', edgecolor='k', alpha=0.7)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=250, marker='*', edgecolor='k')
plt.xlabel('Total Gastado ($)')
plt.ylabel('Frecuencia de Compras (veces/año)')
plt.title("Clustering de Clientes con K-Means")
plt.grid(True)
plt.show()


