import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split

# Función para calcular la distancia euclidiana entre dos puntos
def distancia_euclidiana(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

# Importar datos
iris = pd.read_csv("irisne.csv")

# Separar todas las columnas de las que quiero predecir
X = iris.drop('Species', axis=1).values
y = np.array(iris['Species'])

# Separar los datos de "train" en entrenamiento 
# y prueba para probar los algoritmos
# 70% entrenamiento y 30% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print("Datos de entrenamiento y prueba creados")
print("Son {} datos para entrenamiento y {} datos para prueba".format(X_train.shape[0], X_test.shape[0]))

# Clasificador KNN
class ClasificadorKNN:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for sample in X:
            distances = []
            for i, train_sample in enumerate(self.X_train):
                distance = distancia_euclidiana(sample, train_sample)
                distances.append((distance, self.y_train[i]))
            
            distances.sort(key=lambda x: x[0])
            neighbors = distances[:self.n_neighbors]
            neighbor_labels = [neighbor[1] for neighbor in neighbors]
            prediction = max(set(neighbor_labels), key=neighbor_labels.count)
            y_pred.append(prediction)
        return y_pred

# Pedir al usuario el número de vecinos a considerar
n_neighbors_input = int(input("Introduce el número de vecinos a considerar: "))

# Crear una instancia del clasificador KNN con el número de vecinos especificado
knn_classifier = ClasificadorKNN(n_neighbors=n_neighbors_input)

# Entrenar el clasificador con los datos de entrenamiento
knn_classifier.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred = knn_classifier.predict(X_test)

# Calcular la precisión del modelo en los datos de prueba
accuracy = np.mean(y_pred == y_test)
print("Precisión del clasificador KNN con {} vecinos: {}".format(n_neighbors_input, accuracy))