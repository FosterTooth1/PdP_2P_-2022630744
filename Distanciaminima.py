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
print("Son {} datos para entrenamiento y {} datos para prueba"
      .format(X_train.shape[0], X_test.shape[0]))

# Clasificador de distancia mínima
class ClasificadorDistanciaMinima:
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for sample in X:
            min_distance = float('inf')
            nearest_label = None
            for i, train_sample in enumerate(self.X_train):
                distance = distancia_euclidiana(sample, train_sample)
                if distance < min_distance:
                    min_distance = distance
                    nearest_label = self.y_train[i]
            y_pred.append(nearest_label)
        return y_pred

# Crear una instancia del clasificador de distancia mínima
min_distance = ClasificadorDistanciaMinima()

# Entrenar el clasificador con los datos de entrenamiento
min_distance.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred = min_distance.predict(X_test)

# Calcular la precisión del modelo en los datos de prueba
accuracy = np.mean(y_pred == y_test)
print("Precisión de la clasificación de distancia mínima: {}"
      .format(accuracy))


# Método de validación Leave-One-Out (LOO)
n_samples = len(X)
correct_predictions_loo = 0

for i in range(n_samples):
    X_train_loo = np.delete(X, i, axis=0)
    y_train_loo = np.delete(y, i)
    X_test_loo = X[i].reshape(1, -1)
    y_test_loo = y[i]

    # Crear una instancia del clasificador de mínima distancia para LOO
    min_distance_loo = ClasificadorDistanciaMinima()

    # Entrenar el clasificador con los datos de entrenamiento para LOO
    min_distance_loo.fit(X_train_loo, y_train_loo)

    # Realizar predicciones en el dato de prueba para LOO
    y_pred_loo = min_distance_loo.predict(X_test_loo)

    # Comparar la etiqueta predicha con la etiqueta real para LOO
    if y_pred_loo[0] == y_test_loo:
        correct_predictions_loo += 1

accuracy_loo = correct_predictions_loo / n_samples
print("Precisión Distancia Minima (Leave-One-Out): {}"
      .format(accuracy_loo))


