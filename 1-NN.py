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

# Separar los datos de "train" en entrenamiento y prueba para probar los algoritmos
# 70% entrenamiento y 30% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print("Datos de entrenamiento y prueba creados")
print("Son {} datos para entrenamiento y {} datos para prueba".format(X_train.shape[0], X_test.shape[0]))

# Clasificador de 1-NN
class clasificador_1NN:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        
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

# Crear una instancia del clasificador 1-NN
one_nn = clasificador_1NN()

# Entrenar el clasificador con los datos de entrenamiento
one_nn.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred = one_nn.predict(X_test)

# Calcular la precisión del modelo en los datos de prueba
accuracy = sum(y_pred == y_test) / len(y_test)
print("Precisión 1-NN (H-O 70E/30P): {}".format(accuracy))

# Método de validación Leave-One-Out (LOO)
n_samples = len(X)
correct_predictions = 0

for i in range(n_samples):
    X_train_loo = np.delete(X, i, axis=0)
    y_train_loo = np.delete(y, i)
    X_test_loo = X[i].reshape(1, -1)
    y_test_loo = y[i]

    # Crear una instancia del clasificador 1-NN para LOO
    one_nn_loo = clasificador_1NN()

    # Entrenar el clasificador con los datos de entrenamiento para LOO
    one_nn_loo.fit(X_train_loo, y_train_loo)

    # Realizar predicciones en el dato de prueba para LOO
    y_pred_loo = one_nn_loo.predict(X_test_loo)

    # Comparar la etiqueta predicha con la etiqueta real para LOO
    if y_pred_loo[0] == y_test_loo:
        correct_predictions += 1

accuracy_loo = correct_predictions / n_samples
print("Precisión 1-NN (Leave-One-Out): {}".format(accuracy_loo))
