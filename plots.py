import pandas as pd


#Importar datos
iris =pd.read_csv("iris.csv")

#Vemos los datos
#print(iris.head())
print("Informacion del data set")
print(iris.describe())
print("")
print("Distribucion de las especies de Iris")
print(iris.groupby('Species').size())

#Graficamos los datos
import matplotlib.pyplot as plt
## Grafica Sepal - Longitud vs Ancho
fig = iris[iris.Species == 'Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='blue', label='Setosa')
iris[iris.Species == 'Iris-versicolor'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='green', label='versicolor', ax=fig)
iris[iris.Species == 'Iris-virginica'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='red', label='virginica', ax=fig)
fig.set_xlabel("Longitud del Sepalo")
fig.set_ylabel("Ancho del Sepalo")
fig.set_title("Sepalo Longitud vs Ancho")
plt.show()

## Grafica Petalo - Longitud vs Ancho
fig = iris[iris.Species == 'Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='blue', label='Setosa')
iris[iris.Species == 'Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='green', label='versicolor', ax=fig)
iris[iris.Species == 'Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='red', label='virginica', ax=fig)
fig.set_xlabel("Longitud del Petalo")
fig.set_ylabel("Ancho del Petalo")
fig.set_title("Petalo Longitud vs Ancho")
plt.show()
