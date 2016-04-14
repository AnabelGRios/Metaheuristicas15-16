import numpy as np
from sklearn import neighbors
from sklearn import cross_validation

# Función para obtener un subconjunto del conjunto inicial con todas las características, eliminando aquellas que no se vayan a utilizar,
# es decir, aquellas cuya posición en la máscara estén a False.
def getSubconjunto(conjunto, mascara):
	posiciones = np.arange(0,len(mascara))
	posiciones = posiciones[mascara]
	subconjunto = np.empty([len(conjunto), len(posiciones)], dtype=np.float32)
	i = 0
	for j in posiciones:
		subconjunto[:,i] = conjunto[:,j]
		i += 1

	return subconjunto


# Función para calcular la tasa utilizando el 3NN del módulo sklearn de python. Lo entrenamos con el conjunto de datos de entrenamiento
# datos_train y las características que estamos teniendo en cuenta, hacemos el leave one out para cada dato y obtemos la tasa de acierto
# en los demás. Finalmente devolvemos la media de las tasas
def calcularTasaKNNTrain(subconjunto, clases):
	suma_tasas = 0.0
	leave = cross_validation.LeaveOneOut(len(clases))
	knn = neighbors.KNeighborsClassifier(n_neighbors = 3)

	for train_index, test_index in leave:
		x_train, x_test = subconjunto[train_index], subconjunto[test_index]
		y_train, y_test = clases[train_index], clases[test_index]
		knn.fit(x_train, y_train)
		suma_tasas += 100*knn.score(x_test, y_test)

	tasa = suma_tasas / len(clases)
	return tasa


# Función para calcular la tasa utilizando el 3NN del módulo sklearn de python. Lo entrenamos con el conjunto de datos de entrenamiento
# datos_train y las características que estamos teniendo en cuenta y después obtenemos la tasa de acierto con el conjunto de test
def calcularTasaKNNTest(subconjunto, clases_test, datos_train, clases_train, mascara):
	knn = neighbors.KNeighborsClassifier(n_neighbors = 3)
	sub_train = getSubconjunto(datos_train, mascara)
	knn.fit(sub_train, clases_train)
	tasa = 100*knn.score(subconjunto, clases_test)
	return tasa
