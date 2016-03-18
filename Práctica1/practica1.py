from scipy.io import arff
import numpy as np
from math import sqrt
import time

datos = arff.loadarff('wdbc.arff')

# Hay que elegir primero la mitad de los datos para trabajar con ellos y guardarlos junto con su clase
# Extraemos aleatoriamente la mitad de los datos
tope = len(datos[0]) // 2
perm = np.random.permutation(len(datos[0]))
tipos_clase = np.array([b'B', b'M'])
posiciones = perm[0:tope]
posiciones2 = perm[tope:len(datos[0])]
# Pasamos los datos a un array de numpy, quitándole la clase, que estaba en la primera posición
train = np.array([[datos[0][i][j] for j in range(1,len(datos[0][0]))] for i in posiciones],float)
# Nos quedamos ahora con la clase de estos mismos datos, en un array distinto
clase_train = np.array([datos[0][i][0] for i in posiciones])

# Pasamos los datos de test a un numpy sin la clase_train
test = np.array([[datos[0][i][j] for j in range(1,len(datos[0][0]))] for i in posiciones2],float)
# Nos quedamos ahora con la clase de estos mismos datos, en un array distinto
clase_test = np.array([datos[0][i][0] for i in posiciones2])

# Normalizamos los datos y los dejamos entre 0 y 1
minimo = train.min()
maximo = train.max()
for row in train:
	for element in row:
		element = (element - minimo) / (maximo - minimo)


# Función para obtener la distancia euclídea de dos datos, dados los dos datos
def getDistancia(dato1, dato2, mascara):
	vec1 = dato1*mascara
	vec2 = dato2*mascara
	dif = np.subtract(vec1, vec2)
	cuadrados = np.power(dif, 2)
	distancia = sum(cuadrados)
	return sqrt(distancia)


# Función para obtener las posiciones de los 3 vecinos más cercanos junto con sus distancias
# según las características que nos diga la máscara que hay que mirar
def get3NN(entrenamiento, dato, mascara):
	# Creamos un vector con las posiciones donde están los 3 vecinos más cercanos en cada momento
	vec3NN = np.array([0, 1, 2])
	# Creamos también un vector con las distancias euclídeas que hay a esas posiciones
	distancias3NN = np.array([getDistancia(dato, entrenamiento[0,], mascara), getDistancia(dato, entrenamiento[1,], mascara),
		getDistancia(dato, entrenamiento[2,], mascara)])

	for i in range(3, len(entrenamiento)):
		dis_actual = getDistancia(dato, entrenamiento[i,], mascara)
		if dis_actual < distancias3NN.max():
			pos_max = distancias3NN.argmax()
			vec3NN[pos_max] = i
			distancias3NN[pos_max] = dis_actual

	# Devolvemos una lista donde la primera componente son las posiciones y la segunda las distancias
	ret = [vec3NN, distancias3NN]
	return ret


# Función para obtener la clase de un dato, obteniendo previamente sus tres vecinos más cercanos junto con sus distancias
def getClasePunto(entrenamiento, dato, mascara, clases):
	ret = get3NN(entrenamiento, dato, mascara)
	vec3NN = ret[0]
	distancias3NN = ret[1]

	# Hacemos un vector con las clases a las que pertenecen los tres vecinos más cercanos
	clases3NN = np.array([clases[vec3NN[0]], clases[vec3NN[1]], clases[vec3NN[2]]])

	# # Contamos el número de veces que aparecen las dos clases en este vector
	# Bs = sum(clases3NN == tipos_clase[0])
	# Ms = sum(clases3NN == tipos_clase[1])
	#
	# # Nos quedamos con el que más apariciones tenga
	# if Bs > Ms:
	# 	clase = b'B'
	# else:
	# 	clase = b'M'

	# Contamos el número de veces que aparece cada clase y nos quedamos con el máximo
	clase = 0
	maximo = sum(clases3NN == tipos_clase[0])
	for i in range(1, len(tipos_clase)):
		loc = sum(clases3NN == tipos_clase[i])
		if loc > maximo:
			maximo = loc
			clase = i

	# Si el maximo es 1, entonces es porque hay un empate. Nos quedamos con el que esté más cerca
	if maximo == 1:
		clase = distancias3NN.argmin()

	clase = tipos_clase[clase]

	return clase


# Función para obtener la clase de un conjunto de datos, distinguiendo entre si es el de entrenamiento o no,
# puesto que si es el de entrenamiento habrá que quitar al calcular los vecinos el dato en concreto al que
# se le calcula la clase.
def getClases(conjunto, is_train, mascara, clase_train):
	clases_deducidas = np.empty(len(conjunto), '|S1')
	for i in range(0, len(conjunto)):
		if is_train:
			entrenamiento = np.delete(conjunto, i, 0)
			clases = np.delete(clase_train, i, 0)
		else:
			entrenamiento = train
			clases = clase_train

		dato = conjunto[i]
		clases_deducidas[i] = getClasePunto(entrenamiento, dato, mascara, clases)

	return clases_deducidas


#Función para calcular la tasa y saber cómo de bien se han deducido las clases
def calcularTasa(clases, clases_deducidas):
	#Calulamos el número de instancias bien clasificadas
	bien = sum(clases == clases_deducidas)
	tasa = 100*(bien / len(clases))
	return tasa


# Función para obtener la característica siguiente más prometedora. Le pasamos por argumento la máscara que
# tenemos hasta ese momento y nos devuelve la máscara modificada con un 1 en aquella posición que sea la
# más prometedora.
def caracteristicaMasPrometedora(clases, mascara, conjunto, is_train):
	# Buscamos las posiciones que estén a cero, que son las que podemos cambiar a unos
	pos = np.array(range(0, len(mascara)))
	pos = pos[mascara == 0]
	# Buscamos ahora la posición que dé mejor tasa
	mejor_tasa = 0
	mejor_pos = 0

	for i in pos:
		nueva_mascara = list(mascara)
		nueva_mascara[i] = 1
		clases_deducidas = getClases(conjunto, is_train, nueva_mascara, clase_train)
		nueva_tasa = calcularTasa(clases, clases_deducidas)
		if nueva_tasa > mejor_tasa:
			mejor_tasa = nueva_tasa
			mejor_pos = i

	# Devolvemos la nueva máscara y la mejor tasa, por si no se ha producido ganancia entre la
	# nueva máscara y la que teníamos
	mascara[mejor_pos] = 1
	ret = [mascara, mejor_tasa]
	return ret


# Algoritmo greedy SFS
def algoritmoSFS(clases, conjunto, is_train):
	# El siguiente vector serán las características que debemos tener en cuenta para hacer la selección
	# y que iremos modificando a lo largo del algoritmo greedy
	caracteristicas = np.zeros(len(conjunto[0]), int)
	mejora = True
	tasa_actual = 0
	while(mejora):
		ret = caracteristicaMasPrometedora(clases, caracteristicas, conjunto, is_train)
		nueva_tasa = ret[1]
		print(nueva_tasa)
		if nueva_tasa > tasa_actual:
			caracteristicas = ret[0]
			tasa_actual = nueva_tasa
		else:
			mejora = False

	return caracteristicas

# com = time.time()
# mejores_car = algoritmoSFS(clase_train, train, True)
# print(mejores_car)
# fin = time.time()
# print("El tiempo transcurrido, en segundos y para los datos de entrenamiento, ha sido:", fin-com)
#
# # Vamos ahora a calcular el tiempo y la tasa para los nuevos datos
# com2 = time.time()
# clases_test_deducidas = getClases(test, False, mejores_car, clase_train)
# tasa_test = calcularTasa(clase_test, clases_test_deducidas)
# fin2 = time.time()
# print("La tasa de acierto para el conjunto de test ha sido: ", tasa_test)
# print("El tiempo transcurrido en segundo para dicho conjunto ha sido: ", fin2-com2)


def Flip(mascara, posicion):
	if mascara[posicion] == 0:
		mascara[posicion] = 1
	else:
		mascara[posicion] = 0
	return mascara


def busquedaLocal(clases, conjunto, is_train):
	# Generamos una solución inicial aleatoria de ceros y unos
	caracteristicas = np.random.random_integers(0, 1, len(conjunto[0]))
	mejora = True
	vuelta_completa = True
	tasa_actual = 0
	i = 0
	while(mejora and i < 15000):
		for j in range(0, len(conjunto[0])):
			mascara_actual = Flip(caracteristicas, j)
			clases_deducidas = getClases(conjunto, is_train, mascara_actual, clase_train)
			nueva_tasa = calcularTasa(clases, clases_deducidas)
			if nueva_tasa > tasa_actual:
				tasa_actual = nueva_tasa
				vuelta_completa = False
				caracteristicas = Flip(caracteristicas, j)
				print(nueva_tasa)

		if vuelta_completa:
			mejora = False
		else:
			vuelta_completa = True

		i += 1
	return [caracteristicas, tasa_actual]

com = time.time()
ret = busquedaLocal(clase_train, train, True)
mejores_car = ret[0]
tasa = ret[1]
print(mejores_car)
print(tasa)
fin = time.time()
print("El tiempo transcurrido, en segundos y para los datos de entrenamiento, ha sido:", fin-com)

# Vamos ahora a calcular el tiempo y la tasa para los nuevos datos
com2 = time.time()
clases_test_deducidas = getClases(test, False, mejores_car, clase_train)
tasa_test = calcularTasa(clase_test, clases_test_deducidas)
fin2 = time.time()
print("La tasa de acierto para el conjunto de test ha sido: ", tasa_test)
print("El tiempo transcurrido en segundo para dicho conjunto ha sido: ", fin2-com2)
