from scipy.io import arff
import numpy as np
import time
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler

base = arff.loadarff('wdbc.arff')

# Pasamos los datos a numpy array
datos = np.array([[base[0][i][j] for j in range(1, len(base[0][0]))] for i in range(0, len(base[0]))], float)
clases = np.array([base[0][i][0] for i in range(0, len(base[0]))])

# Los tipos de clase que hay
tipos_clase = np.array([b'B', b'M'])

# Normalizamos los datos por columnas
scaler = MinMaxScaler()
datos = scaler.fit_transform(datos)

# Tenemos que extraer los datos aleatoriamente pero de forma proporcionada. Separamos los índices de los datos por clases y hacemos
# un aleatorio en cada clase, fijando primero la semilla
np.random.seed(567891234)
posiciones_train = np.array([], int)
posiciones_test = np.array([], int)
# Para cada clase, cogemos las posiciones en las que hay datos de esa clase, le hacemos una permutación aleatoria y nos quedamos con
# la mitad para entrenamiento y la otra mitad para test.
for i in range(0, len(tipos_clase)):
	posiciones = np.array(range(0, len(clases)))
	pos_tipo = posiciones[clases == tipos_clase[i]]
	pos_tipo = np.random.permutation(pos_tipo)
	mitad = len(pos_tipo) // 2
	posiciones_train = np.append(posiciones_train, pos_tipo[0:mitad])
	posiciones_test = np.append(posiciones_test, pos_tipo[mitad:len(pos_tipo)])

# Ahora metemos en cuatro vectores, dos para datos y otros dos para clases, los que son para entrenamiento y los que son para test, según las
# posiciones que acabamos de obtener
datos_train = np.array([datos[i] for i in posiciones_train])
clases_train = np.array([clases[i] for i in posiciones_train])
datos_test = np.array([datos[i] for i in posiciones_test])
clases_test = np.array([clases[i] for i in posiciones_test])


# Función para obtener un subconjunto del conjunto inicial con todas las características, eliminando aquellas que no se vayan a utilizar,
# es decir, aquellas cuya posición en la máscara estén a False.
def getSubconjunto(conjunto, mascara):
	i = len(conjunto[0])
	subconjunto = np.copy(conjunto)
	while(i > 0):
		i -= 1
		if (mascara[i] == False):
			subconjunto = np.delete(subconjunto, i, 1)

	return subconjunto


# Función para calcular la tasa utilizando el 3NN del módulo sklearn de python. Lo entrenamos con el conjunto de datos de entrenamiento
# datos_train y las características que estamos teniendo en cuenta, hacemos el leave one out para cada dato y obtemos la tasa de acierto
# en los demás. Finalmente devolvemos la media de las tasas
def calcularTasaKNNTrain(subconjunto, clases):
	suma_tasas = 0
	for i in range(0, len(subconjunto)):
		dato = np.array([subconjunto[i]])
		clase = np.array([clases[i]])
		sub_prueba = np.delete(subconjunto, i, 0)
		clases_prueba = np.delete(clases, i, 0)
		knn = neighbors.KNeighborsClassifier(3)
		knn.fit(sub_prueba, clases_prueba)
		suma_tasas += 100*knn.score(dato, clase)

	tasa = suma_tasas / len(subconjunto)
	return tasa


# Función para calcular la tasa utilizando el 3NN del módulo sklearn de python. Lo entrenamos con el conjunto de datos de entrenamiento
# datos_train y las características que estamos teniendo en cuenta y después obtenemos la tasa de acierto con el conjunto de test
def calcularTasaKNNTest(subconjunto, clases_test, datos_train, mascara):
	knn = neighbors.KNeighborsClassifier(3)
	sub_train = getSubconjunto(datos_train, mascara)
	knn.fit(sub_train, clases_train)
	tasa = 100*knn.score(subconjunto, clases_test)
	return tasa


# Función para obtener la característica siguiente más prometedora. Le pasamos por argumento la máscara que
# tenemos hasta ese momento y nos devuelve la máscara modificada con un True en aquella posición que sea la
# más prometedora.
def caracteristicaMasPrometedora(clases, mascara, conjunto):
	# Buscamos las posiciones que estén a False, que son las que podemos cambiar a True
	pos = np.array(range(0, len(mascara)))
	pos = pos[mascara == False]
	# Buscamos ahora la posición que dé mejor tasa
	mejor_tasa = 0
	mejor_pos = 0

	for i in pos:
		nueva_mascara = np.copy(mascara)
		nueva_mascara[i] = True
		# Nos quedamos con aquellas columnas que vayamos a utilizar, es decir, aquellas cuya posición en la máscara esté a True
		subconjunto = getSubconjunto(conjunto, nueva_mascara)
		nueva_tasa = calcularTasaKNNTrain(subconjunto, clases)

		if nueva_tasa > mejor_tasa:
			mejor_tasa = nueva_tasa
			mejor_pos = i

	# Devolvemos la nueva máscara y la mejor tasa, por si no se ha producido ganancia entre la
	# nueva máscara y la que teníamos
	nueva_mascara = np.copy(mascara)
	nueva_mascara[mejor_pos] = 1
	ret = [nueva_mascara, mejor_tasa]
	return ret


# Algoritmo greedy SFS
def algoritmoSFS(clases, conjunto):
	# El siguiente vector serán las características que debemos tener en cuenta para hacer la selección
	# y que iremos modificando a lo largo del algoritmo greedy. Al principio no hemos cogido ninguna característica,
	# por lo que tenemos un vector de False.
	caracteristicas = np.repeat(False, len(conjunto[0]))
	mejora = True
	tasa_actual = 0

	while(mejora):
		# Obtenemos la siguiente característica más prometedora en un vector de características donde habrá una nueva puesta a True
		car = caracteristicaMasPrometedora(clases, caracteristicas, conjunto)
		nueva_tasa = car[1]

		# Si con la nueva característica sigue habiendo mejora seguimos, si no lo paramos y nos quedamos con el vector que teníamos.
		if nueva_tasa > tasa_actual:
			print(nueva_tasa)
			caracteristicas = car[0]
			tasa_actual = nueva_tasa
		else:
			mejora = False

	return caracteristicas

# # Calculamos el tiempo y la tasa de acierto para los datos de entrenamiento
# com = time.time()
# mejores_car = algoritmoSFS(clases_train, datos_train)
# print(mejores_car)
# fin = time.time()
# print("El tiempo transcurrido, en segundos y para los datos de entrenamiento, ha sido:", fin-com)
#
# # Vamos ahora a calcular el tiempo y la tasa para los nuevos datos
# com2 = time.time()
# subcjto = getSubconjunto(datos_test, mejores_car)
# tasa_test = calcularTasaKNNTest(subcjto, clases_test, datos_train, mejores_car)
# fin2 = time.time()
# print("La tasa de acierto para el conjunto de test ha sido: ", tasa_test)
# print("El tiempo transcurrido en segundos para dicho conjunto ha sido: ", fin2-com2)

def Flip(mascara, posicion):
	nueva_mascara = np.copy(mascara)
	if nueva_mascara[posicion] == False:
		nueva_mascara[posicion] = True
	else:
		nueva_mascara[posicion] = False
	return nueva_mascara


def generarSecuencia(longitud):
	inicio = np.random.random_integers(0,longitud)
	secuencia = np.arange(inicio, longitud)
	np.append(secuencia, np.arange(0, inicio))
	return secuencia


def busquedaLocal(clases, conjunto):
	# Generamos una solución inicial aleatoria de True y False para la solución inicial
	caracteristicas = np.random.choice(np.array([True, False]), len(conjunto[0]))
	mejora = True
	vuelta_completa = True
	tasa_actual = 0
	i = 0
	while(mejora and i < 15000):
		# Hacemos que el inicio de la vuelta sea aleatorio
		posiciones = generarSecuencia(len(conjunto[0]))
		for j in posiciones:
			mascara_actual = Flip(caracteristicas, j)
			subconjunto = getSubconjunto(conjunto, mascara_actual)
			nueva_tasa = calcularTasaKNNTrain(subconjunto, clases)
			if nueva_tasa > tasa_actual:
				tasa_actual = nueva_tasa
				vuelta_completa = False
				caracteristicas = mascara_actual
				print(nueva_tasa)

		if vuelta_completa:
			mejora = False
		else:
			vuelta_completa = True

		i += 1
	return [caracteristicas, tasa_actual]

com = time.time()
ret = busquedaLocal(clases_train, datos_train)
mejores_car = ret[0]
tasa = ret[1]
print(mejores_car)
print(tasa)
fin = time.time()
print("El tiempo transcurrido, en segundos y para los datos de entrenamiento, ha sido:", fin-com)

# Vamos ahora a calcular el tiempo y la tasa para los nuevos datos
com2 = time.time()
subcjto = getSubconjunto(datos_test, mejores_car)
tasa_test = calcularTasaKNNTest(subcjto, clases_test, datos_train, mejores_car)
fin2 = time.time()
print("La tasa de acierto para el conjunto de test ha sido: ", tasa_test)
print("El tiempo transcurrido en segundo para dicho conjunto ha sido: ", fin2-com2)
