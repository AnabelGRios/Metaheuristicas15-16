from scipy.io import arff
import numpy as np
import argparse
import time
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn import cross_validation

parser = argparse.ArgumentParser()
parser.add_argument("semilla", help="semilla que se va a utilizarn en la ejecución", type=int)
parser.add_argument("base", help="base de datos a utilizar. Escribir 1 para WDBC, 2 para movement libras y 3 para arritmia", type=int)
args = parser.parse_args()

np.random.seed(args.semilla)
if args.base == 1:
	base = arff.loadarff('wdbc.arff')
	# Los tipos de clase que hay
	tipos_clase = np.array([b'B', b'M'])
	# Pasamos los datos a numpy array (las clases están en la primera columna)
	datos = np.array([[base[0][i][j] for j in range(1, len(base[0][0]))] for i in range(0, len(base[0]))], float)
	clases = np.array([base[0][i][0] for i in range(0, len(base[0]))])
elif args.base == 2:
	base = arff.loadarff('movement_libras.arff')
	tipos_clase = np.array(np.arange(1,16),'|S5')
	# Pasamos los datos a numpy array (las clases están en la última columna)
	datos = np.array([[base[0][i][j] for j in range(0, (len(base[0][0]))-1)] for i in range(0, len(base[0]))], float)
	clases = np.array([base[0][i][(len(base[0][0])-1)] for i in range(0, len(base[0]))])
else:
	base = arff.loadarff('arrhythmia.arff')
	tipos_clase = np.array([1,2,6,10,16], '|S5')
	# Pasamos los datos a numpy array (las clases están en la última columna)
	datos = np.array([[base[0][i][j] for j in range(0, (len(base[0][0]))-1)] for i in range(0, len(base[0]))], float)
	clases = np.array([base[0][i][(len(base[0][0])-1)] for i in range(0, len(base[0]))])


# Normalizamos los datos por columnas
scaler = MinMaxScaler()
datos = scaler.fit_transform(datos)

# Tenemos que extraer los datos aleatoriamente pero de forma proporcionada. Separamos los índices de los datos por clases y hacemos
# un aleatorio en cada clase, una vez hemos fijado la semilla
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
	posiciones = np.arange(0,len(mascara))
	posiciones = posiciones[mascara]
	subconjunto = np.empty([len(conjunto), len(posiciones)])
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
	# for i in range(0, len(subconjunto)):
	# 	dato = np.array([subconjunto[i]])
	# 	clase = np.array([clases[i]])
	# 	sub_prueba = np.delete(subconjunto, i, 0)
	# 	clases_prueba = np.delete(clases, i, 0)
	# 	knn = neighbors.KNeighborsClassifier(n_neighbors = 3)
	# 	knn.fit(sub_prueba, clases_prueba)
	# 	suma_tasas += 100*knn.score(dato, clase)

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
		mascara[i] = True
		# Nos quedamos con aquellas columnas que vayamos a utilizar, es decir, aquellas cuya posición en la máscara esté a True
		subconjunto = getSubconjunto(conjunto, mascara)
		nueva_tasa = calcularTasaKNNTrain(subconjunto, clases)
		# Volvemos a dejar la máscara como estaba
		mascara[i] = False

		if nueva_tasa > mejor_tasa:
			mejor_tasa = nueva_tasa
			mejor_pos = i

	# Devolvemos la mejor tasa y la posición, por si no se ha producido ganancia entre la
	# nueva máscara y la que teníamos
	ret = [mejor_tasa, mejor_pos]
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
		nueva_tasa = car[0]
		mejor_pos = car[1]

		# Si con la nueva característica sigue habiendo mejora seguimos, si no lo paramos y nos quedamos con el vector que teníamos.
		if nueva_tasa > tasa_actual:
			print(nueva_tasa)
			caracteristicas[mejor_pos] = True
			tasa_actual = nueva_tasa
		else:
			mejora = False

	return caracteristicas

# print("GREEDY")
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
# tasa_test = calcularTasaKNNTest(subcjto, clases_test, datos_train, clases_train, mejores_car)
# fin2 = time.time()
# print("La tasa de acierto para el conjunto de test ha sido: ", tasa_test)
# print("El tiempo transcurrido en segundos para dicho conjunto ha sido: ", fin2-com2)
#
# print("Le damos la vuelta a la partición y volvemos a ejecutar el algoritmo")
# clases_train2 = clases_test
# clases_test2 = clases_train
# datos_train2 = datos_test
# datos_test2 = datos_train
#
# # Calculamos el tiempo y la tasa de acierto para los datos de entrenamiento
# com = time.time()
# mejores_car = algoritmoSFS(clases_train2, datos_train2)
# print(mejores_car)
# fin = time.time()
# print("El tiempo transcurrido, en segundos y para los datos de entrenamiento, ha sido:", fin-com)
#
# # Vamos ahora a calcular el tiempo y la tasa para los nuevos datos
# com2 = time.time()
# subcjto = getSubconjunto(datos_test2, mejores_car)
# tasa_test = calcularTasaKNNTest(subcjto, clases_test2, datos_train2, clases_train2, mejores_car)
# fin2 = time.time()
# print("La tasa de acierto para el conjunto de test ha sido: ", tasa_test)
# print("El tiempo transcurrido en segundos para dicho conjunto ha sido: ", fin2-com2)


# Función para cambiar una posición de la máscara que se pasa por argumento
def Flip(mascara, posicion):
	mascara[posicion] = not mascara[posicion]
	return mascara


# Función para generar una secuencia que empieza por un número aleatorio y da una vuelta completa,
# acabando donde empezó.
def generarSecuencia(longitud):
	inicio = np.random.random_integers(0,longitud-1)
	secuencia = np.arange(inicio, longitud)
	np.append(secuencia, np.arange(0, inicio))
	return secuencia


# Algoritmo de Búsqueda Local
def busquedaLocal(clases, conjunto):
	# Generamos una solución inicial aleatoria de True y False
	caracteristicas = np.random.choice(np.array([True, False]), len(conjunto[0]))
	mejora = True
	vuelta_completa = True
	tasa_actual = 0
	i = 0
	while(mejora and i < 15000):
		# Hacemos que el inicio de la vuelta sea aleatorio
		posiciones = generarSecuencia(len(conjunto[0]))
		for j in posiciones:
			caracteristicas = Flip(caracteristicas, j)
			# Contamos que hemos generado una nueva solución
			i += 1
			subconjunto = getSubconjunto(conjunto, caracteristicas)
			nueva_tasa = calcularTasaKNNTrain(subconjunto, clases)
			# Si mejora la tasa nos quedamos con esa característica cambiada
			if nueva_tasa > tasa_actual:
				tasa_actual = nueva_tasa
				vuelta_completa = False
				print(nueva_tasa)
				break
			# Si no mejora, lo dejamos como estaba
			else:
				caracteristicas = Flip(caracteristicas, j)

			# Comprobamos que no hemos pasado de las evaluaciones permitidas también en este bucle
			if (i > 15000):
				break

		# Si ha dado una vuelta completa al vecindario y no ha encontrado mejora, nos quedamos con la solución
		# que teníamos y finaliza el algoritmo
		if vuelta_completa:
			mejora = False
		else:
			vuelta_completa = True

	return [caracteristicas, tasa_actual]

# print("Búsqueda Local")
# com = time.time()
# ret = busquedaLocal(clases_train, datos_train)
# mejores_car = ret[0]
# tasa = ret[1]
# print(mejores_car)
# print(tasa)
# fin = time.time()
# print("El tiempo transcurrido, en segundos y para los datos de entrenamiento, ha sido:", fin-com)
#
# # Vamos ahora a calcular el tiempo y la tasa para los nuevos datos
# com2 = time.time()
# subcjto = getSubconjunto(datos_test, mejores_car)
# tasa_test = calcularTasaKNNTest(subcjto, clases_test, datos_train, clases_train, mejores_car)
# fin2 = time.time()
# print("La tasa de acierto para el conjunto de test ha sido: ", tasa_test)
# print("El tiempo transcurrido en segundo para dicho conjunto ha sido: ", fin2-com2)
#
# print("Le damos la vuelta a la partición y volvemos a ejecutar el algoritmo")
# clases_train2 = clases_test
# clases_test2 = clases_train
# datos_train2 = datos_test
# datos_test2 = datos_train
#
# com = time.time()
# ret = busquedaLocal(clases_train2, datos_train2)
# mejores_car = ret[0]
# tasa = ret[1]
# print(mejores_car)
# print(tasa)
# fin = time.time()
# print("El tiempo transcurrido, en segundos y para los datos de entrenamiento, ha sido:", fin-com)
#
# # Vamos ahora a calcular el tiempo y la tasa para los nuevos datos
# com2 = time.time()
# subcjto = getSubconjunto(datos_test2, mejores_car)
# tasa_test = calcularTasaKNNTest(subcjto, clases_test2, datos_train2, clases_train2, mejores_car)
# fin2 = time.time()
# print("La tasa de acierto para el conjunto de test ha sido: ", tasa_test)
# print("El tiempo transcurrido en segundo para dicho conjunto ha sido: ", fin2-com2)


# Algoritmo de Enfriamiento Simulado
def enfriamientoSimulado(clases, conjunto):
	# Temperatura final y parámetros
	Tf = 0.001
	fi = 0.3
	mu = 0.3
	# Generamos una solución inicial aleatoria de True y False
	caracteristicas = np.random.choice(np.array([True, False]), len(conjunto[0]))
	# Calculamos la tasa de la solución inicial
	sub = getSubconjunto(conjunto, caracteristicas)
	tasa_inicial = calcularTasaKNNTrain(sub, clases)
	# Inicilizamos la mejor solución a la solución inicial junto con su tasa
	mejor_solucion = np.copy(caracteristicas)
	mejor_tasa = tasa_inicial
	tasa_actual = tasa_inicial

	# Con todo esto calculamos la temperatura inicial
	Tini = (mu*tasa_inicial)/(-np.log(fi))
	Tactual = Tini
	Tk = Tini
	# Definimos el número máximo de vecinos y de éxitos
	max_vecinos = 10*len(conjunto[0])
	max_exitos = 0.1*max_vecinos
	M = 15000/max_vecinos
	# Esquema de Cauchy modificando
	beta = (Tini - Tf)/(M*Tini*Tf)

	num_evaluaciones = 0 	# Número de evaluaciones que se llevan hechas
	num_vecinos = 0			# Número de vecinos generado
	exitos_actual = 0		# Número de exitos en el enfriamiento actual
	no_exitos = False		# Controlamos si ha habido o no exitos en el enfriamiento actual

	while(num_evaluaciones < 15000 and not no_exitos):
		while(num_vecinos < max_vecinos and exitos_actual < max_exitos):
			# Generamos una nueva solución
			pos = np.random.random_integers(len(conjunto[0])-1)
			caracteristicas = Flip(caracteristicas, pos)
			num_vecinos += 1	# Aumentamos el número de vecinos generados
			sub = getSubconjunto(conjunto, caracteristicas)
			nueva_tasa = calcularTasaKNNTrain(sub, clases)
			num_evaluaciones += 1	# Aumentamos el número de evaluaciones hechas
			print(num_evaluaciones)
			if nueva_tasa > tasa_actual or np.random.uniform() <= np.exp((nueva_tasa-tasa_actual)/Tk):
				print(nueva_tasa)
				tasa_actual = nueva_tasa
				exitos_actual += 1		# Aumentamos el número de vecinos que nos quedamos
				if (tasa_actual > mejor_tasa):
					mejor_solucion = np.copy(caracteristicas)
					mejor_tasa = tasa_actual
			# Si no nos quedamos con la solución, volvemos a poner caracteristicas como estaba
			else:
				caracteristicas = Flip(caracteristicas, pos)

			# Comprobamos también en este bucle que no hemos pasado el número de evaluaciones
			if (num_evaluaciones > 15000):
				break

		print(exitos_actual)
		# Si no nos hemos quedado con ninguna solución, paramos el algoritmo.
		if exitos_actual == 0:
			no_exitos = True
		else:
			exitos_actual = 0

		# Actualizamos la temperatura
		Tactual = Tk
		Tk = Tactual/(1+beta*Tactual)
		print(Tk)

		# Volvemos a poner num_vecinos a 0
		num_vecinos = 0

	return [mejor_solucion, mejor_tasa]

print("Enfriamiento Simulado")
com = time.time()
ret = enfriamientoSimulado(clases_train, datos_train)
mejores_car = ret[0]
tasa = ret[1]
print(mejores_car)
print(tasa)
fin = time.time()
print("El tiempo transcurrido, en segundos y para los datos de entrenamiento, ha sido:", fin-com)

# Vamos ahora a calcular el tiempo y la tasa para los nuevos datos
com2 = time.time()
subcjto = getSubconjunto(datos_test, mejores_car)
tasa_test = calcularTasaKNNTest(subcjto, clases_test, datos_train, clases_train, mejores_car)
fin2 = time.time()
print("La tasa de acierto para el conjunto de test ha sido: ", tasa_test)
print("El tiempo transcurrido en segundo para dicho conjunto ha sido: ", fin2-com2)
