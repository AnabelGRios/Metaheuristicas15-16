from scipy.io import arff
import numpy as np
import argparse
import time
from sklearn.preprocessing import MinMaxScaler
import random

from knnLooGPU import *
from SFS import *
from AGG import *
from AGE import *

parser = argparse.ArgumentParser()
parser.add_argument("semilla", help="semilla que se va a utilizarn en la ejecución", type=int)
parser.add_argument("base", help="base de datos a utilizar. Escribir 1 para WDBC, 2 para movement libras y 3 para arritmia", type=int)
parser.add_argument("algoritmo", help="algoritmo a utilizar. Escribir 1 para SFS, 2 para AGG, 3 para AGE y 4 para KNN", type=int)
args = parser.parse_args()

np.random.seed(args.semilla)
random.seed(args.semilla)
if args.base == 1:
	base = arff.loadarff('wdbc.arff')
	# Los tipos de clase que hay
	tipos_clase = np.array([0, 1], dtype=np.int32)
	# Pasamos los datos a numpy array (las clases están en la primera columna)
	datos = np.array([[base[0][i][j] for j in range(1, len(base[0][0]))] for i in range(0, len(base[0]))], np.float32)
	clases = np.array([base[0][i][0] for i in range(0, len(base[0]))])
	clases[clases == b'B'] = 0
	clases[clases == b'M'] = 1
	clases = np.asarray(clases.tolist(), dtype=np.int32)
elif args.base == 2:
	base = arff.loadarff('movement_libras.arff')
	tipos_clase = np.array(np.arange(1,16), dtype=np.int32)
	# Pasamos los datos a numpy array (las clases están en la última columna)
	datos = np.array([[base[0][i][j] for j in range(0, (len(base[0][0]))-1)] for i in range(0, len(base[0]))], np.float32)
	clases = np.array([base[0][i][(len(base[0][0])-1)] for i in range(0, len(base[0]))])
	clases = np.asarray(clases.tolist(), dtype=np.int32)
else:
	base = arff.loadarff('arrhythmia.arff')
	tipos_clase = np.array([1,2,6,10,16], dtype=np.int32)
	# Pasamos los datos a numpy array (las clases están en la última columna)
	datos = np.array([[base[0][i][j] for j in range(0, (len(base[0][0]))-1)] for i in range(0, len(base[0]))], np.float32)
	clases = np.array([base[0][i][(len(base[0][0])-1)] for i in range(0, len(base[0]))])
	clases = np.asarray(clases.tolist(), dtype=np.int32)


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

# Creamos el knn
knnGPU = knnLooGPU(len(datos_train), len(datos_test), len(datos_train[0]), 3)

if args.algoritmo == 1:
	print("Greedy")
	com = time.time()
	mejores_car, tasa = SFS(clases_train, datos_train, knnGPU)
	fin = time.time()
elif args.algoritmo == 2:
	print("AGG")
	com = time.time()
	mejores_car, tasa = AGG(clases_train, datos_train, knnGPU)
	fin = time.time()
elif args.algoritmo == 3:
	print("AGE")
	com = time.time()
	mejores_car, tasa = AGE(clases_train, datos_train, knnGPU)
	fin = time.time()
else:
	print("KNN")
	com = time.time()
	mejores_car = np.repeat(True, len(datos_train[0]))
	tasa = knnGPU.scoreSolution(datos_train, clases_train)
	fin = time.time()


print("Características seleccionadas")
print(mejores_car)
red = (len(mejores_car)-len(mejores_car[mejores_car==True]))/len(mejores_car)
print("Tasa de reduccion")
print(red)
print("Tasa de acierto para el conjunto de train: ", tasa)
print("El tiempo transcurrido en segundos ha sido: ", fin-com)

# Vamos ahora a calcular la tasa para los nuevos datos
subcjto = getSubconjunto(datos_test, mejores_car)
subcjto_train = getSubconjunto(datos_train, mejores_car)
tasa_test = knnGPU.scoreOut(subcjto_train, subcjto, clases_train, clases_test)
print("La tasa de acierto para el conjunto de test ha sido: ", tasa_test)

print("Le damos la vuelta a la partición y volvemos a ejecutar el algoritmo")

# Volvemos a crear el knn, ahora con los datos cambiados
knnGPU = knnLooGPU(len(datos_test), len(datos_train), len(datos_test[0]), 3)

if args.algoritmo == 1:
	print("Greedy")
	com = time.time()
	mejores_car, tasa = SFS(clases_test, datos_test, knnGPU)
	fin = time.time()
elif args.algoritmo == 2:
	print("AGG")
	com = time.time()
	mejores_car, tasa = AGG(clases_test, datos_test, knnGPU)
	fin = time.time()
elif args.algoritmo == 3:
	print("AGE")
	com = time.time()
	mejores_car, tasa = AGE(clases_test, datos_test, knnGPU)
	fin = time.time()
else:
	print("KNN")
	com = time.time()
	mejores_car = np.repeat(True, len(datos_train[0]))
	tasa = knnGPU.scoreSolution(datos_train, clases_train)
	fin = time.time()


print("Características seleccionadas")
print(mejores_car)
red = (len(mejores_car)-len(mejores_car[mejores_car==True]))/len(mejores_car)
print("Tasa de reduccion")
print(red)
print("Tasa de acierto para el conjunto de train: ", tasa)
print("El tiempo transcurrido en segundos ha sido:", fin-com)

# Vamos ahora a calcular la tasa para los nuevos datos
subcjto = getSubconjunto(datos_train, mejores_car)
subcjto_train = getSubconjunto(datos_test, mejores_car)
tasa_test = knnGPU.scoreOut(subcjto_train, subcjto, clases_test, clases_train)
print("La tasa de acierto para el conjunto de test ha sido: ", tasa_test)
