import numpy as np
import random
from utils import *

# Función para hacer el operador de cruce entre dos cromosomas (entre dos
# soluciones distintas). Devuelve los dos hijos.
def cruce(sol1, sol2):
	tam = len(sol1[0])
	hijo1 = np.empty(tam, bool)
	hijo2 = np.empty(tam, bool)
	for i in range(tam):
		if sol1[0][i] == sol2[0][i]:
			hijo1[i] = sol1[0][i]
			hijo2[i] = sol1[0][i]
		else:
			if np.random.random_sample() < 0.5:
				hijo1[i] = sol1[0][i]
				hijo2[i] = sol2[0][i]
			else:
				hijo1[i] = sol2[0][i]
				hijo2[i] = sol1[0][i]
	return hijo1, hijo2

# Función para evaluar la población inicial. El resto de individuos se irán
# evaluando según se vayan generando.
# poblacion será un numpy array de dos dimensiones, que vamos a convertir en
# un array estructurado en el que guardaremos cada cromosoma junto con su
# tasa de acierto.
def evaluarPoblacionInicial(conjunto, clases, poblacion, knn):
	tasas = np.empty(len(poblacion), np.float32)
	for i in range(len(poblacion)):
		# Nos quedamos con aquellas columnas que vayamos a utilizar.
		subconjunto = getSubconjunto(conjunto, poblacion[i])
		tasas[i] = knn.scoreSolution(subconjunto, clases)
	# Juntamos cada cromosoma con su tasa
	dtype = [('cromosoma', str(len(poblacion[0]))+'bool'), ('tasa', np.float32)]
	pobEval = np.array([i for i in zip(poblacion, tasas)], dtype)
	return pobEval

# Función para generar la población inicial en algoritmos genéticos, consistente
# en 30 soluciones aleatorias que devolveremos en un numpy array estructurado,
# de forma que contenga todos los cromosomas de la población junto con su tasa
# de acierto.
def generarPoblacionInicial(numPob, numCar, conjunto, clases, knn):
	poblacion = np.empty(numPob, np.object)
	# Generamos numPob cromosomas de forma aleatoria
	for i in range(numPob):
		poblacion[i] = np.random.choice(np.array([True, False]), numCar)

	# Evaluamos la población inicial
	pobEval = evaluarPoblacionInicial(conjunto, clases, poblacion, knn)
	return pobEval

# Función que implementa el operador de selección: el torneo binario. Elegimos
# dos individuos aleatorios de la población y devolvemos el mejor de ellos.
# Recibe por parámetros la población completa y devuelve la posición en el
# array en el que se encuentra el mejor de los dos aleatorios.
def torneo(poblacion):
	# Elegimos dos cromosomas aleatorios
	crom1 = np.random.choice(len(poblacion))
	crom2 = np.random.choice(len(poblacion))

	if poblacion[crom1][1] > poblacion[crom2][1]:
		elegido = crom1
	else:
		elegido = crom2

	return elegido
