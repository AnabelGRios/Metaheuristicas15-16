import numpy as np
from geneticos import *
from utils import *

# Función para cambiar los dos peores de la población (ordenada) por dos nuevas
# soluciones que se le pasan por argumento junto con sus tasas de ser éstas mejores
def cambiarSiMejor(tasa_mejor, tasa, hijo_mejor, hijo, poblacion):
	if tasa_mejor > poblacion[1][1]:
		poblacion[1][1] = tasa_mejor
		poblacion[1][0] = hijo_mejor
		if tasa > poblacion[0][1]:
			poblacion[0][1] = tasa
			poblacion[0][0] = hijo
	elif tasa_mejor > poblacion[0][1]:
		poblacion[0][1] = tasa_mejor
		poblacion[0][0] = hijo_mejor

# Algoritmo genético estacionario
def AGE(clases, conjunto, knn):
	# Generamos la población inicial, ya evaluada, con lo que tenemos que contar
	# 30 evaluaciones
	poblacion = generarPoblacionInicial(30, len(conjunto[0]), conjunto, clases, knn)
	num_evaluaciones = 30

	# Fijamos la probabilidad de mutar un gen en cada etapa (teniendo en cuenta
	# que sólo mutamos a los hijos de los que cruzamos, es decir, 2)
	prob_mutacion = 0.001*2*len(conjunto[0])

	while(num_evaluaciones < 15000):
		# Ordenamos la población según la tasa (de menor a mayor)
		poblacion = np.sort(poblacion, order = 'tasa')
		# Seleccionamos los dos padres que vamos a combinar por torneo binario
		posicion_padre1 = torneo(poblacion)
		posicion_padre2 = torneo(poblacion)
		padre1 = poblacion[posicion_padre1]
		padre2 = poblacion[posicion_padre2]

		# Obtenemos los dos hijos y mutamos si procede
		hijo1, hijo2 = cruce(padre1, padre2)
		if np.random.random_sample() > prob_mutacion:
			crom = np.random.choice(2)
			gen = np.random.choice(len(conjunto[0]))
			if crom == 0:
				Flip(hijo1, gen)
			else:
				Flip(hijo2, gen)

		# Evaluamos los hijos
		subconjunto = getSubconjunto(conjunto, hijo1)
		tasa1 = knn.scoreSolution(subconjunto, clases)
		subconjunto = getSubconjunto(conjunto, hijo2)
		tasa2 = knn.scoreSolution(subconjunto, clases)
		num_evaluaciones = num_evaluaciones + 2

		# Comprobamos si los hijos son mejores que los peores de la población
		# y sustituimos en ese caso
		if tasa1 > tasa2:
			cambiarSiMejor(tasa1, tasa2, hijo1, hijo2, poblacion)
		else:
			cambiarSiMejor(tasa2, tasa1, hijo2, hijo1, poblacion)

	# Nos quedamos con el que tenga la mayor tasa de la poblacion
	pos = np.argmax(poblacion["tasa"])
	return poblacion[pos][0], poblacion[pos][1]
