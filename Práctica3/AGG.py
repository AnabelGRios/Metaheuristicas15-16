import numpy as np
from geneticos import *
from utils import *

# Selección para el genético generacional. Devuelve la población seleccionada
# para hacer el cruce (de 30 cromosomas).
def seleccionGeneracional(poblacion):
	dtype = [('cromosoma', str(len(poblacion[0][0]))+'bool'), ('tasa', np.float32)]
	seleccion = np.empty(30, dtype)
	for i in range(30):
		pos = torneo(poblacion)
		seleccion[i] = poblacion[pos]
	return seleccion


# Algoritmo genético generacional
def AGG(clases, conjunto, knn):
	# Generamos la población inicial, ya evaluada, con lo que tenemos que contar
	# 30 evaluaciones
	poblacion = generarPoblacionInicial(30, len(conjunto[0]), conjunto, clases, knn)
	num_evaluaciones = 30
	# Fijamos el número de mutaciones que habrá en cada etapa
	mutaciones = int(np.ceil(0.001*30*len(conjunto[0])))

	while(num_evaluaciones < 15000):
		# Ordenamos la población según la tasa (de menor a mayor)
		poblacion = np.sort(poblacion, order = 'tasa')
		# Seleccionamos la población que vamos a combinar
		seleccion = seleccionGeneracional(poblacion)
		# Decidimos cuántas parejas cruzan
		tope = np.round(0.7*len(conjunto[0])/2)
		i = 0
		while(i < 2*tope):
			hijo1, hijo2 = cruce(seleccion[i], seleccion[i+1])
			# Evaluamos los hijos
			subconjunto = getSubconjunto(conjunto, hijo1)
			tasa = knn.scoreSolution(subconjunto, clases)
			# Vamos guardando la nueva poblacion
			seleccion[i][0] = hijo1
			seleccion[i][1] = tasa
			subconjunto = getSubconjunto(conjunto, hijo2)
			tasa = knn.scoreSolution(subconjunto, clases)
			seleccion[i+1][0] = hijo2
			seleccion[i+1][1] = tasa
			num_evaluaciones = num_evaluaciones + 2
			i = i+2

		# seleccion tiene ahora mismo las primeras N posiciones con los hijos de
		# cruces y las demás como las tenía

		# MUTAMOS
		for k in range(mutaciones):
			crom = np.random.choice(30)
			gen = np.random.choice(len(conjunto[0]))
			Flip(seleccion[crom][0], gen)
			# Calculamos la nueva tasa después de la mutación
			subconjunto = getSubconjunto(conjunto, seleccion[crom][0])
			tasa = knn.scoreSolution(subconjunto, clases)
			seleccion[crom][1] = tasa
			num_evaluaciones += 1

		# Comprobamos si la mejor solución que teníamos sigue, para no perderla
		max_tasa = np.max(seleccion["tasa"])
		if poblacion[29][1] > max_tasa:
			#np.sort(seleccion, order = 'tasa')
			#seleccion[0] = poblacion[29]
			pos = np.argmin(seleccion["tasa"])
			seleccion[pos] = poblacion[29]

		# Actualizamos la población
		poblacion = seleccion

	# Buscamos la mejor solución encontrada y la devolvemos junto con su tasa
	pos = np.argmax(poblacion["tasa"])
	return poblacion[pos][0], poblacion[pos][1]
