import numpy as np
from geneticos import *
from utils import *
from BL import *

# Selección para el genético generacional. Devuelve la población seleccionada
# para hacer el cruce (de 10 cromosomas).
def seleccionGeneracional(poblacion):
	dtype = [('cromosoma', str(len(poblacion[0][0]))+'bool'), ('tasa', np.float32)]
	seleccion = np.empty(10, dtype)
	for i in range(10):
		pos = torneo(poblacion)
		seleccion[i] = poblacion[pos]
	return seleccion


# Algoritmo genético generacional
def AM3(clases, conjunto, knn):
	# Generamos la población inicial, ya evaluada, con lo que tenemos que contar
	# 10 evaluaciones
	poblacion = generarPoblacionInicial(10, len(conjunto[0]), conjunto, clases, knn)
	num_evaluaciones = 10
	# Fijamos el número de mutaciones que habrá en cada etapa
	mutaciones = int(np.ceil(0.001*10*len(conjunto[0])))

	num_generaciones = 0
	num_BL = int(0.1*10)

	while(num_evaluaciones < 15000):
		# Ordenamos la población según la tasa (de menor a mayor)
		poblacion = np.sort(poblacion, order = 'tasa')
		# Seleccionamos la población que vamos a combinar
		seleccion = seleccionGeneracional(poblacion)
		# Decidimos cuántas parejas cruzan
		tope = np.round(0.7*5)
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
			crom = np.random.choice(10)
			gen = np.random.choice(len(conjunto[0]))
			Flip(seleccion[crom][0], gen)
			# Calculamos la nueva tasa después de la mutación
			subconjunto = getSubconjunto(conjunto, seleccion[crom][0])
			tasa = knn.scoreSolution(subconjunto, clases)
			seleccion[crom][1] = tasa
			num_evaluaciones += 1

		# Comprobamos si la mejor solución que teníamos sigue, para no perderla
		max_tasa = np.max(seleccion["tasa"])
		if poblacion[9][1] > max_tasa:
			pos = np.argmin(seleccion["tasa"])
			seleccion[pos] = poblacion[9]

		# Actualizamos la población
		poblacion = seleccion
		num_generaciones += 1

		# Realizamos búsqueda local si han pasado 10 iteraciones sobre el mejor de la población
		if num_generaciones == 10:
			num_generaciones = 0
			for l in range(num_BL):
				crom = np.argmax(poblacion["tasa"])
				poblacion[crom][0], nueva_tasa, iter_BL = busquedaLocal(clases, conjunto, poblacion[crom][0], knn)
				poblacion[crom][1] = nueva_tasa
				num_evaluaciones += iter_BL

	# Buscamos la mejor solución encontrada y la devolvemos junto con su tasa
	pos = np.argmax(poblacion["tasa"])
	return poblacion[pos][0], poblacion[pos][1]
