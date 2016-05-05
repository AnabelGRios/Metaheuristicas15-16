import numpy as np
from geneticos import *

# Selección para el genético generacional. Devuelve la población seleccionada
# para hacer el cruce
def seleccionGeneracional(poblacion):
	dtype = [('cromosoma', str(len(poblacion[0]))+'bool'), ('tasa', np.float32)]
	seleccion = np.empty(30, dtype)
	for i in range(30):
		pos = torneo(poblacion)
		seleccion[i] = poblacion[pos]
	return seleccion

# Función para comprobar si la mejor solución que tenemos hasta el momento sigue
# en la poblacionstaIncluida
def incluida(poblacion, mejor_sol):
	for i in range(30):
		sol = mejor_sol == poblacion[i][0]
		if (sol[sol] == len(polacion[0])):
			return True

	return False

# Algoritmo genético generacional
def AGG(clases, conjunto, knn):
	# Generamos la población inicial, ya evaluada, con lo que tenemos que contar
	# 30 evaluaciones
	poblacion = generarPoblacionInicial(30, len(conjunto[0], conjunto, knn))
	num_evaluaciones = 30

	while(num_evaluaciones < 15000):
		# Ordenamos la población según la tasa (de menor a mayor)
		poblacion = np.sort(poblacion, order = 'tasa')
		# Seleccionamos la población que vamos a combinar
		seleccion = seleccionGeneracional(poblacion)
		# Decidimos cuántas parejas cruzan
		tope = 0.7*np.ceil(len(conjunto[0])/2)
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
		mutaciones = np.ceil(0.001*30*len(conjunto[0]))
		for k in range(mutaciones):
			crom = np.random.choice(30)
			gen = np.random.choice(len(conjunto[0]))
			Flip(seleccion[crom][0], gen)

		# Comprobamos si la mejor solución continúa
		estaIncluida = incluida(seleccion, poblacion[29][0])
		if (!estaIncluida):
			np.sort(seleccion, order = 'tasa')
			seleccion[0] = poblacion[29]
			#pos = np.argmin(seleccion["tasa"])
			#seleccion[pos] = poblacion[29]

		# Actualizamos la población
		poblacion = seleccion

	# Volvemos a ordenar para quedarnos con la mejor solución encontrada y la
	# devolvemos junto con su tasa
	poblacion = np.sort(poblacion, order = 'tasa')
	return poblacion[29][0], poblacion[29][1]
