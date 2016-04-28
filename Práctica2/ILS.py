import numpy as np
from utils import *
from BL import *
from math import ceil

# Función para mutar una solución de tamaño n en t posiciones aleatorias
def mutar(solucion, n, t):
	mutada = np.copy(solucion)
	pos = np.random.permutation(n)
	for i in range(t):
		Flip(mutada, pos[i])
	return mutada


# Algoritmo ILS
def ILS(clases, conjunto, knn):
	n = len(conjunto[0])
	t = ceil(0.1*n)

	# Generamos una solución aleatoria inicial
	sol_aleatoria = np.random.choice(np.array([True, False]), n)
	# La optimizamos con el método de búsqueda Local
	mejor_sol, mejor_tasa = busquedaLocal(clases, conjunto, sol_aleatoria, knn)

	for i in range(24):
		# Mutamos la mejor solución
		mutada = mutar(mejor_sol, n, t)
		# La optimizamos con el método de búsqueda Local
		sol_actual, tasa_actual = busquedaLocal(clases, conjunto, mutada, knn)

		if(tasa_actual > mejor_tasa):
			mejor_sol = np.copy(sol_actual)
			mejor_tasa = tasa_actual

	return mejor_sol, mejor_tasa
