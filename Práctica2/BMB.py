import numpy as np
#from knnLooGPU import *
#from knn import *
from BL import *

# Algoritmo BMB
def busquedaMultiBasica(clases, conjunto):
	mejor_tasa = 0

	for i in range(25):
		# Generamos una solución aleatoria inicial
		sol_aleatoria = np.random.choice(np.array([True, False]), len(conjunto[0]))
		# La optimizamos con el método de búsqueda Local
		sol_actual, tasa_actual = busquedaLocal(clases, conjunto, sol_aleatoria)

		if(tasa_actual > mejor_tasa):
			mejor_sol = np.copy(sol_actual)
			mejor_tasa = tasa_actual

	return mejor_sol, mejor_tasa
