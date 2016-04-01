import numpy as np
from knn import *

# Función para cambiar una posición de la máscara que se pasa por argumento
def Flip(mascara, posicion):
	mascara[posicion] = not mascara[posicion]
	return mascara

# Algoritmo Búsqueda Tabú
def busquedaTabu(clases, conjunto):
	# Generamos una solución inicial aleatoria de True y False
	caracteristicas = np.random.choice(np.array([True, False]), len(conjunto[0]))
	# Calculamos la tasa de la solución inicial
	sub = getSubconjunto(conjunto, caracteristicas)
	tasa_actual = calcularTasaKNNTrain(sub, clases)

	# Inicilizamos la mejor solución
	mejor_solucion = np.copy(caracteristicas)
	mejor_tasa = tasa_actual

	# Creamos la lista tabú
	tam = len(conjunto[0])//3
	lista_tabu = np.repeat(-1, tam)
	# Índice que nos dirá cuál ha sido la última posición que hemos metido
	plista = -1

	num_evaluaciones = 0

	while(num_evaluaciones < 15000):
		tasa_actual = 0
		mejor_pos = -1
		# Generamos los vecinos de forma aleatoria
		pos = np.random.choice(len(conjunto[0]), 30, replace=False)
		# Buscamos el mejor vecino
		for j in pos:
			caracteristicas = Flip(caracteristicas, j)
			sub = getSubconjunto(conjunto, caracteristicas)
			nueva_tasa = calcularTasaKNNTrain(sub, clases)
			num_evaluaciones += 1
			if np.in1d(j, lista_tabu)[0]:
				if nueva_tasa > mejor_tasa and nueva_tasa > tasa_actual:
					print("mejor tasa que la mejor solucion")
					print(nueva_tasa)
					tasa_actual = nueva_tasa
					mejor_pos = j
			elif nueva_tasa > tasa_actual:
				tasa_actual = nueva_tasa
				mejor_pos = j
			caracteristicas = Flip(caracteristicas, j)

		# Nos quedamos con el mejor vecino
		print("mejor vecino")
		print(tasa_actual)
		caracteristicas = Flip(caracteristicas, mejor_pos)
		plista = (plista+1)%tam
		lista_tabu[plista] = mejor_pos
		# Comprobamos si el mejor vecino es mejor que la mejor solución hasta el momento
		if (tasa_actual > mejor_tasa):
			mejor_tasa = tasa_actual
			mejor_solucion = np.copy(caracteristicas)

	return [mejor_solucion, mejor_tasa]
