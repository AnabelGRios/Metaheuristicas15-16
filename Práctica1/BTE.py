import numpy as np
from knn import *

# Función para cambiar una posición de la máscara que se pasa por argumento
def Flip(mascara, posicion):
	mascara[posicion] = not mascara[posicion]
	return mascara

# Algoritmo Búsqueda Tabú Extendida
def busquedaTabuExtendida(clases, conjunto):
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

	# Creamos un vector de frecuencias para almacenar el número de veces que se ha incluido la característica i
	frec = np.repeat(0, len(conjunto[0]))
	# Creamos un contador para saber el número de soluciones aceptadas
	num_soluciones_aceptadas = 0
	# Contador para saber el número de veces que no mejoramos la solución global
	no_mejora_global = 0

	num_evaluaciones = 0
	mejora = True
	num_no_mejora = 0;
	tasa_mejora = 0;

	while(mejora and num_evaluaciones < 15000):
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
					#print("mejor tasa que la mejor solucion")
					#print(nueva_tasa)
					tasa_actual = nueva_tasa
					mejor_pos = j
			elif nueva_tasa > tasa_actual:
				tasa_actual = nueva_tasa
				mejor_pos = j
			caracteristicas = Flip(caracteristicas, j)

		# Nos quedamos con el mejor vecino
		#print("mejor vecino")
		#print(tasa_actual)
		caracteristicas = Flip(caracteristicas, mejor_pos)
		# Aumentamos la frecuencia de esa característica y el número de soluciones que llevamos
		if caracteristicas[mejor_pos] == True:
			frec[mejor_pos] += 1
		num_soluciones_aceptadas += 1

		# Actualizamos la lista tabú
		plista = (plista+1)%tam
		lista_tabu[plista] = mejor_pos

		# Comprobamos si el mejor vecino es mejor que la mejor solución hasta el momento
		if (tasa_actual > mejor_tasa):
			print("mejor tasa que la de la mejor solución")
			mejor_tasa = tasa_actual
			mejor_solucion = np.copy(caracteristicas)
			no_mejora_global = 0
		else:
			no_mejora_global += 1

		if no_mejora_global == 10:
			print("Cambiando longitud de la lista tabú y solución")
			no_mejora_global = 0
			prob = np.random.choice([0,1,2], p = [0.25, 0.25, 0.5])

			if prob == 0:
				# Generamos una solución aleatoria
				caracteristicas = np.random.choice(np.array([True, False]), len(conjunto[0]))
			elif prob == 1:
				# Partimos de la mejor solución obtenida
				caracteristicas = np.copy(mejor_solucion)
			else:
				# Generamos una solución aleatoria con la memoria a largo plazo
				u = np.random.uniform()
				for i in range(len(caracteristicas)):
					if u < 1 - (frec[i]/num_soluciones_aceptadas):
						caracteristicas[i] = True
					else:
						caracteristicas[i] = False

			# Calculamos el coste de la nueva solución
			sub = getSubconjunto(conjunto, caracteristicas)
			tasa_actual = calcularTasaKNNTrain(sub, clases)

			# Cambiamos el tamaño de la lista tabú
			u = np.random.uniform()
			if u < 0.5:
				tam = tam + tam//2
			else:
				tam = tam - tam//2

			# Inicilizamos de nuevo la lista tabú
			lista_tabu = np.repeat(-1, tam)
			plista = -1


		if (abs(tasa_mejora - mejor_tasa) < 3):
			num_no_mejora += 1
			if (num_no_mejora > 30):
				mejora = False
				print("Salgo por no haber mejora")
		else:
			tasa_mejora = mejor_tasa
			num_no_mejora = 0

	return [mejor_solucion, mejor_tasa]
