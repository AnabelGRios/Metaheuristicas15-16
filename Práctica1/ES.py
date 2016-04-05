import numpy as np
from knn import *

# Función para cambiar una posición de la máscara que se pasa por argumento
def Flip(mascara, posicion):
	mascara[posicion] = not mascara[posicion]
	return mascara

# Algoritmo de Enfriamiento Simulado
def enfriamientoSimulado(clases, conjunto):
	# Temperatura final y parámetros
	Tf = 0.001
	fi = 0.3
	mu = 0.3
	# Generamos una solución inicial aleatoria de True y False
	caracteristicas = np.random.choice(np.array([True, False]), len(conjunto[0]))
	# Calculamos la tasa de la solución inicial
	sub = getSubconjunto(conjunto, caracteristicas)
	tasa_inicial = calcularTasaKNNTrain(sub, clases)
	# Inicilizamos la mejor solución a la solución inicial junto con su tasa
	mejor_solucion = np.copy(caracteristicas)
	mejor_tasa = tasa_inicial
	tasa_actual = tasa_inicial

	# Con todo esto calculamos la temperatura inicial
	Tini = (mu*tasa_inicial)/(-np.log(fi))
	Tactual = Tini
	Tk = Tini
	# Definimos el número máximo de vecinos y de éxitos
	max_vecinos = 10*len(conjunto[0])
	max_exitos = 0.1*max_vecinos
	M = 15000/max_vecinos
	# Esquema de Cauchy modificando
	beta = (Tini - Tf)/(M*Tini*Tf)

	num_evaluaciones = 0 	# Número de evaluaciones que se llevan hechas
	num_vecinos = 0			# Número de vecinos generado
	exitos_actual = 0		# Número de exitos en el enfriamiento actual
	no_exitos = False		# Controlamos si ha habido o no exitos en el enfriamiento actual

	while(not no_exitos and Tk > Tf and num_evaluaciones < 15000):
		while(num_vecinos < max_vecinos and exitos_actual < max_exitos):
			# Generamos una nueva solución
			pos = np.random.random_integers(len(conjunto[0])-1)
			caracteristicas = Flip(caracteristicas, pos)
			num_vecinos += 1	# Aumentamos el número de vecinos generados
			sub = getSubconjunto(conjunto, caracteristicas)
			nueva_tasa = calcularTasaKNNTrain(sub, clases)
			num_evaluaciones += 1	# Aumentamos el número de evaluaciones hechas
			delta = nueva_tasa - tasa_actual
			if delta != 0 and (delta > 0 or np.random.uniform() <= np.exp(delta/Tk)):
				tasa_actual = nueva_tasa
				exitos_actual += 1		# Aumentamos el número de vecinos que nos quedamos
				if (tasa_actual > mejor_tasa):
					mejor_solucion = np.copy(caracteristicas)
					mejor_tasa = tasa_actual
			# Si no nos quedamos con la solución, volvemos a poner caracteristicas como estaba
			else:
				caracteristicas = Flip(caracteristicas, pos)

			# Comprobamos también en este bucle que no hemos pasado el número de evaluaciones
			if (num_evaluaciones > 15000):
				break

		# Si no nos hemos quedado con ninguna solución, paramos el algoritmo.
		if exitos_actual == 0:
			no_exitos = True
		else:
			exitos_actual = 0

		# Actualizamos la temperatura
		Tk = Tk/(1+beta*Tk)

		# Volvemos a poner num_vecinos a 0
		num_vecinos = 0

	return [mejor_solucion, mejor_tasa]
