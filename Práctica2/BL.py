import numpy as np
from utils import *

# Función para generar una secuencia que empieza por un número aleatorio y da una vuelta completa,
# acabando donde empezó.
def generarSecuencia(longitud):
	inicio = np.random.random_integers(0,longitud-1)
	secuencia = np.arange(inicio, longitud)
	np.append(secuencia, np.arange(0, inicio))
	return secuencia


# Algoritmo de Búsqueda Local modificado para partir de una solución inicial dada
def busquedaLocal(clases, conjunto, solucion_inicial, knn):
	caracteristicas = solucion_inicial
	mejora = True
	vuelta_completa = True
	tasa_actual = 0
	i = 0
	while(mejora and i < 15000):
		# Hacemos que el inicio de la vuelta sea aleatorio
		posiciones = generarSecuencia(len(conjunto[0]))
		for j in posiciones:
			caracteristicas = Flip(caracteristicas, j)
			# Contamos que hemos generado una nueva solución
			i += 1
			subconjunto = getSubconjunto(conjunto, caracteristicas)
			nueva_tasa = knn.scoreSolution(subconjunto, clases)
			#nueva_tasa = calcularTasaKNNTrain(subconjunto, clases)
			# Si mejora la tasa nos quedamos con esa característica cambiada
			if nueva_tasa > tasa_actual:
				tasa_actual = nueva_tasa
				vuelta_completa = False
				break
			# Si no mejora, lo dejamos como estaba
			else:
				caracteristicas = Flip(caracteristicas, j)

			# Comprobamos que no hemos pasado de las evaluaciones permitidas también en este bucle
			if (i > 15000):
				break

		# Si ha dado una vuelta completa al vecindario y no ha encontrado mejora, nos quedamos con la solución
		# que teníamos y finaliza el algoritmo
		if vuelta_completa:
			mejora = False
		else:
			vuelta_completa = True

	return caracteristicas, tasa_actual
