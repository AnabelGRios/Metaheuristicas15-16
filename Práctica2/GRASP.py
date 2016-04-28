import numpy as np
from utils import *

# Función para obtener la siguiente característica en el algoritmo greedy aleatorizado en función de un umbral
def siguienteCaracteristica(clases, mascara, conjunto, alpha, knn):
	# Buscamos las posiciones que estén a False, que son las que podemos cambiar a True
	pos = np.array(range(0, len(mascara)))
	pos = pos[mascara == False]
	# Construimos un vector con las tasas que da añadir cada característica, en el orden que nos marca
	# el vector pos.
	tasas = np.empty(len(pos))

	for i in pos:
		mascara[i] = True
		# Nos quedamos con aquellas columnas que vayamos a utilizar, es decir, aquellas cuya posición en la máscara esté a True
		subconjunto = getSubconjunto(conjunto, mascara)
		nueva_tasa = knn.scoreSolution(subconjunto, clases)
		tasas[i] = nueva_tasa
		# Volvemos a dejar la máscara como estaba
		mascara[i] = False

	# Ahora  calculamos el umbral en el que vamos a elegir la características
	maximo = np.max(tasas)
	umbral = maximo-alpha*(maximo-np.min(tasas))
	mejores = [[pos[i], tasas[i]] for i in range(pos) if tasas[i] > umbral]
	elegida = np.random.randint(0,len(mejores))
	nueva_tasa = mejores[elegida][1]
	posicion = mejores[elegida][0]

	# Devolvemos la mejor tasa y la posición, por si no se ha producido ganancia entre la
	# nueva máscara y la que teníamos
	return nueva_tasa, posicion


# Algoritmo greedy SFS aleatorizado
def GRASP(clases, conjunto, knn):
	# El siguiente vector serán las características que debemos tener en cuenta para hacer la selección
	# y que iremos modificando a lo largo del algoritmo greedy. Al principio no hemos cogido ninguna característica,
	# por lo que tenemos un vector de False.
	caracteristicas = np.repeat(False, len(conjunto[0]))
	mejora = True
	tasa_actual = 0

	while(mejora):
		# Obtenemos la siguiente característica más prometedora en un vector de características donde habrá una nueva puesta a True
		nueva_tasa, mejor_pos = siguienteCaracteristica(clases, caracteristicas, conjunto, 0.3, knn)

		# Si con la nueva característica sigue habiendo mejora seguimos, si no lo paramos y nos quedamos con el vector que teníamos.
		if nueva_tasa > tasa_actual:
			caracteristicas[mejor_pos] = True
			tasa_actual = nueva_tasa
		else:
			mejora = False

	return caracteristicas, tasa_actual
