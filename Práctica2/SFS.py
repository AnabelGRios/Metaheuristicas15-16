import numpy as np
from knn import *

# Función para obtener la característica siguiente más prometedora. Le pasamos por argumento la máscara que
# tenemos hasta ese momento y nos devuelve la máscara modificada con un True en aquella posición que sea la
# más prometedora.
def caracteristicaMasPrometedora(clases, mascara, conjunto):
	# Buscamos las posiciones que estén a False, que son las que podemos cambiar a True
	pos = np.array(range(0, len(mascara)))
	pos = pos[mascara == False]
	# Buscamos ahora la posición que dé mejor tasa
	mejor_tasa = 0
	mejor_pos = 0

	for i in pos:
		mascara[i] = True
		# Nos quedamos con aquellas columnas que vayamos a utilizar, es decir, aquellas cuya posición en la máscara esté a True
		subconjunto = getSubconjunto(conjunto, mascara)
		nueva_tasa = calcularTasaKNNTrain(subconjunto, clases)
		# Volvemos a dejar la máscara como estaba
		mascara[i] = False

		if nueva_tasa > mejor_tasa:
			mejor_tasa = nueva_tasa
			mejor_pos = i

	# Devolvemos la mejor tasa y la posición, por si no se ha producido ganancia entre la
	# nueva máscara y la que teníamos
	ret = [mejor_tasa, mejor_pos]
	return ret


# Algoritmo greedy SFS
def algoritmoSFS(clases, conjunto):
	# El siguiente vector serán las características que debemos tener en cuenta para hacer la selección
	# y que iremos modificando a lo largo del algoritmo greedy. Al principio no hemos cogido ninguna característica,
	# por lo que tenemos un vector de False.
	caracteristicas = np.repeat(False, len(conjunto[0]))
	mejora = True
	tasa_actual = 0

	while(mejora):
		# Obtenemos la siguiente característica más prometedora en un vector de características donde habrá una nueva puesta a True
		car = caracteristicaMasPrometedora(clases, caracteristicas, conjunto)
		nueva_tasa = car[0]
		mejor_pos = car[1]

		# Si con la nueva característica sigue habiendo mejora seguimos, si no lo paramos y nos quedamos con el vector que teníamos.
		if nueva_tasa > tasa_actual:
			caracteristicas[mejor_pos] = True
			tasa_actual = nueva_tasa
		else:
			mejora = False

	return caracteristicas, tasa_actual
