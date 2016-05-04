import numpy as np

# Función para cambiar una posición de la máscara que se pasa por argumento
def Flip(mascara, posicion):
	mascara[posicion] = not mascara[posicion]
	return mascara

# Función para obtener un subconjunto del conjunto inicial con todas las características, eliminando aquellas que no se vayan a utilizar,
# es decir, aquellas cuya posición en la máscara estén a False.
def getSubconjunto(conjunto, mascara):
	posiciones = np.arange(0,len(mascara))
	posiciones = posiciones[mascara]
	subconjunto = np.empty([len(conjunto), len(posiciones)], dtype=np.float32)
	i = 0
	for j in posiciones:
		subconjunto[:,i] = conjunto[:,j]
		i += 1

	return subconjunto
