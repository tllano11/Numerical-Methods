import numpy as np
import random

matrix_length = 10
density = 0.1

def create_format_matrix():
	val = 0 
	pos = 0 
	aux_pos = 0
	pointerB = []
	pointerE = []
	columns = []
	values = []

	pointerB.append(pos)
	for i in range(0, matrix_length):
		if(pos != aux_pos):
			pointerB.append(pos)
		aux_pos = pos
		for j in range(0, matrix_length):
			probability = random.random()
			if probability < density:
				pos += 1
				val = random.randint(1,10)
				values.append(val)
				columns.append(j)
		pointerE.append(pos)

	print("values")
	print(values)
	print("columns")
	print(columns)
	print("pointerB")
	print(pointerB)
	print("pointerE")
	print(pointerE)

if __name__ == '__main__':
	create_format_matrix()