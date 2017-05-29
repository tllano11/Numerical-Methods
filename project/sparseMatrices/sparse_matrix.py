#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File name: sparse_matrix.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date last modified: 29-May-2017
    Python Version: 3.6.0
"""

import numpy as np
import random
import json
import sys
from pprint import pprint


class SparseMatrix():
  def create_sparse_matrix(self, filename, matrix_length, density):

    pos = 0
    aux_pos = 0
    matrix = []
    pointerB = []
    pointerE = []
    columns = []
    values = []

    pointerB.append(pos)
    for i in range(0, matrix_length):
      row = []
      if (pos != aux_pos):
        pointerB.append(pos)
      aux_pos = pos
      for j in range(0, matrix_length):
        probability = random.random()
        if probability < density:
          pos += 1
          val = random.randint(1, 10)
          values.append(val)
          columns.append(j)
        else:
          val = 0
        row.append(val)
        if (pos != aux_pos):
          pointerE.append(pos)
        matrix.append(row)
    data = {"values": values, "columns": columns, "pointerB": pointerB, "pointerE": pointerE}
    data_json = json.dumps(data)
    file = open(filename, 'w')
    file.write(data_json)
    file.close()
    print("Matrix")
    print(matrix)

  def load_sparse_matrix(self, filename):
    with open(filename) as data_file:
      data = json.load(data_file)
    values = data["values"]
    print("JSON")
    print(values)


def main(argv):
  if len(argv) != 2:
    print("Unsage: ./program filename")
    sys.exit()
  sparseMatrix = SparseMatrix()
  sparseMatrix.create_sparse_matrix(argv[1])
  sparseMatrix.load_sparse_matrix(argv[1])


if __name__ == '__main__':
  main(sys.argv)
