#!/usr/bin/env python3.6
#-*- coding: utf-8 -*-

"""
    File name: substitution.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date created: 03-May-2017
    Date last modified: 03-May-2017
    Python Version: 3.6.0
"""


def forward_substitution(A, b, n):
  x = []
  for i in range(0, n):
    if A[i][i] != 0:
      accum = 0
      for j in range(0, i):
        accum += A[i][j]*x[j]
      x.append((b[i] - accum)/A[i][i])
  return x


def back_substitution(A, b, n):
  x = [0] * n
  for i in range(n-1, -1, -1):
    if A[i][i] != 0:
      accum = 0
      for j in range(i, n):
        accum += A[i][j]*x[j]
      x[i] = ((b[i] - accum)/A[i][i])
  return x


def main():
  A = [[3, 0, 0, 0], [-1, 1, 0, 0], [3, -2, -1, 0], [1, -2, 6, 2]]
  b = [5, 6, 4, 2]
  print("forward subs: ", forward_substitution(A, b, 4))

  A = [[4, -1, 2, 3], [0, -2, 7, -4], [0, 0, 6, 5], [0, 0, 0, 3]]
  b = [20, -7, 4, 6]
  print("back subs: ", back_substitution(A, b, 4))


if __name__ == "__main__":
  main()
