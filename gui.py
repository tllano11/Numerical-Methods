#!/usr/bin/env python
#-*- coding: utf-8 -*-

from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
import csv
import numpy as np
from jacobi import jacobi
from cuda_utils import to_gpu

np_matrix = None
np_b_matrix = None

class MyFrame(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.master.title("Example")
        self.master.rowconfigure(5, weight=1)
        self.master.columnconfigure(5, weight=1)
        self.grid(sticky=W+E+N+S)

        self.button = Button(self,
                             text="BROWSE A",
                             command=self.load_file,
                             width=10)
        self.button.grid(row=1, column=0, sticky=W)

        self.button = Button(self,
                             text="BROWSE b",
                             command=self.load_b_matrix,
                             width=10)
        self.button.grid(row=2, column=0, sticky=W)

        self.button = Button(self,
                             text="JACOBI",
                             command=self.start_jacobi,
                             width=10)
        self.button.grid(row=3, column=0, sticky=W)


        self.quit_b = Button(self,
                             text="QUIT",
                             fg="red",
                             command=self.destroy,
                             width=10)
        self.quit_b.grid(row=4, column=0)

    def start_jacobi(self):
        global np_matrix, np_b_matrix
        if np_matrix != None and np_b_matrix != None:
            length = len(np_b_matrix)
            x_current = np.zeros(length, dtype=np.float64)
            x_next = np.zeros(length, dtype=np.float64)
            gpu_x_current = to_gpu(x_current)
            gpu_x_next = to_gpu(x_next)
            gpu_A = to_gpu(np_matrix)
            gpu_b = to_gpu(np_b_matrix)
            jacobi(gpu_A, gpu_b, gpu_x_current, gpu_x_next, length, length)
            x_next = to_cpu(gpu_x_next)
            np.savetxt("result.csv", x_next, fmt="%1.9f", delimiter=",")
        else:
            print("error")

    def load_file(self):
        global np_matrix
        fname = askopenfilename(filetypes=[("Comma Separated Values", "*.csv")])
        if fname:
            with open(fname) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                matrix = list(reader)
                np_matrix = np.array(matrix).astype("float64")
                np_matrix = np_matrix.flatten()
            return

    def load_b_matrix(self):
        global np_b_matrix
        fname = askopenfilename(filetypes=[("Comma Separated Values", "*.csv")])
        if fname:
            with open(fname) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                matrix = list(reader)
                np_b_matrix = np.array(matrix).astype("float64")
                np_b_matrix = np_matrix.flatten()
            return


if __name__ == "__main__":
    MyFrame().mainloop()
