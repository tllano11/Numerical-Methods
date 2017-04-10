#!/usr/bin/env python
#-*- coding: utf-8 -*-

from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
import csv
import numpy as np

np_matrix = None

class MyFrame(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.master.title("Example")
        self.master.rowconfigure(5, weight=1)
        self.master.columnconfigure(5, weight=1)
        self.grid(sticky=W+E+N+S)

        self.button = Button(self,
                             text="BROWSE",
                             command=self.load_file,
                             width=10)
        self.button.grid(row=1, column=0, sticky=W)

        self.button = Button(self,
                             text="JACOBI",
                             command=self.start_jacobi,
                             width=10)
        self.button.grid(row=2, column=0, sticky=W)


        self.quit_b = Button(self,
                             text="QUIT",
                             fg="red",
                             command=self.destroy,
                             width=10)
        self.quit_b.grid(row=3, column=0)

    def start_jacobi(self):
        global np_matrix
        print(np_matrix)

    def load_file(self):
        global np_matrix
        fname = askopenfilename(filetypes=[("Comma Separated Values", "*.csv")])
        if fname:
            with open(fname) as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                matrix = list(reader)
                np_matrix = np.array(matrix).astype("float")
                np_matrix = np_matrix.flatten()
            return


if __name__ == "__main__":
    MyFrame().mainloop()
