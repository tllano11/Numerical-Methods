#!/usr/bin/env python3.6
#-*- coding: utf-8 -*-

'''
    File name: guiNum.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date created: 13-April-2017
    Date last modified: 20-May-2017
    Python Version: 3.6.0
'''

import sys
sys.path.append("/usr/lib/python3.6/site-packages/")
sys.path.append("./sparseMatrices")
sys.path.append("./matrixGenerator")
sys.path.append("./jacobi")
sys.path.append("./gauss_jordan")
sys.path.append("./gaussian_elimination")
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from sparse_matrix_tab import SparseMatrixTab
from matrix_generator_tab import MatrixGeneratorTab
from jacobi_tab import JacobiTab
from gauss_jordan_tab import GaussJordanTab
from gaussian_elimination_tab import GaussianEliminationTab

class PyApp(Gtk.Window):

  def __init__(self):
    super(PyApp, self).__init__()

    # Tabs of the window.
    self.sparse_matrix_tab = SparseMatrixTab()
    self.matrix_generator_tab = MatrixGeneratorTab()
    self.jacobi_tab = JacobiTab()
    self.gauss_jordan_tab = GaussJordanTab()
    self.gaussian_elimination_tab = GaussianEliminationTab()

    # Elements of the current window.
    self.set_title("Methods")
    self.set_default_size(500, 400)

    # Creates a notebook to add tabs there.
    notebook = Gtk.Notebook()
    notebook.set_tab_pos(Gtk.PositionType.TOP)

    # Adding sparse matrix
    vbox_sparse_matrix = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    valign_sparse_matrix = Gtk.Alignment.new(0.5,0.25, 0, 0)

    valign_sparse_matrix.add(self.sparse_matrix_tab.get_sparse_tab())
    vbox_sparse_matrix.pack_start(valign_sparse_matrix, True, True, 6)
    notebook.append_page(vbox_sparse_matrix)
    notebook.set_tab_label_text(vbox_sparse_matrix, "Sparse matrix")


    # Adding Matrix generator
    vbox_matrix_generator = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    valign_matrix_generator = Gtk.Alignment.new(0.5,0.25, 0, 0)

    valign_matrix_generator.add(self.matrix_generator_tab.get_tab())
    vbox_matrix_generator.pack_start(valign_matrix_generator, True, True, 6)
    notebook.append_page(vbox_matrix_generator)
    notebook.set_tab_label_text(vbox_matrix_generator, "Matrix Generator")

    # Adding Jacobi solver
    vbox_jacobi = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    valign_jacobi = Gtk.Alignment.new(0.5,0.25, 0, 0)

    valign_jacobi.add(self.jacobi_tab.get_tab())
    vbox_jacobi.pack_start(valign_jacobi, True, True, 6)

    notebook.append_page(vbox_jacobi)
    notebook.set_tab_label_text(vbox_jacobi, "Jacobi")

    # Adding Gauss Jordan Tab
    vbox_gauss_jordan = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    valign_gauss_jordan = Gtk.Alignment.new(0.5,0.25, 0, 0)

    valign_gauss_jordan.add(self.gauss_jordan_tab.get_tab())
    vbox_gauss_jordan.pack_start(valign_gauss_jordan, True, True, 6)

    notebook.append_page(vbox_gauss_jordan)
    notebook.set_tab_label_text(vbox_gauss_jordan, "Gauss Jordan")

    # Adding Gaussian Elimination Tab
    vbox_gaussian_elimination = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    valign_gaussian_elimination = Gtk.Alignment.new(0.5,0.25, 0, 0)

    valign_gaussian_elimination.add(self.gaussian_elimination_tab.get_tab())
    vbox_gaussian_elimination.pack_start(valign_gaussian_elimination, True, True, 6)

    notebook.append_page(vbox_gaussian_elimination)
    notebook.set_tab_label_text(vbox_gaussian_elimination, "Gaussian Elimination")








    '''
    hb = Gtk.HButtonBox()

    btn1 = Gtk.RadioButton(None,"Degree")
    hb.add(btn1)

    btn2 = Gtk.RadioButton(btn1,"P.G.")
    hb.add(btn2)

    btn3 = Gtk.RadioButton(btn1,"Doctorate")
    hb.add(btn3)

    notebook.append_page(hb)
    notebook.set_tab_label_text(hb, "Qualification")

    tv = Gtk.TextView()
    notebook.append_page(tv)
    notebook.set_tab_label_text(tv, "about")
    '''
    self.add(notebook)
    self.connect("destroy", Gtk.main_quit)
    self.show_all()

  def create_sparse_matrix(self, widget, data=None):
    filename = self.filename_entry.get_text()
    self.sparseMatrix.create_sparse_matrix(filename)

if __name__ == '__main__':
    PyApp()
    Gtk.main()
