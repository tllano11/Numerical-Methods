#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

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
sys.path.append("./lu_decomposition")
sys.path.append("./block_operations")
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf
from sparse_matrix_tab import SparseMatrixTab
from matrix_generator_tab import MatrixGeneratorTab
from jacobi_tab import JacobiTab
from gauss_jordan_tab import GaussJordanTab
from gaussian_elimination_tab import GaussianEliminationTab
from lu_decomposition_tab import LUDecompositionTab
from block_operations_tab import BlockTab


class PyApp(Gtk.Window):
    def __init__(self):
        super(PyApp, self).__init__()

        # Tabs of the window.
        self.sparse_matrix_tab = SparseMatrixTab()
        self.matrix_generator_tab = MatrixGeneratorTab()
        self.jacobi_tab = JacobiTab()
        self.gauss_jordan_tab = GaussJordanTab()
        self.gaussian_elimination_tab = GaussianEliminationTab()
        self.lu_decomposition_tab = LUDecompositionTab()
        self.blocks_tab = BlockTab()

        # Elements of the current window.
        self.set_title("Methods")
        self.set_default_size(500, 400)

        # Creates a notebook to add tabs there.
        notebook = Gtk.Notebook()
        notebook.set_tab_pos(Gtk.PositionType.TOP)

        # About us tab
        hbox = Gtk.Box(spacing=10)
        hbox.set_homogeneous(False)
        vbox_left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_left.set_homogeneous(False)
        vbox_right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox_right.set_homogeneous(False)

        hbox.pack_start(vbox_left, True, True, 0)
        hbox.pack_start(vbox_right, True, True, 0)

        label = Gtk.Label()
        label.set_markup("<big>Numerical Methods\n in GPU</big>")
        vbox_left.pack_start(label, True, True, 0)

        label = Gtk.Label()
        label.set_markup("<b>Johan Sebastián Yepes Rios</b>\n"
                          "<b>Tomás Felipe Llano Ríos</b>\n"
                          "<b>Juan Diego Ocampo García</b>")
        vbox_left.pack_start(label, True, True, 0)

        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale("./photo", width=500, height=500,
                                                 preserve_aspect_ratio=True)
        photo = Gtk.Image.new_from_pixbuf(pixbuf)
        vbox_right.pack_start(photo, True, True, 0)

        notebook.append_page(hbox)
        notebook.set_tab_label_text(hbox, "About us")


        # Adding sparse matrix
        vbox_sparse_matrix = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        valign_sparse_matrix = Gtk.Alignment.new(0.5, 0.25, 0, 0)

        valign_sparse_matrix.add(self.sparse_matrix_tab.get_sparse_tab())
        vbox_sparse_matrix.pack_start(valign_sparse_matrix, True, True, 6)
        notebook.append_page(vbox_sparse_matrix)
        notebook.set_tab_label_text(vbox_sparse_matrix, "Sparse matrix")

        # Adding Matrix generator
        vbox_matrix_generator = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        valign_matrix_generator = Gtk.Alignment.new(0.5, 0.25, 0, 0)

        valign_matrix_generator.add(self.matrix_generator_tab.get_tab())
        vbox_matrix_generator.pack_start(valign_matrix_generator, True, True, 6)
        notebook.append_page(vbox_matrix_generator)
        notebook.set_tab_label_text(vbox_matrix_generator, "Matrix Generator")

        # Adding Jacobi solver
        vbox_jacobi = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        valign_jacobi = Gtk.Alignment.new(0.5, 0.25, 0, 0)

        valign_jacobi.add(self.jacobi_tab.get_tab())
        vbox_jacobi.pack_start(valign_jacobi, True, True, 6)

        notebook.append_page(vbox_jacobi)
        notebook.set_tab_label_text(vbox_jacobi, "Jacobi")

        # Adding Gauss Jordan Tab
        vbox_gauss_jordan = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        valign_gauss_jordan = Gtk.Alignment.new(0.5, 0.25, 0, 0)

        valign_gauss_jordan.add(self.gauss_jordan_tab.get_tab())
        vbox_gauss_jordan.pack_start(valign_gauss_jordan, True, True, 6)

        notebook.append_page(vbox_gauss_jordan)
        notebook.set_tab_label_text(vbox_gauss_jordan, "Gauss Jordan")

        # Adding Gaussian Elimination Tab
        vbox_gaussian_elimination = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        valign_gaussian_elimination = Gtk.Alignment.new(0.5, 0.25, 0, 0)

        valign_gaussian_elimination.add(self.gaussian_elimination_tab.get_tab())
        vbox_gaussian_elimination.pack_start(valign_gaussian_elimination, True, True, 6)

        notebook.append_page(vbox_gaussian_elimination)
        notebook.set_tab_label_text(vbox_gaussian_elimination, "Gaussian Elimination")

        # Adding LU Decomposition Tab
        vbox_lu_decomposition = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        valign_lu_decomposition = Gtk.Alignment.new(0.5, 0.25, 0, 0)

        valign_lu_decomposition.add(self.lu_decomposition_tab.get_tab())
        vbox_lu_decomposition.pack_start(valign_lu_decomposition, True, True, 6)

        notebook.append_page(vbox_lu_decomposition)
        notebook.set_tab_label_text(vbox_lu_decomposition, "LU Decomposition")

        # Operations by blocks
        vbox_blocks = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        valign_blocks = Gtk.Alignment.new(0.5, 0.25, 0, 0)

        valign_blocks.add(self.blocks_tab.get_tab())
        vbox_blocks.pack_start(valign_blocks, True, True, 6)

        notebook.append_page(vbox_blocks)
        notebook.set_tab_label_text(vbox_blocks, "Operations by Blocks")


        self.add(notebook)
        self.connect("destroy", Gtk.main_quit)
        self.show_all()

if __name__ == '__main__':
    PyApp()
    Gtk.main()
