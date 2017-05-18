import sys

sys.path.append("/usr/lib/python3.6/site-packages/")
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import numpy as np

from matrix_generator import MatrixGenerator


class MatrixGeneratorTab():
    def __init__(self):
        self.matrix_filename_entry = None
        self.vector_filename_entry = None
        self.length_entry = None
        self.selected_generator = None

    def get_tab(self):
        gen_matrix_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        matrix_length_lbl = Gtk.Label("Matrix and vector length")
        gen_matrix_box.pack_start(matrix_length_lbl, True, True, 10)
        self.length_entry = Gtk.Entry()
        gen_matrix_box.pack_start(self.length_entry, True, True, 10)

        matrix_filename_lbl = Gtk.Label("Matrix filename")
        gen_matrix_box.pack_start(matrix_filename_lbl, True, True, 10)
        self.matrix_filename_entry = Gtk.Entry()
        gen_matrix_box.pack_start(self.matrix_filename_entry, True, True, 10)

        vector_filename_lbl = Gtk.Label("Vector filename")
        gen_matrix_box.pack_start(vector_filename_lbl, True, True, 10)
        self.vector_filename_entry = Gtk.Entry()
        gen_matrix_box.pack_start(self.vector_filename_entry, True, True, 10)

        button1 = Gtk.RadioButton.new_with_label_from_widget(None, "Diagonally dominant")
        button1.connect("toggled", self.set_generator, "1")
        gen_matrix_box.pack_start(button1, True, True, 10)

        button2 = Gtk.RadioButton.new_with_label_from_widget(button1, "Symmetric")
        button2.connect("toggled", self.set_generator, "2")
        gen_matrix_box.pack_start(button2, True, True, 10)

        button3 = Gtk.RadioButton.new_with_label_from_widget(button1, "Band")
        button3.connect("toggled", self.set_generator, "3")
        gen_matrix_box.pack_start(button3, True, True, 10)

        button4 = Gtk.RadioButton.new_with_label_from_widget(button1, "Identity")
        button4.connect("toggled", self.set_generator, "4")
        gen_matrix_box.pack_start(button4, True, True, 10)

        button5 = Gtk.RadioButton.new_with_label_from_widget(button1, "Diagonal")
        button5.connect("toggled", self.set_generator, "5")
        gen_matrix_box.pack_start(button5, True, True, 10)

        button6 = Gtk.RadioButton.new_with_label_from_widget(button1, "Scalar")
        button6.connect("toggled", self.set_generator, "6")
        gen_matrix_box.pack_start(button6, True, True, 10)

        button7 = Gtk.RadioButton.new_with_label_from_widget(button1, "Antisymmetric")
        button7.connect("toggled", self.set_generator, "7")
        gen_matrix_box.pack_start(button7, True, True, 10)

        button8 = Gtk.RadioButton.new_with_label_from_widget(button1, "Lower")
        button8.connect("toggled", self.set_generator, "8")
        gen_matrix_box.pack_start(button8, True, True, 10)

        button9 = Gtk.RadioButton.new_with_label_from_widget(button1, "Upper")
        button9.connect("toggled", self.set_generator, "9")
        gen_matrix_box.pack_start(button9, True, True, 10)

        gen_button = Gtk.Button("Generate matrix and vector")
        gen_button.connect("clicked", self.gen_matrix, None)
        gen_matrix_box.pack_start(gen_button, True, True, 10)

        return gen_matrix_box

    def set_generator(self, button, name):
        if self.selected_generator != int(name):
            self.selected_generator = int(name)

    def gen_matrix(self, widget, data=None):
        matrix_filename = self.matrix_filename_entry.get_text()
        vector_filename = self.vector_filename_entry.get_text()
        length = int(self.length_entry.get_text())

        if self.selected_generator == 1:
            matrix = MatrixGenerator.gen_dominant(length)
        elif self.selected_generator == 2:
            matrix = MatrixGenerator.gen_symmetric_matrix(length)
        elif self.selected_generator == 3:
            matrix = MatrixGenerator.gen_band_matrix(length)
        elif self.selected_generator == 4:
            matrix = MatrixGenerator.gen_identity_matrix(length)
        elif self.selected_generator == 5:
            matrix = MatrixGenerator.gen_diagonal_matrix(length)
        elif self.selected_generator == 6:
            matrix = MatrixGenerator.gen_scalar_matrix(length)
        elif self.selected_generator == 7:
            matrix = MatrixGenerator.gen_antisymmetric_matrix(length)
        elif self.selected_generator == 8:
            matrix = MatrixGenerator.gen_lower_matrix(length)
        elif self.selected_generator == 9:
            matrix = MatrixGenerator.gen_upper_matrix(length)

        # Save file with numpy
        np.savetxt(matrix_filename, matrix, fmt="%1.9f", delimiter=" ")

        b = MatrixGenerator.gen_vector(length)

        # Save file with numpy
        np.savetxt(vector_filename, b, fmt="%1.9f", delimiter=" ")
