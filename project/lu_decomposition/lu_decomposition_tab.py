import sys

sys.path.append("/usr/lib/python3.6/site-packages/")
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
import csv
import numpy as np
from gaussian_lu_decomposition import GuassianLUDecomposition


class LUDecompositionTab:
    def __init__(self):
        self.gaussian_lu_decomposition = GuassianLUDecomposition()
        self.A_matrix = None
        self.b_vector = None
        self.L_matrix = None
        self.U_matrix = None

    def get_tab(self):
        box_outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        list_box = Gtk.ListBox()
        box_outer.pack_start(list_box, True, True, 0)

        row = Gtk.ListBoxRow()

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        row.add(main_box)

        matrix_button = Gtk.Button("Load A matrix")
        matrix_button.connect("clicked", self.load_matrix, None)
        main_box.pack_start(matrix_button, True, True, 10)

        lu_button = Gtk.Button("Run LU Decomposition")
        lu_button.connect("clicked", self.lu_decomposition, None)
        main_box.pack_start(lu_button, True, True, 10)

        vector_button = Gtk.Button("Load b vector")
        vector_button.connect("clicked", self.load_vector, None)
        main_box.pack_start(vector_button, True, True, 10)

        out_lbl = Gtk.Label("Output Filename")
        main_box.pack_start(out_lbl, True, True, 10)
        self.out_entry = Gtk.Entry()
        main_box.pack_start(self.out_entry, True, True, 10)

        list_box.add(row)

        row = Gtk.ListBoxRow()
        buttons_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        row.add(buttons_box)

        solve_button = Gtk.Button("Solve System")
        solve_button.connect("clicked", self.substitution, None)
        buttons_box.pack_start(solve_button, True, True, 10)

        determinant_button = Gtk.Button("Calculate Determinant")
        determinant_button.connect("clicked", self.get_determinant, None)
        buttons_box.pack_start(determinant_button, True, True, 10)
        
        inverse_button = Gtk.Button("Calculate Inverse Matrix")
        inverse_button.connect("clicked", self.get_inverse, None)
        buttons_box.pack_start(inverse_button, True, True, 10)

        list_box.add(row)

        return box_outer

    def load_matrix(self, widget, data=None):
        matrix_chooser = Gtk.FileChooserDialog("Select matrix file", None, Gtk.FileChooserAction.OPEN,
                                               (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                                Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        response = matrix_chooser.run()

        if response == Gtk.ResponseType.OK:
            filename = matrix_chooser.get_filename()

            with open(filename) as matrix_file:
                reader = csv.reader(matrix_file, delimiter=' ')
                matrix = list(reader)
                self.A_matrix = np.array(matrix).astype("float")

        matrix_chooser.destroy()

    def load_vector(self, widget, data=None):
        vector_chooser = Gtk.FileChooserDialog("Select vector file", None, Gtk.FileChooserAction.OPEN,
                                               (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                                Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        response = vector_chooser.run()

        if response == Gtk.ResponseType.OK:
            filename = vector_chooser.get_filename()

            with open(filename) as vector_file:
                reader = csv.reader(vector_file, delimiter=' ')
                vector = list(reader)
                self.b_vector = np.array(vector).astype("float")

        vector_chooser.destroy()

    def lu_decomposition(self, widget, data=None):
        pass

    def substitution(self, widget, data=None):
        pass

    def get_determinant(self, widget, data=None):
        pass

    def get_inverse(self, widget, data=None):
        pass
