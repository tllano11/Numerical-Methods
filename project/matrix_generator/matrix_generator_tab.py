import sys

sys.path.append("/usr/lib/python3.6/site-packages/")
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import numpy as np
from matrix_generator import MatrixGenerator


class MatrixGeneratorTab:
    def __init__(self):
        self.matrix_filename_entry = None
        self.vector_filename_entry = None
        self.length_entry = None
        self.selected_generator = 1

    def get_tab(self):
        gen_matrix_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        matrix_length_lbl = Gtk.Label("Matrix and vector length")
        gen_matrix_box.pack_start(matrix_length_lbl, True, True, 10)
        self.length_entry = Gtk.Entry()
        gen_matrix_box.pack_start(self.length_entry, True, True, 10)

        button1 = Gtk.RadioButton.new_with_label_from_widget(None, "Diagonally dominant")
        button1.connect("toggled", self.set_generator, "1")
        gen_matrix_box.pack_start(button1, True, True, 10)

        button2 = Gtk.RadioButton.new_with_label_from_widget(button1, "Symmetric")
        button2.connect("toggled", self.set_generator, "2")
        gen_matrix_box.pack_start(button2, True, True, 10)

        button3 = Gtk.RadioButton.new_with_label_from_widget(button1, "Band (default band is 2)")
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

        button10 = Gtk.RadioButton.new_with_label_from_widget(button1, "Random")
        button10.connect("toggled", self.set_generator, "10")
        gen_matrix_box.pack_start(button10, True, True, 10)

        band_lbl = Gtk.Label("Horizontal Band")
        gen_matrix_box.pack_start(band_lbl, True, True, 10)
        self.hband_entry = Gtk.Entry()
        self.hband_entry.set_editable(False)
        gen_matrix_box.pack_start(self.hband_entry, True, True, 10)

        band_lbl = Gtk.Label("Vertical Band")
        gen_matrix_box.pack_start(band_lbl, True, True, 10)
        self.vband_entry = Gtk.Entry()
        self.vband_entry.set_editable(False)
        gen_matrix_box.pack_start(self.vband_entry, True, True, 10)

        image = Gtk.Image(stock=Gtk.STOCK_SAVE_AS)
        gen_button = Gtk.Button(" Save matrices as", image=image)
        gen_button.connect("clicked", self.gen_matrix, None)
        gen_matrix_box.pack_start(gen_button, True, True, 10)

        return gen_matrix_box

    def set_generator(self, button, name):
        if self.selected_generator != int(name):
            self.selected_generator = int(name)

        if self.selected_generator == 3:
            self.vband_entry.set_editable(True)
            self.hband_entry.set_editable(True)
        else:
            self.vband_entry.set_editable(False)
            self.hband_entry.set_editable(False)

    def gen_matrix(self, widget, data=None):
        length = self.length_entry.get_text()

        if length == "":
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please enter a length first")
            dialog.run()
            dialog.destroy()
        else:
            dialog = Gtk.FileChooserDialog("Please choose a file", None,
                        Gtk.FileChooserAction.SAVE,
                        (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                         Gtk.STOCK_SAVE, Gtk.ResponseType.OK))

            Gtk.FileChooser.set_current_name(dialog, "matrix")
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                length = int(length)

                if self.selected_generator == 1:
                    matrix_A, vector_x, vector_b = MatrixGenerator.gen_dominant(length)
                elif self.selected_generator == 2:
                    matrix_A, vector_x, vector_b = MatrixGenerator.gen_symmetric_matrix(length)
                elif self.selected_generator == 3:
                    k1 = self.vband_entry.get_text()
                    k2 = self.hband_entry.get_text()
                    if k1 == "" or k2 == "":
                        matrix_A, vector_x, vector_b = MatrixGenerator.gen_band_matrix(length)
                    else:
                        k1 = int(k1)
                        k2 = int(k2)
                        matrix_A, vector_x, vector_b = MatrixGenerator.gen_band_matrix(length, k1, k2)
                elif self.selected_generator == 4:
                    matrix_A, vector_x, vector_b = MatrixGenerator.gen_identity_matrix(length)
                elif self.selected_generator == 5:
                    matrix_A, vector_x, vector_b = MatrixGenerator.gen_diagonal_matrix(length)
                elif self.selected_generator == 6:
                    matrix_A, vector_x, vector_b = MatrixGenerator.gen_scalar_matrix(length)
                elif self.selected_generator == 7:
                    matrix_A, vector_x, vector_b = MatrixGenerator.gen_antisymmetric_matrix(length)
                elif self.selected_generator == 8:
                    matrix_A, vector_x, vector_b = MatrixGenerator.gen_lower_matrix(length)
                elif self.selected_generator == 9:
                    matrix_A, vector_x, vector_b = MatrixGenerator.gen_upper_matrix(length)
                elif self.selected_generator == 10:
                    matrix_A, vector_x, vector_b = MatrixGenerator.gen_random_matrix(length)

                filename = Gtk.FileChooser.get_filename(dialog)
                # Save file with numpy
                np.savetxt(filename+"_A", matrix_A, fmt="%1.9f", delimiter=" ")
                np.savetxt(filename+"_x", vector_x, fmt="%1.9f", delimiter=" ")
                np.savetxt(filename+"_b", vector_b, fmt="%1.9f", delimiter=" ")

            dialog.destroy()

            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                       Gtk.ButtonsType.OK, "Matrix Generated Successfully")
            dialog.run()
            dialog.destroy()
