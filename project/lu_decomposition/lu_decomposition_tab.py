import sys

sys.path.append("/usr/lib/python3.6/site-packages/")
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
import csv
import numpy as np
from gaussian_lu_decomposition import GuassianLUDecomposition
from serial_decomposition_LU import SerialLUDecomposition


class LUDecompositionTab:
    def __init__(self):
        self.gaussian_lu_decomposition = GuassianLUDecomposition()
        self.serial_lu_decomposition = SerialLUDecomposition()
        self.A_matrix = None
        self.b_vector = None
        self.L = None
        self.U = None
        self.inverse = None
        self.x_vector = None

    def get_tab(self):
        box_outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        list_box = Gtk.ListBox()
        box_outer.pack_start(list_box, True, True, 0)

        row = Gtk.ListBoxRow()

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        row.add(main_box)

        image = Gtk.Image(stock=Gtk.STOCK_OPEN)
        matrix_button = Gtk.Button(" Load A matrix", image=image)
        matrix_button.connect("clicked", self.load_matrix, None)
        main_box.pack_start(matrix_button, True, True, 10)

        image = Gtk.Image(stock=Gtk.STOCK_EXECUTE)
        lu_button = Gtk.Button(" Run Parallel LU Decomposition", image=image)
        lu_button.connect("clicked", self.lu_decomposition, None)
        main_box.pack_start(lu_button, True, True, 10)

        image = Gtk.Image(stock=Gtk.STOCK_EXECUTE)
        lu_button = Gtk.Button(" Run Serial LU Decomposition", image=image)
        lu_button.connect("clicked", self.serial_lu, None)
        main_box.pack_start(lu_button, True, True, 10)

        image = Gtk.Image(stock=Gtk.STOCK_OPEN)
        vector_button = Gtk.Button(" Load b vector", image=image)
        vector_button.connect("clicked", self.load_vector, None)
        main_box.pack_start(vector_button, True, True, 10)

        list_box.add(row)

        row = Gtk.ListBoxRow()
        buttons_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        row.add(buttons_box)

        image = Gtk.Image(stock=Gtk.STOCK_EXECUTE)
        solve_button = Gtk.Button(" Solve System", image=image)
        solve_button.connect("clicked", self.substitution, None)
        buttons_box.pack_start(solve_button, True, True, 10)

        image = Gtk.Image(stock=Gtk.STOCK_EXECUTE)
        determinant_button = Gtk.Button(" Calculate Determinant", image=image)
        determinant_button.connect("clicked", self.get_determinant, None)
        buttons_box.pack_start(determinant_button, True, True, 10)

        image = Gtk.Image(stock=Gtk.STOCK_EXECUTE)
        inverse_button = Gtk.Button(" Calculate Inverse Matrix", image=image)
        inverse_button.connect("clicked", self.get_inverse, None)
        buttons_box.pack_start(inverse_button, True, True, 10)

        list_box.add(row)

        row = Gtk.ListBoxRow()
        button_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        row.add(button_box)

        image = Gtk.Image(stock=Gtk.STOCK_SAVE_AS)
        save_button = Gtk.Button(" Save LU as", image=image)
        save_button.connect("clicked", self.save_lu, None)
        button_box.pack_start(save_button, True, True, 10)

        image = Gtk.Image(stock=Gtk.STOCK_SAVE_AS)
        save_button = Gtk.Button(" Save Solution as", image=image)
        save_button.connect("clicked", self.save_x, None)
        button_box.pack_start(save_button, True, True, 10)

        image = Gtk.Image(stock=Gtk.STOCK_SAVE_AS)
        save_button = Gtk.Button(" Save Inverse as", image=image)
        save_button.connect("clicked", self.save_inverse, None)
        button_box.pack_start(save_button, True, True, 10)

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
                self.A_matrix = np.array(matrix).astype("float128")

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
                self.b_vector = np.array(vector).astype("float128")

        vector_chooser.destroy()

    def lu_decomposition(self, widget, data=None):
        if self.A_matrix is None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please load a matrix A and vector b first")
            dialog.run()
            dialog.destroy()
        else:
            A_matrix = self.A_matrix.astype(dtype=np.float64)
            self.L, self.U = self.gaussian_lu_decomposition.start(A_matrix.copy())
            if self.L is not None and self.U is not None:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK, "LU decomposition ended successfully")
                dialog.run()
                dialog.destroy()
            else:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK, "LU decomposition failed because of a division by zero")
                dialog.run()
                dialog.destroy()
            print("L=", self.L)
            print("U=", self.U)

    def serial_lu(self, widget, data=None):
        if self.A_matrix is None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please load a matrix A and vector b first")
            dialog.run()
            dialog.destroy()
        else:
            self.L, self.U = self.serial_lu_decomposition.decomposition_LU(self.A_matrix.copy())
            if self.L is not None and self.U is not None:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK, "LU decomposition ended successfully")
                dialog.run()
                dialog.destroy()
            else:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK, "LU decomposition failed because of a division by zero")
                dialog.run()
                dialog.destroy()
            print("L=", self.L)
            print("U=", self.U)

    def substitution(self, widget, data=None):
        if self.L is None or self.U is None or self.b_vector is None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please decompose the matrix A and load a vector b first")
            dialog.run()
            dialog.destroy()
        else:
            self.x_vector = self.gaussian_lu_decomposition.get_solution(self.L, self.U, self.b_vector.flatten())
            print(self.x_vector)
            if self.x_vector is not None:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK, "Solve ended successfully")
                dialog.run()
                dialog.destroy()
            else:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK, "Solve failed")
                dialog.run()
                dialog.destroy()

    def get_determinant(self, widget, data=None):
        if self.L is None or self.U is None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please decompose the matrix A first")
            dialog.run()
            dialog.destroy()
        else:
            determinant = self.gaussian_lu_decomposition.get_determinant(self.L, self.U)
            if determinant is not None:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK, "Determinant is {}".format(determinant))
                dialog.run()
                dialog.destroy()
            else:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK, "Get determinant failed")
                dialog.run()
                dialog.destroy()

    def get_inverse(self, widget, data=None):
        if self.L is None or self.U is None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please decompose the matrix A first")
            dialog.run()
            dialog.destroy()
        else:
            self.inverse = self.gaussian_lu_decomposition.get_inverse(self.L, self.U)
            print(self.inverse)
            if self.inverse is not None:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK, "Inverse matrix was calculated successfully")
                dialog.run()
                dialog.destroy()
            else:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK, "Matrix has not inverse")
                dialog.run()
                dialog.destroy()

    def save_lu(self, widget, data=None):
        if self.L is None or self.U is None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please decompose the matrix A first")
            dialog.run()
            dialog.destroy()
        else:
            dialog = Gtk.FileChooserDialog("Please choose a file", None,
                                           Gtk.FileChooserAction.SAVE,
                                           (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                            Gtk.STOCK_SAVE, Gtk.ResponseType.OK))

            Gtk.FileChooser.set_current_name(dialog, "LU.txt")
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                filename = Gtk.FileChooser.get_filename(dialog)
                with open(filename, "ab") as f:
                    f.write("L = \n".encode())
                    np.savetxt(f, self.L, delimiter=" ")
                    f.write("U = \n".encode())
                    np.savetxt(f, self.L, delimiter=" ")
                    f.close()

            dialog.destroy()

    def save_inverse(self, widget, data=None):
        if self.inverse is None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please get the inverse first")
            dialog.run()
            dialog.destroy()
        else:
            dialog = Gtk.FileChooserDialog("Please choose a file", None,
                                           Gtk.FileChooserAction.SAVE,
                                           (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                            Gtk.STOCK_SAVE, Gtk.ResponseType.OK))

            Gtk.FileChooser.set_current_name(dialog, "inverse_A.txt")
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                filename = Gtk.FileChooser.get_filename(dialog)
                np.savetxt(filename, self.inverse, delimiter=" ")

            dialog.destroy()

    def save_x(self, widget, data=None):
        if self.x_vector is None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please solve the system first")
            dialog.run()
            dialog.destroy()
        else:
            dialog = Gtk.FileChooserDialog("Please choose a file", None,
                                           Gtk.FileChooserAction.SAVE,
                                           (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                            Gtk.STOCK_SAVE, Gtk.ResponseType.OK))

            Gtk.FileChooser.set_current_name(dialog, "x_vector.txt")
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                filename = Gtk.FileChooser.get_filename(dialog)
                np.savetxt(filename, self.x_vector, delimiter=" ")

            dialog.destroy()
