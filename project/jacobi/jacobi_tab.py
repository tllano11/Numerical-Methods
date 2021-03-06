import sys

sys.path.append("/usr/lib/python3.6/site-packages/")
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
import csv
from jacobi_parallel import JacobiParallel
from jacobi_serial import SerialJacobi
import numpy as np


class JacobiTab:
    def __init__(self):
        self.jacobiParallel = JacobiParallel()
        self.jacobiSerial = SerialJacobi()
        self.niter_entry = None
        self.A_matrix = None
        self.b_vector = None
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

        image = Gtk.Image(stock=Gtk.STOCK_OPEN)
        vector_button = Gtk.Button(" Load b vector", image=image)
        vector_button.connect("clicked", self.load_vector, None)
        main_box.pack_start(vector_button, True, True, 10)

        niter_lbl = Gtk.Label("Number of iterations")
        main_box.pack_start(niter_lbl, True, True, 10)
        self.niter_entry = Gtk.Entry()
        main_box.pack_start(self.niter_entry, True, True, 10)

        error_lbl = Gtk.Label("Tolerance")
        main_box.pack_start(error_lbl, True, True, 10)
        self.error_entry = Gtk.Entry()
        main_box.pack_start(self.error_entry, True, True, 10)

        rel_lbl = Gtk.Label("Relaxation (default = 1)")
        main_box.pack_start(rel_lbl, True, True, 10)
        self.rel_entry = Gtk.Entry()
        main_box.pack_start(self.rel_entry, True, True, 10)

        list_box.add(row)

        row = Gtk.ListBoxRow()
        buttons_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        row.add(buttons_box)

        image = Gtk.Image(stock=Gtk.STOCK_EXECUTE)
        parallel_button = Gtk.Button(" Run Parallel Jacobi", image=image)
        parallel_button.connect("clicked", self.jacobi_parallel, None)
        buttons_box.pack_start(parallel_button, True, True, 10)

        image = Gtk.Image(stock=Gtk.STOCK_EXECUTE)
        serial_button = Gtk.Button(" Run Serial Jacobi", image=image)
        serial_button.connect("clicked", self.jacobi_serial, None)
        buttons_box.pack_start(serial_button, True, True, 10)

        list_box.add(row)

        row = Gtk.ListBoxRow()
        button_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        row.add(button_box)

        image = Gtk.Image(stock=Gtk.STOCK_SAVE_AS)
        save_button = Gtk.Button(" Save As", image=image)
        save_button.connect("clicked", self.save, None)
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
                self.A_matrix = np.array(matrix).astype("float64")

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
                self.b_vector = np.array(vector).astype("float64")

        vector_chooser.destroy()

    def jacobi_parallel(self, widget, data=None):
        niter = self.niter_entry.get_text()
        tol = self.error_entry.get_text()

        if self.A_matrix is None or self.b_vector is None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please load a matrix A and vector b first")
            dialog.run()
            dialog.destroy()
        elif niter == "" or tol == "":
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please enter a number of iterations and tolerance first")
            dialog.run()
            dialog.destroy()
        else:
            niter = int(niter)
            tol = float(tol)
            rel = self.rel_entry.get_text()
            A_matrix = self.A_matrix.astype(dtype=np.float64)
            b_vector = self.b_vector.astype(dtype=np.float64)
            if rel == "":
                self.x_vector, niter, error = self.jacobiParallel.start(A_matrix, b_vector, niter, tol)
            else:
                rel = float(rel)
                self.x_vector, niter, error = self.jacobiParallel.start(A_matrix, b_vector, niter, tol, rel)

            print(self.x_vector)
            if self.x_vector is None and niter is None and error is None:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK,
                                           "Jacobi can't be executed because of a division by zero")
                dialog.run()
                dialog.destroy()
            elif self.x_vector is None:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK, "Jacobi failed to obtain a solution that satisfies " \
                                                               "the given tolerance in the provided number of iterations")
                dialog.run()
                dialog.destroy()
            else:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK,
                                           "Jacobi ended successfully in {} iterations with an error of {}".format(niter,                                                                                                                   error))
                dialog.run()
                dialog.destroy()

    def jacobi_serial(self, widget, data=None):
        niter = self.niter_entry.get_text()
        tol = self.error_entry.get_text()

        if self.A_matrix is None or self.b_vector is None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please load a matrix A and vector b first")
            dialog.run()
            dialog.destroy()
        elif niter == "" or tol == "":
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please enter a number of iterations and tolerance first")
            dialog.run()
            dialog.destroy()
        else:
            niter = int(niter)
            tol = float(tol)
            rel = self.rel_entry.get_text()
            if rel == "":
                self.x_vector, niter, error = self.jacobiSerial.jacobi(self.A_matrix, self.b_vector, niter, tol)
            else:
                rel = float(rel)
                self.x_vector, niter, error = self.jacobiSerial.jacobi(self.A_matrix, self.b_vector.flatten(), niter, tol,
                                                                       rel)
            print(self.x_vector)
            if self.x_vector is None and niter is None and error is None:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK,
                                           "Jacobi can't be executed because of a division by zero")
                dialog.run()
                dialog.destroy()
            elif self.x_vector is None:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK, "Jacobi failed to obtain a solution that satisfies " \
                                                               "the given tolerance in the provided number of iterations ")
                dialog.run()
                dialog.destroy()
            else:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK,
                                           "Jacobi ended successfully in {} iterations with an error of {}".format(niter,
                                                                                                                   error))
                dialog.run()
                dialog.destroy()

    def save(self, widget, data=None):
        if self.x_vector is None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please execute Jacobi first")
            dialog.run()
            dialog.destroy()
        else:
            dialog = Gtk.FileChooserDialog("Please choose a file", None,
                                           Gtk.FileChooserAction.SAVE,
                                           (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                            Gtk.STOCK_SAVE, Gtk.ResponseType.OK))

            Gtk.FileChooser.set_current_name(dialog, "x_vector")
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                filename = Gtk.FileChooser.get_filename(dialog)
                np.savetxt(filename, self.x_vector, delimiter=" ")

            dialog.destroy()
