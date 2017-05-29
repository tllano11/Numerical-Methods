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
        self.jacobi_parallel = JacobiParallel()
        self.jacobi_serial = SerialJacobi()
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
        parallel_button.connect("clicked", self.jacobiParallel, None)
        buttons_box.pack_start(parallel_button, True, True, 10)

        image = Gtk.Image(stock=Gtk.STOCK_EXECUTE)
        serial_button = Gtk.Button(" Run Serial Jacobi", image=image)
        serial_button.connect("clicked", self.jacobiSerial, None)
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

    def jacobiParallel(self, widget, data=None):
        niter = int(self.niter_entry.get_text())
        tol = float(self.error_entry.get_text())
        rel = self.rel_entry.get_text()

        if rel == "":
          self.x_vector, niter, error = self.jacobi_parallel.start(self.A_matrix, self.b_vector, niter, tol)
        else:
          rel = float(rel)
          self.x_vector, niter, error = self.jacobi_parallel.start(self.A_matrix, self.b_vector, niter, tol, rel)

        if self.x_vector is None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.WARNING,
                                       Gtk.ButtonsType.OK_CANCEL, "Jacobi Failed in {} iterations".format(niter))
            dialog.run()
            dialog.destroy()
        else:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                       Gtk.ButtonsType.OK,
                                       "Jacobi ended successfully in {} iterations with an error of {}".format(niter,
                                                                                                               error))
            dialog.run()
            dialog.destroy()

    def jacobiSerial(self, widget, data=None):
        niter = int(self.niter_entry.get_text())
        tol = float(self.error_entry.get_text())
        rel = self.rel_entry.get_text()

        if rel == "":
            self.x_vector, niter, error = self.jacobi_serial.jacobi(self.A_matrix, self.b_vector, niter, tol)
        else:
            rel = float(rel)
            self.x_vector, niter, error = self.jacobi_serial.jacobi(self.A_matrix, self.b_vector, niter, tol, rel)

        if self.x_vector is None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                       Gtk.ButtonsType.OK, "Jacobi Failed in {} iterations".format(niter))
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
