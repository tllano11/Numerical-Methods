#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""
    File name: gaussian_elimination_tab.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date last modified: 29-May-2017
    Python Version: 3.6.0
"""

import sys

sys.path.append("/usr/lib/python3.6/site-packages/")
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
import csv
import numpy as np
from gaussian_elimination import GaussianElimination
from serial_gaussian_elimination import SerialGaussianElimination


class GaussianEliminationTab:
    def __init__(self):
        self.gaussian_elimination = GaussianElimination()
        self.serial_gaussian_elimination = SerialGaussianElimination()
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

        list_box.add(row)

        row = Gtk.ListBoxRow()
        buttons_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        row.add(buttons_box)

        image = Gtk.Image(stock=Gtk.STOCK_EXECUTE)
        parallel_button = Gtk.Button(" Run Parallel Gaussian Elimination", image=image)
        parallel_button.connect("clicked", self.gaussParallel, None)
        buttons_box.pack_start(parallel_button, True, True, 10)

        image = Gtk.Image(stock=Gtk.STOCK_EXECUTE)
        serial_button = Gtk.Button(" Run Serial Gaussian Elimination", image=image)
        serial_button.connect("clicked", self.gaussSerial, None)
        buttons_box.pack_start(serial_button, True, True, 10)

        list_box.add(row)

        row = Gtk.ListBoxRow()
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        row.add(button_box)

        image = Gtk.Image(stock=Gtk.STOCK_SAVE_AS)
        save_button = Gtk.Button(" Save result as", image=image)
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

    def gaussParallel(self, widget, data=None):
        A_matrix = self.A_matrix.astype(dtype=np.float64)
        b_vector = self.b_vector.astype(dtype=np.float64)
        self.x_vector = self.gaussian_elimination.start(A_matrix.copy(), b_vector.copy())
        print(self.x_vector)
        if self.x_vector is not None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                       Gtk.ButtonsType.OK, "Gaussian Elimination ended successfully")
            dialog.run()
            dialog.destroy()
        else:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                       Gtk.ButtonsType.OK, "Gaussian Elimination failed because of a division by zero")
            dialog.run()
            dialog.destroy()

    def gaussSerial(self, widget, data=None):
        self.x_vector = self.serial_gaussian_elimination.elimination(self.A_matrix.copy(), self.b_vector.copy())
        print(self.x_vector)
        if self.x_vector is not None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                       Gtk.ButtonsType.OK, "Gaussian Elimination ended successfully")
            dialog.run()
            dialog.destroy()
        else:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                       Gtk.ButtonsType.OK, "Gaussian Elimination failed because of a division by zero")
            dialog.run()
            dialog.destroy()

    def save(self, widget, data=None):
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
