#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""
    File name: block_operations_tab.py
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
from read_rows import start


class BlockTab:
    def __init__(self):
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

        size_lbl = Gtk.Label("Matrix length N")
        main_box.pack_start(size_lbl, True, True, 10)
        self.size_entry = Gtk.Entry()
        main_box.pack_start(self.size_entry, True, True, 10)

        rows_lbl = Gtk.Label("Number of rows to read by block")
        main_box.pack_start(rows_lbl, True, True, 10)
        self.rows_entry = Gtk.Entry()
        main_box.pack_start(self.rows_entry, True, True, 10)

        tol_lbl = Gtk.Label("Tolerance")
        main_box.pack_start(tol_lbl, True, True, 10)
        self.tol_entry = Gtk.Entry()
        main_box.pack_start(self.tol_entry, True, True, 10)

        list_box.add(row)

        row = Gtk.ListBoxRow()
        buttons_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        row.add(buttons_box)

        image = Gtk.Image(stock=Gtk.STOCK_EXECUTE)
        parallel_button = Gtk.Button(" Run Parallel Jacobi", image=image)
        parallel_button.connect("clicked", self.jacobi_by_blocks, None)
        buttons_box.pack_start(parallel_button, True, True, 10)

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
            self.A_matrix = matrix_chooser.get_filename()

        matrix_chooser.destroy()

    def load_vector(self, widget, data=None):
        vector_chooser = Gtk.FileChooserDialog("Select vector file", None, Gtk.FileChooserAction.OPEN,
                                               (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                                Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        response = vector_chooser.run()

        if response == Gtk.ResponseType.OK:
            self.b_vector = vector_chooser.get_filename()

        vector_chooser.destroy()

    def jacobi_by_blocks(self, widget, data=None):
        niter = self.niter_entry.get_text()
        size = self.size_entry.get_text()
        rows_to_read = self.rows_entry.get_text()
        tol = self.tol_entry.get_text()

        if self.A_matrix is None or self.b_vector:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please load a matrix A and vector b first")
            dialog.run()
            dialog.destroy()
        elif niter == "" or size == "" or rows_to_read == "" or tol == "":
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please enter a number of iterations, length, number of rows and tolerance first")
            dialog.run()
            dialog.destroy()
        else:
            niter = int(niter)
            size = int(niter)
            rows_to_read = int(rows_to_read)
            tol = float(tol)
            self.x_vector, error, count = start(self.A_matrix, self.b_vector, rows_to_read, size, niter, tol)
            print(self.x_vector)
            if self.x_vector is None and error is None and count is None:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK,
                                           "Jacobi can't be executed because of a division by zero")
                dialog.run()
                dialog.destroy()
            elif self.x_vector is None:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK, "Jacobi Failed")
                dialog.run()
                dialog.destroy()
            else:
                dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
                                           Gtk.ButtonsType.OK,
                                           "Jacobi ended successfully in {} iterations with an error of {}".format(count,
                                                                                                                   error))
                dialog.run()
                dialog.destroy()

    def save(self, widget, data=None):
        if self.x_vector is None:
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "Please execute jacobi first")
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
