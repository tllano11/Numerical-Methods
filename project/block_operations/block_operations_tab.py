# -*- coding: utf-8 -*-
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

        size_lbl = Gtk.Label("Matrix size (N x N)")
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
        niter = int(self.niter_entry.get_text())
        size = int(self.size_entry.get_text())
        rows_to_read = int(self.rows_entry.get_text())
        tol = float(self.tol_entry.get_text())
        self.x_vector, error, niter = start(self.A_matrix, self.b_vector, rows_to_read, size, niter, tol)
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
