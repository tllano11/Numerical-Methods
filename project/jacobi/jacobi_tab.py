import sys
sys.path.append("/usr/lib/python3.6/site-packages/")
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
import csv
from jacobi_parallel import JacobiParallel
from jacobi_serial import JacobiSerial
import numpy as np

class JacobiTab():
  def __init__(self):
    self.jacobi_parallel = JacobiParallel()
    self.jacobi_serial = JacobiSerial()
    self.niter_entry = None
    self.A_matrix = None
    self.b_vector = None

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

    vector_button = Gtk.Button("Load b vector")
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

    out_lbl = Gtk.Label("Output Filename")
    main_box.pack_start(out_lbl, True, True, 10)
    self.out_entry = Gtk.Entry()
    main_box.pack_start(self.out_entry, True, True, 10)

    list_box.add(row)

    row = Gtk.ListBoxRow()
    buttons_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    row.add(buttons_box)

    parallel_button = Gtk.Button("Run Parallel Jacobi")
    parallel_button.connect("clicked", self.jacobiParallel, None)
    buttons_box.pack_start(parallel_button, True, True, 10)

    serial_button = Gtk.Button("Run Serial Jacobi")
    serial_button.connect("clicked", self.jacobiSerial, None)
    buttons_box.pack_start(serial_button, True, True, 10)

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
    vector_chooser = Gtk.FileChooserDialog("Select vector file", None,Gtk.FileChooserAction.OPEN,
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
    rel = float(self.rel_entry.get_text())
    filename = self.out_entry.get_text()
    x_vector = self.jacobi_parallel.start(self.A_matrix, self.b_vector, niter, tol, rel)
    if x_vector is not None and filename != "":
      np.savetxt(filename, x_vector, delimiter=" ")


  def jacobiSerial(self, widget, data=None):
    niter = int(self.niter_entry.get_text())
    tol = float(self.error_entry.get_text())
    filename = self.out_entry.get_text()
    x_vector = self.jacobi_serial.jacobi(self.A_matrix, self.b_vector, niter, tol)
    if x_vector is not None and filename != "":
      np.savetxt(filename, x_vector, delimiter=" ")
