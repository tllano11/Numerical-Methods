import sys
sys.path.append("/usr/lib/python3.6/site-packages/")
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
import csv
from jacobi_parallel import JacobiParallel
from jacobi_serial import JacobiSerial
from numpy import array

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

    jacobi_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    row.add(jacobi_box)

    matrix_button = Gtk.Button("Load A matrix")
    matrix_button.connect("clicked", self.load_matrix, None)
    jacobi_box.pack_start(matrix_button, True, True, 10)

    vector_button = Gtk.Button("Load b vector")
    vector_button.connect("clicked", self.load_vector, None)
    jacobi_box.pack_start(vector_button, True, True, 10)

    niter_lbl = Gtk.Label("Number of iterations")
    jacobi_box.pack_start(niter_lbl, True, True, 10)
    self.niter_entry = Gtk.Entry()
    jacobi_box.pack_start(self.niter_entry, True, True, 10)

    out_lbl = Gtk.Label("Output Filename")
    jacobi_box.pack_start(out_lbl, True, True, 10)
    self.out_entry = Gtk.Entry()
    jacobi_box.pack_start(self.out_entry, True, True, 10)

    list_box.add(row)

    row = Gtk.ListBoxRow()
    jacobi_buttons_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
    row.add(jacobi_buttons_box)

    jacobi_parallel_button = Gtk.Button("Run Parallel Jacobi")
    jacobi_parallel_button.connect("clicked", self.jacobiParallel, None)
    jacobi_buttons_box.pack_start(jacobi_parallel_button, True, True, 10)

    jacobi_serial_button = Gtk.Button("Run Serial Jacobi")
    jacobi_serial_button.connect("clicked", self.jacobiSerial, None)
    jacobi_buttons_box.pack_start(jacobi_serial_button, True, True, 10)

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
        self.A_matrix = array(matrix).astype("float")
        self.A_matrix = self.A_matrix.flatten()

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
        self.b_vector = array(vector).astype("float")
        self.b_vector = self.b_vector.flatten()

    vector_chooser.destroy()

  def jacobiParallel(self, widget, data=None):
    niter = int(self.niter_entry.get_text())
    filename = self.out_entry.get_text())
    x_vector = self.jacobi_parallel.start(self.A_matrix, self.b_vector, niter)
    numpy.savetxt(filename, x_vector, delimiter=" ")


  def jacobiSerial(self, widget, data=None):
    niter = int(self.niter_entry.get_text())
    filename = self.out_entry.get_text())
    x_vector = self.jacobi_serial.jacobi(self.A_matrix, self.b_vector, niter)
    numpy.savetxt(filename, x_vector, delimiter=" ")
