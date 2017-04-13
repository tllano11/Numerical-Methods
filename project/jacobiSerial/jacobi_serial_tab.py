import gtk
import csv
from jacobi_serial import JacobiSerial
from numpy import array

class JacobiSerialTab():
	def __init__(self):
		self.jacobi_serial = JacobiSerial()
		self.niter_entry = None
		self.A_matrix = None
		self.b_vector = None

	def get_tab(self):
		jacobi_serial_box = gtk.VBox()

		matrix_button = gtk.Button("Load A matrix")
		matrix_button.connect("clicked", self.load_matrix, None)
		jacobi_serial_box.pack_start(matrix_button, True, True, 10)

		vector_button = gtk.Button("Load b vector")
		vector_button.connect("clicked", self.load_vector, None)
		jacobi_serial_box.pack_start(vector_button, True, True, 10)

		niter_lbl = gtk.Label("Number of iterations")
		jacobi_serial_box.pack_start(niter_lbl, True, True, 10)
		self.niter_entry = gtk.Entry()
		jacobi_serial_box.pack_start(self.niter_entry, True, True, 10)

		jacobi_button = gtk.Button("Jacobi")
		jacobi_button.connect("clicked", self.jacobi, None)
		jacobi_serial_box.pack_start(jacobi_button, True, True, 10)
		return jacobi_serial_box

	def load_matrix(self, widget, data=None):
		matrix_chooser = gtk.FileChooserDialog("Select matrix file", None, gtk.FILE_CHOOSER_ACTION_OPEN,
		(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OPEN, gtk.RESPONSE_OK))
		response = matrix_chooser.run()
		filename = matrix_chooser.get_filename()

		with open(filename) as matrix_file:
			reader = csv.reader(matrix_file, delimiter=' ')
			matrix = list(reader)
			self.A_matrix = array(matrix).astype("float")
		matrix_chooser.destroy()

	def load_vector(self, widget, data=None):
		vector_chooser = gtk.FileChooserDialog("Select vector file", None, gtk.FILE_CHOOSER_ACTION_OPEN,
		(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OPEN, gtk.RESPONSE_OK))
		response = vector_chooser.run()
		filename = vector_chooser.get_filename()

		with open(filename) as vector_file:
			reader = csv.reader(vector_file, delimiter=' ')
			vector = list(reader)
			self.b_vector = array(vector).astype("float")
			self.b_vector = self.b_vector.flatten()
		vector_chooser.destroy()

	def jacobi(self, widget, data=None):
		niter = int(self.niter_entry.get_text())
		self.jacobi_serial.jacobi(self.A_matrix, self.b_vector, niter)
