import gtk
from coefficient_generator import CoefficientGenerator

class CoefficientGeneratorTab():
	def __init__(self):
		self.matrix_filename_entry = None
		self.vector_filename_entry = None
		self.length_entry = None
		self.coefficient_generator = CoefficientGenerator()

	def get_tab(self):
		gen_matrix_box = gtk.VBox()
      
		matrix_length_lbl = gtk.Label("Matrix and vector length")
		gen_matrix_box.pack_start(matrix_length_lbl, True, True, 10)
		self.length_entry = gtk.Entry()
		gen_matrix_box.pack_start(self.length_entry, True, True, 10)

		matrix_filename_lbl = gtk.Label("Matrix filename")
		gen_matrix_box.pack_start(matrix_filename_lbl, True, True, 10)
		self.matrix_filename_entry = gtk.Entry()
		gen_matrix_box.pack_start(self.matrix_filename_entry, True, True, 10)

		vector_filename_lbl = gtk.Label("Vector filename")
		gen_matrix_box.pack_start(vector_filename_lbl, True, True, 10)
		self.vector_filename_entry = gtk.Entry()
		gen_matrix_box.pack_start(self.vector_filename_entry, True, True, 10)

		gen_button = gtk.Button("Generate matrix and vector")
		gen_button.connect("clicked", self.generate_coefficients, None)
		gen_matrix_box.pack_start(gen_button, True, True, 10)

		return gen_matrix_box

	def generate_coefficients(self, widget, data=None):
		matrix_filename = self.matrix_filename_entry.get_text()
		vector_filename = self.vector_filename_entry.get_text()
		length = int(self.length_entry.get_text())
		self.coefficient_generator.gen_matrix(matrix_filename, length)
		self.coefficient_generator.gen_vector(vector_filename, length)

