import gtk
from gen_matrix import GenMatrix

class GenMatrixTab():
	def __init__(self):
		self.filename_entry = None
		self.matrix_length_entry = None
		self.gen_matrix = GenMatrix()

	def get_tab(self):
		gen_matrix_box = gtk.VBox()
      
		filename_lbl = gtk.Label("Filename")
		gen_matrix_box.pack_start(filename_lbl, True, True, 10)
		self.filename_entry = gtk.Entry()
		gen_matrix_box.pack_start(self.filename_entry, True, True, 10)

		matrix_length_lbl = gtk.Label("Matrix length")
		gen_matrix_box.pack_start(matrix_length_lbl, True, True, 10)
		self.matrix_length_entry = gtk.Entry()
		gen_matrix_box.pack_start(self.matrix_length_entry, True, True, 10)

		gen_button = gtk.Button("Generate matrix")
		gen_button.connect("clicked", generate_matrix, None)
		gen_matrix_box.pack_start(gen_button, True, True, 10)

		return gen_matrix_box

	def generate_matrix(self, widget, data=None):
		filename = self.filename_entry.get_text()
		matrix_length = int(self.matrix_length_entry.get_text())
		self.gen_matrix.gen_matrix(filename, matrix_length)

