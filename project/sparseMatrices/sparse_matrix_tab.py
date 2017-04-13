import gtk
from sparse_matrix import SparseMatrix

class SparseMatrixTab():
	def __init__(self):
		self.sparseMatrix = SparseMatrix()
		self.filename_entry = None
		self.matrix_length_entry = None
		self.matrix_density_entry = None

	def get_sparse_tab(self):
		sparse_matrix_box = gtk.VBox()
      
		filename_lbl = gtk.Label("Filename")
		sparse_matrix_box.pack_start(filename_lbl, True, True, 10)
		self.filename_entry = gtk.Entry()
		sparse_matrix_box.pack_start(self.filename_entry, True, True, 10)

		matrix_length_lbl = gtk.Label("Matrix length")
		sparse_matrix_box.pack_start(matrix_length_lbl, True, True, 10)
		self.matrix_length_entry = gtk.Entry()
		sparse_matrix_box.pack_start(self.matrix_length_entry, True, True, 10)

		matrix_density_lbl = gtk.Label("Matrix density")
		sparse_matrix_box.pack_start(matrix_density_lbl, True, True, 10)
		self.matrix_density_entry = gtk.Entry()
		sparse_matrix_box.pack_start(self.matrix_density_entry, True, True, 10)

		generate_matrix_btn = gtk.Button("Generate matrix")
		generate_matrix_btn.connect("clicked", self.create_sparse_matrix, None)
		sparse_matrix_box.pack_start(generate_matrix_btn, True, True, 10)
		return sparse_matrix_box

	def create_sparse_matrix(self, widget, data=None):
		filename = self.filename_entry.get_text()
		matrix_length = int(self.matrix_length_entry.get_text())
		density = float(self.matrix_density_entry.get_text())
		self.sparseMatrix.create_sparse_matrix(filename, matrix_length, density)
