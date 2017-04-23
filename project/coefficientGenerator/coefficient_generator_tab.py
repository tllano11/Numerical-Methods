import sys
sys.path.append("/usr/lib/python3.6/site-packages/")
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
import sys
from coefficient_generator import CoefficientGenerator

class CoefficientGeneratorTab():
        def __init__(self):
                self.matrix_filename_entry = None
                self.vector_filename_entry = None
                self.length_entry = None
                self.coefficient_generator = CoefficientGenerator()

        def get_tab(self):
                gen_matrix_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

                matrix_length_lbl = Gtk.Label("Matrix and vector length")
                gen_matrix_box.pack_start(matrix_length_lbl, True, True, 10)
                self.length_entry = Gtk.Entry()
                gen_matrix_box.pack_start(self.length_entry, True, True, 10)

                matrix_filename_lbl = Gtk.Label("Matrix filename")
                gen_matrix_box.pack_start(matrix_filename_lbl, True, True, 10)
                self.matrix_filename_entry = Gtk.Entry()
                gen_matrix_box.pack_start(self.matrix_filename_entry, True, True, 10)

                vector_filename_lbl = Gtk.Label("Vector filename")
                gen_matrix_box.pack_start(vector_filename_lbl, True, True, 10)
                self.vector_filename_entry = Gtk.Entry()
                gen_matrix_box.pack_start(self.vector_filename_entry, True, True, 10)

                gen_button = Gtk.Button("Generate matrix and vector")
                gen_button.connect("clicked", self.generate_coefficients, None)
                gen_matrix_box.pack_start(gen_button, True, True, 10)

                return gen_matrix_box

        def generate_coefficients(self, widget, data=None):
                matrix_filename = self.matrix_filename_entry.get_text()
                vector_filename = self.vector_filename_entry.get_text()
                length = int(self.length_entry.get_text())
                self.coefficient_generator.gen_matrix(matrix_filename, length)
                self.coefficient_generator.gen_vector(vector_filename, length)

