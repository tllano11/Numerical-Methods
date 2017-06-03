import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from sparse_matrix import SparseMatrix


class SparseMatrixTab():
    def __init__(self):
        self.sparseMatrix = SparseMatrix()
        self.filename_entry = None
        self.matrix_length_entry = None
        self.matrix_density_entry = None
        self.filename = None

    def get_sparse_tab(self):
        sparse_matrix_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        matrix_length_lbl = Gtk.Label("Matrix length")
        sparse_matrix_box.pack_start(matrix_length_lbl, True, True, 10)
        self.matrix_length_entry = Gtk.Entry()
        sparse_matrix_box.pack_start(self.matrix_length_entry, True, True, 10)

        matrix_density_lbl = Gtk.Label("Matrix density")
        sparse_matrix_box.pack_start(matrix_density_lbl, True, True, 10)
        self.matrix_density_entry = Gtk.Entry()
        sparse_matrix_box.pack_start(self.matrix_density_entry, True, True, 10)

        image = Gtk.Image(stock=Gtk.STOCK_SAVE_AS)
        button1 = Gtk.Button(" Save matrix as", image=image)
        button1.connect("clicked", self.create_sparse_matrix)
        sparse_matrix_box.pack_start(button1, True, True, 10)

        button2 = Gtk.Button("Multiply")
        button2.connect("clicked", self.multiply)
        sparse_matrix_box.pack_start(button2, True, True, 10)

        return sparse_matrix_box

    def create_sparse_matrix(self, widget, data=None):
        dialog = Gtk.FileChooserDialog("Please choose a file", None,
                                       Gtk.FileChooserAction.SAVE,
                                       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                        Gtk.STOCK_SAVE, Gtk.ResponseType.OK))

        Gtk.FileChooser.set_current_name(dialog, "matrix.txt")
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.filename = Gtk.FileChooser.get_filename(dialog)
            matrix_length = int(self.matrix_length_entry.get_text())
            density = float(self.matrix_density_entry.get_text())
            self.sparseMatrix.create_sparse_matrix(filename, matrix_length, density)

        dialog.destroy()

    def multiply(self, widget, data=None):
        vector_chooser = Gtk.FileChooserDialog("Select vector file", None, Gtk.FileChooserAction.OPEN,
                                               (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                                Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        response = vector_chooser.run()
        if response == Gtk.ResponseType.OK:
            filename = vector_chooser.get_filename()

            with open(filename) as vector_file:
                reader = csv.reader(vector_file, delimiter=' ')
                vector = list(reader)
                vector = np.array(vector).astype("float64")
        sparseMatrix = SparseMatrix()
        SparseMatrix.multiply(self.filename, vector)
        matrix_chooser.destroy()
