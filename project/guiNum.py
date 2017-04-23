import sys
sys.path.append("/usr/lib/python3.6/site-packages/")
sys.path.append("./sparseMatrices")
sys.path.append("./coefficientGenerator")
sys.path.append("./jacobiSerial")
sys.path.append("./jacobiParallel")
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from sparse_matrix_tab import SparseMatrixTab
from coefficient_generator_tab import CoefficientGeneratorTab
from jacobi_serial_tab import JacobiSerialTab
from jacobi_parallel_tab import JacobiParallelTab

class PyApp(Gtk.Window):

  def __init__(self):
    super(PyApp, self).__init__()

    # Tabs of the window.
    self.sparse_matrix_tab = SparseMatrixTab()
    self.coefficient_generator_tab = CoefficientGeneratorTab()
    self.jacobi_serial_tab = JacobiSerialTab()
    self.jacobi_parallel_tab = JacobiParallelTab()

    # Elements of the current window.
    self.set_title("Methods")
    self.set_default_size(500, 400)

    # Creates a notebook to add tabs there.
    notebook = Gtk.Notebook()
    notebook.set_tab_pos(Gtk.PositionType.TOP)

    # Adding sparse matrix
    vbox_sparse_matrix = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    valign_sparse_matrix = Gtk.Alignment.new(0.5,0.25, 0, 0)

    valign_sparse_matrix.add(self.sparse_matrix_tab.get_sparse_tab())
    vbox_sparse_matrix.pack_start(valign_sparse_matrix, True, True, 6)
    notebook.append_page(vbox_sparse_matrix)
    notebook.set_tab_label_text(vbox_sparse_matrix, "Sparse matrix")


    # Adding Matrix generator
    vbox_coefficient_generator = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
    valign_coefficient_generator = Gtk.Alignment.new(0.5,0.25, 0, 0)

    valign_coefficient_generator.add(self.coefficient_generator_tab.get_tab())
    vbox_coefficient_generator.pack_start(valign_coefficient_generator, True, True, 6)
    notebook.append_page(vbox_coefficient_generator)
    notebook.set_tab_label_text(vbox_coefficient_generator, "Coefficient Generator")


    jacobi_box_outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
    valign_jacobi_box_outer1 = Gtk.Alignment.new(0.5,0.25, 0, 0)
    valign_jacobi_box_outer2 = Gtk.Alignment.new(0.5,0.5, 0, 0)

    # Adding serial Jacobi solver
    listbox_jacobi_serial = Gtk.ListBox()
    listbox_jacobi_serial.set_selection_mode(Gtk.SelectionMode.NONE)
    valign_jacobi_box_outer1.add(listbox_jacobi_serial)
    #jacobi_box_outer.pack_start(listbox_jacobi_serial, True, True, 0)

    row = Gtk.ListBoxRow()
    vbox_jacobi_serial = self.jacobi_serial_tab.get_tab()
    row.add(vbox_jacobi_serial)

    listbox_jacobi_serial.add(row)


    # Adding parallel Jacobi solver
    listbox_jacobi_parallel = Gtk.ListBox()
    listbox_jacobi_parallel.set_selection_mode(Gtk.SelectionMode.NONE)
    #jacobi_box_outer.pack_start(listbox_jacobi_parallel, True, True, 0)
    valign_jacobi_box_outer2.add(listbox_jacobi_parallel)

    # Adding Jacobi serial solver
    row = Gtk.ListBoxRow()
    vbox_jacobi_parallel = self.jacobi_parallel_tab.get_tab()
    row.add(vbox_jacobi_parallel)

    listbox_jacobi_parallel.add(row)

    jacobi_box_outer.pack_start(valign_jacobi_box_outer1, True, True, 1)
    jacobi_box_outer.pack_start(valign_jacobi_box_outer2, True, True, 1)
    notebook.append_page(jacobi_box_outer)
    notebook.set_tab_label_text(jacobi_box_outer, "Jacobi's Method")

    # Adding Jacobi Parallel solver










    hb = Gtk.HButtonBox()

    btn1 = Gtk.RadioButton(None,"Degree")
    hb.add(btn1)

    btn2 = Gtk.RadioButton(btn1,"P.G.")
    hb.add(btn2)

    btn3 = Gtk.RadioButton(btn1,"Doctorate")
    hb.add(btn3)

    notebook.append_page(hb)
    notebook.set_tab_label_text(hb, "Qualification")

    tv = Gtk.TextView()
    notebook.append_page(tv)
    notebook.set_tab_label_text(tv, "about")

    self.add(notebook)
    self.connect("destroy", Gtk.main_quit)
    self.show_all()

  def create_sparse_matrix(self, widget, data=None):
    filename = self.filename_entry.get_text()
    self.sparseMatrix.create_sparse_matrix(filename)

if __name__ == '__main__':
    PyApp()
    Gtk.main()
