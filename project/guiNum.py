import gtk
import sys
sys.path.append("/home/jocamp18/Desktop/gitHub/Numerical-Methods/project/sparseMatrices")
from sparse_matrix_tab import SparseMatrixTab

class PyApp(gtk.Window):

  def __init__(self):
    super(PyApp, self).__init__()

    # Tabs of the window.
    self.sparse_matrix_tab = SparseMatrixTab()

    # Elements of the current window.
    self.set_title("Methods")
    self.set_default_size(500, 400)
        
    # Creates a notebook to add tabs there.
    notebook = gtk.Notebook()
    notebook.set_tab_pos(gtk.POS_TOP)
    vbox_sparse_matrix = gtk.VBox(False, 5)
        
    sparse_matrix_box = gtk.VBox()
    hbox = gtk.HBox(True, 3)
        
    valign = gtk.Alignment(0.5,0.25, 0, 0)

    # Adding sparse matrix
    valign.add(self.sparse_matrix_tab.get_sparse_tab())
    vbox_sparse_matrix.pack_start(valign)
    notebook.append_page(vbox_sparse_matrix)
    notebook.set_tab_label_text(vbox_sparse_matrix, "Sparse matrix")


    hb = gtk.HButtonBox()
        
    btn1 = gtk.RadioButton(None,"Degree")
    hb.add(btn1)
        
    btn2 = gtk.RadioButton(btn1,"P.G.")
    hb.add(btn2)
        
    btn3 = gtk.RadioButton(btn1,"Doctorate")
    hb.add(btn3)
        
    notebook.append_page(hb)
    notebook.set_tab_label_text(hb, "Qualification")
        
    tv = gtk.TextView()
    notebook.append_page(tv)
    notebook.set_tab_label_text(tv, "about")
        
    self.add(notebook)
    self.connect("destroy", gtk.main_quit)
    self.show_all()

  def create_sparse_matrix(self, widget, data=None):
    filename = self.filename_entry.get_text()
    self.sparseMatrix.create_sparse_matrix(filename)

if __name__ == '__main__':
    PyApp()
    gtk.main()
