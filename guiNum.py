import gtk
from sparse_matrix import create_sparse_matrix
class PyApp(gtk.Window):

  def create_sparse_matrix_page(self):
    sparse_matrix_box = gtk.VBox()
      
    filename_lbl = gtk.Label("Filename")
    sparse_matrix_box.pack_start(filename_lbl, True, True, 10)

    filename_entry = gtk.Entry()
    sparse_matrix_box.pack_start(filename_entry, True, True, 10)

    generate_matrix_btn = gtk.Button("Generate matrix")
    generate_matrix_btn.connect("clicked", self.say_hello, None)
    sparse_matrix_box.pack_start(generate_matrix_btn, True, True, 10)
    return sparse_matrix_box

  def __init__(self):
    super(PyApp, self).__init__()
    self.set_title("Methods")
    self.set_default_size(250, 200)
        
    notebook = gtk.Notebook()
    notebook.set_tab_pos(gtk.POS_TOP)
    vbox = gtk.VBox(False, 5)
        
    #sparse_matrix_box = gtk.VBox()
    hbox = gtk.HBox(True, 3)
        
    valign = gtk.Alignment(0.5,0.25, 0, 0)

    '''
    filename_lbl = gtk.Label("Filename")
    sparse_matrix_box.pack_start(filename_lbl, True, True, 10)

    filename_entry = gtk.Entry()
    sparse_matrix_box.pack_start(filename_entry, True, True, 10)

    generate_matrix_btn = gtk.Button("Generate matrix")
    generate_matrix_btn.connect("clicked", self.say_hello, None)
    sparse_matrix_box.pack_start(generate_matrix_btn, True, True, 10)

    valign.add(sparse_matrix_box)
    '''
    valign.add(creat_sparse_matrix_page())
    vbox.pack_start(valign)
    notebook.append_page(vbox)
    notebook.set_tab_label_text(vbox, "Sparse matrix")


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

  def say_hello(self, widget, data=None):
        print("Hello World")

if __name__ == '__main__':
    PyApp()
    gtk.main()
