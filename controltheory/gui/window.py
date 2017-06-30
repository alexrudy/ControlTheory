import Tkinter as Tk
import os, os.path
import datetime as dt
from os.path import join as pjoin

import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
quantity_support()

import h5py
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

from ..fourier import frequencies
from ..periodogram import periodogram

def read(h5group):
    """Read a masked group"""
    mask = h5group['mask'][...] != -1
    assert mask.any(), "Some elements must be valid."
    axis = int(h5group['data'].attrs.get("TAXIS", 0))
    rate = 250 * u.Hz
    # rate = parse_rate(h5group.attrs.get("WFSCAMRATE", 1.0)) / 2.0
    n = h5group['data'].shape[axis]
    times = np.linspace(0, n / rate.value, n) / rate.unit
    mtimes = np.compress(mask, times)
    mdata = np.compress(mask, h5group['data'], axis=axis)
    mdata = np.moveaxis(mdata, axis, -1)
    return mtimes.to(u.s), mdata

class FileChooser(Tk.Frame, object):
    """Periodogram frame window"""
    def __init__(self, root):
        super(FileChooser, self).__init__(root)
        self._closedloop = None
        self._openloop = None
        self._file_ol = TelemetryFile(self)
        self._file_ol.label.config(text="Open Loop:")
        self._file_ol.grid(row=0, column=0, padx=3, pady=3)
        self._file_cl = TelemetryFile(self)
        self._file_cl.label.config(text="Closed Loop:")
        self._file_cl.grid(row=1, column=0, padx=3, pady=3)
        
        self._keys = Tk.Frame(self)
        self._key_var = Tk.StringVar()
        self._key_label = Tk.Label(self._keys, text='Key:', anchor=Tk.W)
        self._key_label.grid(row=0, column=0)
        self._key_menubutton = Tk.Menubutton(self._keys, text="  ")
        self._key_menubutton.config(width=10)
        self._key_menubutton.grid(row=0, column=1)
        self._key_menu = Tk.Menu(self._key_menubutton)
        self._key_menubutton.config(menu=self._key_menu)
        self._keys.grid(row=2, column=0, padx=3, pady=3)
        
        self._key_var.trace("w", self.make_choice)
        self._file_cl.callbacks.append(self.setup_choices)
        self._file_ol.callbacks.append(self.setup_choices)
        
        self.callbacks = []
        
        self._file_ol.on_set()
        self._file_cl.on_set()
        
    def make_choice(self, *unused):
        """Make a choice"""
        self._key_menubutton.config(text=self._key_var.get())
        self._closedloop = None
        self._openloop = None
        for callback in self.callbacks:
            callback()
        
    def setup_choices(self):
        """Set up telemetry choices"""
        if self._file_ol.exists and self._file_cl.exists:
            with h5py.File(self._file_ol.path, 'r') as fol:
                keys_ol = list(fol.get('telemetry', {}).keys())
            with h5py.File(self._file_cl.path, 'r') as fcl:
                keys_cl = list(fcl.get('telemetry', {}).keys())
            keys = sorted((set(keys_ol) & set(keys_cl)))
            self._key_menu.delete(0, Tk.END)
            for key in keys:
                self._key_menu.add_radiobutton(value=key, variable=self._key_var, label=key)
            self._closedloop = None
            self._openloop = None
            if self._key_var.get() in keys:
                self._key_var.set(self._key_var.get())
        
    def openloop(self):
        """Get the open loop dataset."""
        if self._openloop is None:
            with h5py.File(self._file_ol.path, 'r') as fol:
                self._openloop = read(fol['telemetry'][self._key_var.get()])
        return self._openloop
        
    def closedloop(self):
        """Get the open loop dataset."""
        if self._closedloop is None:
            with h5py.File(self._file_cl.path, 'r') as fcl:
                self._closedloop = read(fcl['telemetry'][self._key_var.get()])
        return self._closedloop
    
class Timeline(Tk.Toplevel, object):
    """Timeline view"""
    def __init__(self, root, chooser):
        super(Timeline, self).__init__(root)
        self.title("Timeline")
        self.figure = Figure(figsize=(7, 4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, column=0)
        self.chooser = chooser
        self.chooser.callbacks.append(self.update)
        
    def update(self, *unused):
        """Clear the axes."""
        self.axes.clear()
        x, y = self.chooser.openloop()
        self.axes.plot(x, np.median(y, axis=0))
        x, y = self.chooser.closedloop()
        self.axes.plot(x, np.median(y, axis=0))
        self.axes.set_xlabel("Time ({0:s})".format(x.unit))
        self.axes.grid(True)
        self.canvas.draw()

class Periodogram(Tk.Toplevel, object):
    """Timeline view"""
    def __init__(self, root, chooser):
        super(Periodogram, self).__init__(root)
        self.title("Periodgoram")
        self.figure = Figure(figsize=(7, 4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, column=0)
        self.chooser = chooser
        self.chooser.callbacks.append(self.update)
        
    def update(self, *unused):
        """Clear the axes."""
        self.axes.clear()
        x, y = self.chooser.openloop()
        fy = periodogram(y, length=256, axis=-1)
        fx = frequencies(256, 250)
        
        self.axes.plot(fx, np.median(fy, axis=0))
        self.axes.set_yscale('log')
        x, y = self.chooser.closedloop()
        fy = periodogram(y, length=256, axis=-1)
        fx = frequencies(256, 250)
        self.axes.plot(fx, np.median(fy, axis=0))
        self.axes.set_xlabel("Frequency ({0:s})".format(u.Hz))
        self.axes.grid(True)
        self.canvas.draw()

class Transfer(Tk.Toplevel, object):
    """Timeline view"""
    def __init__(self, root, chooser):
        super(Transfer, self).__init__(root)
        self.title("Transfer Function")
        self.figure = Figure(figsize=(7, 4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, column=0)
        self.chooser = chooser
        self.chooser.callbacks.append(self.update)
        
    def update(self, *unused):
        """Clear the axes."""
        self.axes.clear()
        x, y = self.chooser.openloop()
        fyo = periodogram(y, length=256, axis=-1)
        fx = frequencies(256, 250)
        x, y = self.chooser.closedloop()
        fyc = periodogram(y, length=256, axis=-1)
        
        self.axes.plot(fx, np.median(fyc / fyo, axis=0))
        self.axes.set_yscale('log')
        self.axes.set_xscale('log')
        self.axes.set_xlim(0.5, 250)
        self.axes.set_xlabel("Frequency ({0:s})".format(u.Hz))
        self.axes.grid(True)
        self.canvas.draw()


class TelemetryFile(Tk.Frame, object):
    """Telemetry file finder / frame"""
    def __init__(self, root):
        super(TelemetryFile, self).__init__(root)
        
        self.label = Tk.Label(self, text="File:", width=10, anchor=Tk.W)
        self.label.grid(row=0, column=0, sticky=Tk.W)
        
        self._path_root_var = Tk.StringVar()
        self._path_root_var.set("~/Documents/Telemetry/ShaneAO")
        self._path_root_widget = Tk.Entry(self, textvariable=self._path_root_var, width=20)
        self._path_root_widget.grid(row=0, column=1)
        
        self._date_var = Tk.StringVar()
        self._date_var.set(dt.date(2017, 06, 06).strftime("%Y-%m-%d"))
        self._date_widget = Tk.Entry(self, textvariable=self._date_var, width=10)
        self._date_widget.grid(row=0, column=2)
        
        self._number_var = Tk.IntVar()
        self._number_var.set('0')
        self._number_widget = Tk.Entry(self, textvariable=self._number_var, width=4)
        self._number_widget.grid(row=0, column=3)
        
        # self._set_button = Tk.Button(self, command=self.on_set, text='Set')
        # self._set_button.grid(row=0, column=4)
        
        self.status = Tk.Label(self, text='', anchor=Tk.W)
        self.status.grid(row=2, column=0, columnspan=3, sticky=Tk.W)
        
        self.callbacks = []
        self._path_root_var.trace('w', self.on_set)
        self._date_var.trace('w', self.on_set)
        self._number_var.trace('w', self.on_set)
        
        self.on_set()
        
    @property
    def number(self):
        """Telemetry number, default to 0"""
        try:
            return int(self._number_var.get())
        except ValueError:
            return 0
        
    @property
    def stem(self):
        """Path stem"""
        return pjoin(self._date_var.get(), "telemetry_{0:04d}.hdf5".format(self.number))
        
    @property
    def exists(self):
        """Does the file exist?"""
        return os.path.exists(self.path)
        
    @property
    def path(self):
        """Full path."""
        root = os.path.expanduser(self._path_root_var.get())
        return pjoin(root, self.stem)
    
    def on_set(self, *unused):
        """On set filename."""
        if self.exists:
            self.status.config(text="Found telemetry_{0:04d}.hdf5".format(self.number))
        else:
            self.status.config(text="Can't find file: {0}".format(self.stem))
        for callback in self.callbacks:
            callback()

def main():
    """Main function."""
    root = Tk.Tk()
    Tk.Label(root, text='Telemetry GUI').grid(row=0, column=0)
    chooser = FileChooser(root)
    chooser.grid(row=0, column=0)
    root.title("File Chooser")
    tl = Timeline(root, chooser)
    pg = Periodogram(root, chooser)
    tf = Transfer(root, chooser)
    # per.grid(row=0, column=0)
    root.mainloop()

if __name__ == '__main__':
    main()