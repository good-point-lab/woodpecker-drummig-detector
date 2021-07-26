#!/usr/bin/env python

import json
import logging
import os
import pathlib
import queue
import re
import sys
import threading
import time
import tkinter as tk
from json import JSONDecodeError
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

manual = """
The application facilitates the interactive data sets acquisition for training ML models. 
Selected spectrograms could be reviewed and discarded if needed. To acquire spectrograms configure the serial connection, 
start it and use the 'capture' button to save the samples. To review saved spectrograms, select the data directory 
and double-click files to view the samples.

The design utilizes a few patterns - separation of the UI from blocking serial I/O [1], custom logging handler [2], 
TkInter GUI libraries [3].  Acquisition from WiFi or similar network I/O will be added later.

Future Improvements:

- Compile as a platform executable, possibly along with Python runtime, using py2app
- Save the last edited configuration parameters.
- Group layout elements, annotate them with colors and borders
- Mouse wheel crashes the application. Upgrade to the latest Tkinter and Python
- Suppress default “Python” menu in macOS/Tkinter

References and Credits:

[1] Combining Tkinter and Asynchronous I/O with Threads, from Python Cookbook
https://www.oreilly.com/library/view/python-cookbook/0596001673/ch09s07.html
https://code.activestate.com/recipes/82965-threads-tkinter-and-asynchronous-io/
[2] A Logging Handler that allows logging to a Tkinter Text Widget
https://gist.github.com/moshekaplan/c425f861de7bbf28ef06
[3] Tkinter's Grid Geometry Manager
https://blog.teclado.com/tkinters-grid-geometry-manager/
 
 """

VERSION = 0.1
LARGE_FONT = ("Verdana", 14)
DEFAULT_SERIAL_PORT = '/dev/tty.SLAB_USBtoUART'
DEFAULT_SERIAL_RATE = 115200
DEFAULT_DATA_DIR = str(pathlib.Path.cwd()) + os.sep + 'data'
FREQUENCY_BANDS = 7
QUEUE_CHECK_PERIOD_MS = 200

MAX_SPECTROGRAM_AMPLITUDE = 8

serial_port = DEFAULT_SERIAL_PORT
serial_rate = DEFAULT_SERIAL_RATE
data_directory = DEFAULT_DATA_DIR

"""
Logging
"""


class TextHandler(logging.Handler):
    def __init__(self, text):
        logging.Handler.__init__(self)
        self.text = text

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text.configure(state='normal')
            self.text.insert(tk.END, msg + '\n')
            self.text.configure(state='disabled')
            self.text.yview(tk.END)

        # This is necessary because we can't modify the Text from other threads
        self.text.after(0, append)


def create_text_logger(text_widget: object) -> object:
    text_handler = TextHandler(text_widget)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[text_handler]  # file or console logging could be added here
    )
    logger = logging.getLogger()
    return logger


"""
GUI
"""


class AppGui:
    def __init__(self,
                 master,
                 queue,
                 end_command,
                 start_serial_io_cmd,
                 stop_serial_io_cmd
                 ):
        self.logger = None
        self.queue = queue
        # Set up the GUI
        self.master = master
        self.end_command = end_command
        self.start_serial_io_cmd = start_serial_io_cmd
        self.stop_serial_io_cmd = stop_serial_io_cmd
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        container = tk.Frame(master)
        container.master.geometry("1000x600")
        container.master.title('Data Sets Acquisition')
        container.grid(row=0, column=0, sticky="nsew")
        container.master.config(background="bisque")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        menu_bar = tk.Menu(master)
        mode = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label='Options', menu=mode)
        mode.add_command(label='Acquire Serial', command=lambda: self.show_frame(AcquireWithSerialPage))
        mode.add_command(label='Acquire Network', command=lambda: self.show_frame(AcquireWithNetworkPage))
        mode.add_command(label='Browse', command=lambda: self.show_frame(BrowsePage))
        mode.add_separator()
        mode.add_command(label='Exit', command=self.close_all)

        help_options = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label='Help', menu=help_options)
        help_options.add_command(label='Documentation', command=lambda: self.show_frame(HelpPage))

        master.config(menu=menu_bar)

        self.frames = {}
        for F in (AcquireWithSerialPage, AcquireWithNetworkPage, BrowsePage, HelpPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(AcquireWithSerialPage)

    def set_logger(self, logger):
        self.logger = logger

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def close_all(self):
        plt.close('all')
        self.master.quit()  # stops mainloop
        self.master.destroy()
        self.end_command()

    """
    Follow the schema to access individual data elements from the serialized JSON object
    """

    @staticmethod
    def get_elements(json_obj):
        return json_obj["spectrogram"], \
               json_obj["series"], \
               float(json_obj["meta"]["time_min"]), \
               float(json_obj["meta"]["time_max"]), \
               int(json_obj["meta"]["time_points"]), \
               float(json_obj["meta"]["bands"]), \
               float(json_obj["meta"]["freq_grid_sz"]), \
               int(json_obj["meta"]["time_grid_sz"]), \
               int(json_obj["meta"]["amplitude_scale"]), \
               int(json_obj["meta"]["band_feature"])

    def init_plots(self, fig, ax, json_obj):
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.rcParams.update({'font.size': 8})
        # Init color bar
        spectrogram, series, t_min, t_max, time_points, bands, freq_grid_sz, time_grid_sz, max_amplitude, main_band = \
            self.get_elements(json_obj)

        max_amplitude = MAX_SPECTROGRAM_AMPLITUDE
        aux_spectrogram = [[0] * time_grid_sz for i in range(time_grid_sz)]
        aux_spectrogram[0][0] = max_amplitude
        axes_image = ax[1].imshow(aux_spectrogram, cmap='jet')
        fig.colorbar(axes_image, ax=ax[1])
        # Init linear plot
        box0 = ax[1].get_position()
        box1 = ax[0].get_position()
        ax[0].set_position([box1.x0, box1.y0, box1.width*0.8, box1.height])
        ax[0].set(ylabel='Amplitude')

    def draw_plots(self, fig, ax, json_obj):
        spectrogram, series, t_min, t_max, time_points, bands, freq_grid_sz, time_grid_sz, max_amplitude, main_band = \
            self.get_elements(json_obj)

        max_amplitude = MAX_SPECTROGRAM_AMPLITUDE
        # Series (for a band with max total)
        ax[0].set_ylim([0, 1.0 * max_amplitude])
        ax[0].grid(True)
        xt = np.arange(t_min, t_max, t_max / time_points).tolist()
        ax[0].set_title('Series for band ' + str(main_band) + ' and complete spectrogram')
        ax[0].set_xlim([t_min, t_max])
        ax[0].set(ylabel='Amplitude')
        line, = ax[0].plot(xt, series)
        line.set_ydata(series)

        # Spectrogram
        try:
            spectrogram = np.array(spectrogram, dtype=float).T.tolist()  #int
        except ValueError:
            return

        ax[1].imshow(spectrogram,
                     origin='lower',
                     cmap='jet',
                     aspect='auto',
                     vmin=0,
                     vmax=max_amplitude,
                     extent=[t_min, t_max, 0, bands]
                     )
        ax[1].grid(which='major', linestyle='-', linewidth='0.5', color='red')
        ax[1].set(xlabel='Time, sec', ylabel='Frequency Bands')


class AcquireWithSerialPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=0)
        self.columnconfigure(4, weight=0)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=0)
        self.json_obj = None

        lb = tk.Label(
            self,
            text=f"Start serial connection and acquire samples",
            bg="LightBlue2",
            anchor="w"
        )
        lb.grid(row=0, column=0, columnspan=5, padx=10, pady=10, sticky="nsew")

        figure, ax = plt.subplots(nrows=2, ncols=1)
        self.figure = figure
        self.ax = ax
        self.is_plot_init = False

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(column=0, row=1, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.log_text = ScrolledText(self, width=40, height=6, font=("Courier", 12), state="disabled", relief=tk.GROOVE,
                                     borderwidth=2)
        self.log_text.configure(font='TkFixedFont')
        self.log_text.grid(column=2, row=1, columnspan=3, rowspan=2, padx=10, pady=10, sticky="nsw")

        btn_capture = tk.Button(self, text="Capture", command=self.save_sample)
        btn_capture.grid(column=0, row=4, padx=10, pady=10)

        btn_start_serial = tk.Button(self, text="Start Serial", command=self.start_serial_io)
        btn_start_serial.grid(column=1, row=4, padx=10, pady=10)  # 2

        btn_stop_serial = tk.Button(self, text="Stop Serial", command=self.stop_serial_io)
        btn_stop_serial.grid(column=2, row=4, padx=10, pady=10, sticky="nsw")  # 3

        btn_stop_serial = tk.Button(self, text="Configuration", command=self.configure_serial)
        btn_stop_serial.grid(column=3, row=4, padx=10, pady=10, sticky="nsw")

    def get_text_widget(self):
        return self.log_text

    def start_serial_io(self):
        self.controller.start_serial_io_cmd()

    def stop_serial_io(self):
        self.controller.stop_serial_io_cmd()

    def configure_serial(self):
        global serial_port
        global serial_rate
        global data_directory
        config_dialog = ConfigDialog(self)
        self.wait_window(config_dialog.top)
        serial_port = config_dialog.serial_port or DEFAULT_SERIAL_PORT
        try:
            serial_rate = config_dialog.serial_rate or DEFAULT_SERIAL_RATE
        except (ValueError, TypeError):
            serial_rate = DEFAULT_SERIAL_RATE
        data_directory = config_dialog.data_dir or DEFAULT_DATA_DIR

    """
    Process all messages currently in the queue, if any.
    """

    def render_figures(self):
        while self.controller.queue.qsize():
            try:
                self.json_obj = self.controller.queue.get(0)
                if not self.is_plot_init:
                    self.controller.init_plots(self.figure, self.ax, self.json_obj)
                    self.is_plot_init = True
                self.controller.draw_plots(self.figure, self.ax, self.json_obj)
                self.figure.canvas.draw()
                self.figure.canvas.flush_events()
                self.ax[1].clear()
                self.ax[0].clear()
            except queue.Empty:
                pass

    def get_render_figures_method(self):
        return self.render_figures

    def save_sample(self):
        if self.json_obj is None:
            return
        pathlib.Path(data_directory).mkdir(parents=True, exist_ok=True)
        file_name = 's_' + time.strftime('%Y_%m_%d_%H_%M_%S') + '.txt'
        file_path = os.path.join(data_directory, file_name)
        try:
            with open(file_path, 'w') as outfile:
                json.dump(self.json_obj, outfile)
        except Exception as e:
            self.controller.logger.info(str(e))


class AcquireWithNetworkPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        lb = tk.Label(
            self,
            text="TBD",
            bg="LightBlue2",
            anchor="w"
        )
        lb.grid(row=0, column=0, columnspan=6, padx=10, pady=10, sticky="nsew")


class BrowsePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.selected_folder = None
        self.selected_file = None

        label = ttk.Label(self, text="Browse Page", font=LARGE_FONT)
        label.grid(row=0, column=4, padx=10, pady=10)
        self.controller = controller
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=0)
        self.columnconfigure(3, weight=0)
        self.columnconfigure(4, weight=1)
        self.columnconfigure(5, weight=0)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=0)
        self.rowconfigure(4, weight=0)
        self.rowconfigure(5, weight=0)

        lb = tk.Label(
            self,
            text=f"Select data folder and browse samples (double-click to select a file)",
            bg="LightBlue2",
            anchor="w"
        )
        lb.grid(row=0, column=0, columnspan=6, padx=10, pady=10, sticky="nsew")

        figure, ax = plt.subplots(nrows=2, ncols=1)
        self.figure = figure
        self.ax = ax
        self.is_plot_init = False

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(column=0, row=1, columnspan=2, rowspan=2, padx=10, pady=10, sticky="nsew")

        self.y_scroll = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.y_scroll.grid(column=4, row=1, rowspan=2, sticky="nsw", padx=1, pady=1)

        self.listbox = tk.Listbox(
            self,
            yscrollcommand=self.y_scroll.set,
            width=32
        )
        self.listbox.grid(column=2, row=1, columnspan=2, rowspan=2, sticky="nsew", padx=3, pady=1)

        self.y_scroll['command'] = self.listbox.yview
        self.listbox.bind('<Double-1>', self.select_file)

        btn_select = tk.Button(self, text="Select Folder", command=self.select_folder)
        btn_select.grid(column=1, row=3, padx=10, pady=10)

        btn_todo = tk.Button(self, text="Delete File", command=self.delete_file)
        btn_todo.grid(column=2, row=3, padx=10, pady=10)

    def select_file(self, event):
        cs = self.listbox.curselection()
        self.selected_file = self.listbox.get(cs)
        try:
            with open(self.full_path(), 'r') as reader:
                json_str = reader.read()
                json_obj = json.loads(json_str)
        except (IOError, JSONDecodeError, UnicodeDecodeError) as e:
            messagebox.showinfo(title='Error', message='Select actual data file \n Message: ' + str(e))
        else:
            self.render_figures(json_obj)

    def select_folder(self):
        self.listbox.delete(0, tk.END)
        self.selected_folder = filedialog.askdirectory(
            parent=self.master,
            initialdir=pathlib.Path.cwd(),
            title='Select data file'
        )
        folder_files = self.list_files()
        self.listbox.insert(tk.END, *folder_files)

    def delete_file(self):
        if self.selected_file is None:
            return
        rv = messagebox.askquestion('Conformation', "Do you want to delete selected file?")
        if rv == 'yes':
            try:
                os.remove(self.full_path())
                self.listbox.delete(0, tk.END)
                folder_files = self.list_files()
                self.listbox.insert(tk.END, *folder_files)
            except Exception as e:
                messagebox.showinfo(title='Error', message='Delete file error \n Message: ' + str(e))

    def list_files(self):
        if self.selected_folder is not None:
            files = os.listdir(self.selected_folder)
        else:
            files = list()
        return files

    def full_path(self):
        return self.selected_folder + os.sep + self.selected_file

    def render_figures(self, json_obj):
        try:
            if not self.is_plot_init:
                self.controller.init_plots(self.figure, self.ax, json_obj)
                self.is_plot_init = True
            self.controller.draw_plots(self.figure, self.ax, json_obj)
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            self.ax[1].clear()
            self.ax[0].clear()
        except Exception as e:
            self.controller.logger.info(str(e))


class HelpPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        label = tk.Label(
            self,
            text="Description, manual, and release notes",
            bg="LightBlue2",
            anchor="w"
        )
        label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        tx = ScrolledText(
            self,
            width=40,
            height=20,
            font=("Courier", 12),
            relief=tk.GROOVE,
            borderwidth=2
        )
        tx.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        tx.insert(
            tk.INSERT,
            manual
        )
        tx.configure(state='disabled')


class ConfigDialog:
    def __init__(self, parent):
        top = self.top = tk.Toplevel(parent)
        self.top.geometry("400x400")
        self.top.title('Configuration')
        self.top.columnconfigure(0, weight=0)
        self.top.rowconfigure(0, weight=0)
        self.top.rowconfigure(1, weight=0)
        self.top.rowconfigure(2, weight=0)
        self.top.rowconfigure(3, weight=0)
        self.top.rowconfigure(4, weight=0)
        self.top.rowconfigure(5, weight=0)
        self.top.rowconfigure(6, weight=0)

        self.serial_port_lb = tk.Label(top, text='Enter serial port:', anchor="w")
        self.serial_port_lb.grid(column=0, row=0, padx=10, pady=10, sticky="nsew")
        self.serial_port_entry = tk.Entry(top, width=40)
        self.serial_port_entry.insert(0, serial_port)
        self.serial_port_entry.grid(column=0, row=1, padx=10, pady=10, sticky="nsew")
        self.serial_port = None

        self.serial_rate_lb = tk.Label(top, text='Enter serial rate:', anchor="w")
        self.serial_rate_lb.grid(column=0, row=2, padx=10, pady=10, sticky="nsew")
        self.serial_rate_entry = tk.Entry(top)
        self.serial_rate_entry.insert(0, serial_rate)
        self.serial_rate_entry.grid(column=0, row=3, padx=10, pady=10, sticky="nsew")
        self.serial_rate = None

        self.data_dir_lb = tk.Label(top, text='Enter data directory:', anchor="w")
        self.data_dir_lb.grid(column=0, row=4, padx=10, pady=10, sticky="nsew")
        self.data_dir_entry = tk.Entry(top)
        self.data_dir_entry.insert(0, data_directory)
        self.data_dir_entry.grid(column=0, row=5, padx=10, pady=10, sticky="nsew")
        self.data_dir = None

        self.submit_button = tk.Button(top, text='Submit', command=self.send)
        self.submit_button.grid(column=0, row=6, padx=10, pady=10, sticky="nsw")
        self.cancel_button = tk.Button(top, text='Cancel', command=self.cancel)
        self.cancel_button.grid(column=0, row=6, padx=10, pady=10, sticky="nse")

    def send(self):
        self.serial_port = self.serial_port_entry.get()
        self.serial_rate = self.serial_rate_entry.get()
        self.data_dir = self.data_dir_entry.get()
        self.top.destroy()

    def cancel(self):
        self.top.destroy()


"""    
    This class is the application 'engine'. It instantiate GUI, starts asynchronous 
    serial input thread, runs periodic read out from the queue.
"""


class AppBase:

    def __init__(self, master):
        self.master = master
        self.queue = queue.Queue()

        # Set up the application GUI
        self.gui = AppGui(
            master,
            self.queue,
            self.end_application_cmd,
            self.start_serial_io_cmd,
            self.stop_serial_io_cmd
        )
        self.acquire_page = self.gui.frames[AcquireWithSerialPage]
        self.browse_page = self.gui.frames[BrowsePage]

        logging_widget = self.acquire_page.get_text_widget()
        self.logger = create_text_logger(logging_widget)
        self.gui.set_logger(self.logger)

        # Set up the thread to do asynchronous I/O
        # More threads can also be created and used, if necessary
        self.running = True
        self.stop_serial_io = True
        self.pause = False

        self.serial_io_thread = threading.Thread(
            target=self.process_serial_io,
            daemon=True,
            kwargs={
                'logger': self.logger
            }
        )

        self.serial_io_thread.start()
        # Periodic calls in the GUI to consume the received data if any
        self.read_queued_samples()

    """
    The queue consumer side (producer puts serial data to the queue in the serial receiver thread).
    """

    def read_queued_samples(self):
        process_incoming = self.acquire_page.get_render_figures_method()
        process_incoming()
        if not self.running:
            import sys
            sys.exit(1)
        self.master.after(QUEUE_CHECK_PERIOD_MS, self.read_queued_samples)

    """
    Process serial I/O and passes received data to the queue. Executes in a separate thread.
    """

    def process_serial_io(self, logger):
        serial_connection = None
        while self.running:
            if not self.stop_serial_io:
                if serial_connection is None:
                    logger.info("Attempt to connect")
                    try:
                        serial_connection = serial.Serial(serial_port, serial_rate, timeout=4)
                    except Exception as e:
                        if serial_connection is not None:
                            serial_connection.close()
                        serial_connection = None
                        logger.error(str(e))
                        time.sleep(5.0)
                        continue
                    else:
                        logger.info("Connected")

                serial_connection.reset_input_buffer()
                try:
                    buff = serial_connection.readline()
                    buff = re.sub(r'\r\n', '', buff.decode('utf-8'))
                    parsed_json = json.loads(buff)
                except (JSONDecodeError, UnicodeDecodeError) as e:
                    logger.error(str(e))
                    continue
                except (serial.SerialException, IOError) as e:
                    logger.error(str(e))
                    if serial_connection is not None:
                        serial_connection.close()
                    serial_connection = None
                    continue

                self.queue.put(parsed_json)
            else:
                if serial_connection is not None:
                    serial_connection.close()
                serial_connection = None
                time.sleep(2.0)

    def end_application_cmd(self):
        self.running = False

    def start_serial_io_cmd(self):
        self.stop_serial_io = False

    def stop_serial_io_cmd(self):
        self.stop_serial_io = True
        self.logger.info("Stopping Serial")


def on_window_closing():
    sys.exit(1)


def main():
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", on_window_closing)
    client = AppBase(root)
    root.mainloop()


if __name__ == '__main__':
    main()
