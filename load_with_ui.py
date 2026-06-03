import logging
import sys
from config import *
from pathlib import Path
import json
from settings import AppSettings
from qtui.autoui import generate_ui
from PySide6 import QtCore, QtWidgets
from os.path import basename, splitext


class QTextBrowserLoggingHandler(QtCore.QObject, logging.Handler):
    log_signal = QtCore.Signal(str)

    def __init__(self, textBrowser):
        QtCore.QObject.__init__(self)
        logging.Handler.__init__(self)
        self.textBrowser = textBrowser

        self.log_signal.connect(self.append_log_to_ui)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

    def append_log_to_ui(self, msg):
        text = self.textBrowser.toPlainText()
        lines = text.split('\n')[-100:]
        lines.append(msg)

        self.textBrowser.setPlainText('\n'.join(lines))
        self.textBrowser.verticalScrollBar().setValue(
            self.textBrowser.verticalScrollBar().maximum()
        )

def run(globals, filename):
    rootClass = globals["AppRoot"]
    rootClass.script_name = filename

    with rootClass() as root:
        w, msgBrowser = generate_ui(root)
        w.resize(1024, 480)

        logging.captureWarnings(True)
        handler = QTextBrowserLoggingHandler(msgBrowser)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        w.show()

        app = QtWidgets.QApplication.instance()
        exit_code = app.exec()

        logging.getLogger().removeHandler(handler)

    return exit_code


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    filename = sys.argv[1]
    globals = {'__name__': splitext(basename(filename))[0]}
    exec(compile(open(filename, 'rb').read(), filename, 'exec'), globals)

    sys.exit(run(globals, filename))
