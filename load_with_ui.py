import html
import logging
import sys
from common.components import ComponentBase
from qtui.autoui import generate_ui
from PySide6 import QtCore, QtWidgets
from os.path import basename, splitext


class QHTMLTextBrowserLoggingHandler(QtCore.QObject, logging.Handler):
    log_signal = QtCore.Signal(str)

    COLOR_MAP = {
        'INFO': 'green',
        'WARNING': '#E6A100',
        'ERROR': 'red',
        'CRITICAL': 'darkred',
    }

    def __init__(self, textBrowser, max_lines=lambda: 250):
        # max_lines: callable, so a changed setting applies to the next message
        QtCore.QObject.__init__(self)
        logging.Handler.__init__(self)
        self.textBrowser = textBrowser
        self.max_lines = max_lines

        self.log_signal.connect(self.append_log_to_ui)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

    def append_log_to_ui(self, msg):
        document = self.textBrowser.document()
        # one block per message, the new one is appended below
        excess = document.blockCount() - max(int(self.max_lines()), 1) + 1
        if excess > 0:
            cursor = self.textBrowser.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            # by blocks, not visual lines: Down would count wrapped lines
            cursor.movePosition(cursor.MoveOperation.NextBlock, cursor.MoveMode.KeepAnchor, excess)
            cursor.removeSelectedText()

        self.textBrowser.append(msg)
        self.textBrowser.verticalScrollBar().setValue(
            self.textBrowser.verticalScrollBar().maximum()
        )


class HTMLFormatter(logging.Formatter):
    def __init__(self, object_name=""):
        super().__init__()

    def format(self, record):
        color = QHTMLTextBrowserLoggingHandler.COLOR_MAP.get(record.levelname, "black")
        asctime = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        obj_name = getattr(record, "object_name", record.name)
        obj_prefix = f"[{obj_name}] " if obj_name and obj_name != 'root' else ""

        if getattr(record, "action_header", False):
            # empty line, then the pressed button as a highlighted line
            return (f'<br><span style="background-color:#dce6f4; color:#1f4e8c;">&nbsp;<b>&#9654; '
                    f'{html.escape(record.getMessage())}</b>&nbsp;{html.escape(obj_prefix)}{asctime}&nbsp;</span>')

        level_html = f'<font color="{color}"><b>{record.levelname}</b></font>'

        return f"{asctime} {level_html} {obj_prefix}: {record.getMessage()}"

def run(globals, filename : str):
    rootClass = globals["AppRoot"]
    ComponentBase.script_name = filename

    with rootClass() as root:
        w, msgBrowser = generate_ui(root)
        available = QtWidgets.QApplication.primaryScreen().availableGeometry()
        size = QtCore.QSize(max(1410, int(0.9 * available.width())),
                            max(792, int(0.9 * available.height())))
        w.resize(size.boundedTo(available.size()))
        w.move(available.center() - w.rect().center())

        logging.captureWarnings(True)
        handler = QHTMLTextBrowserLoggingHandler(msgBrowser, max_lines=lambda: root.settings.max_log_lines)
        handler.setFormatter(HTMLFormatter())
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
    app.setStyle('Fusion')

    filename = sys.argv[1]
    globals = {'__name__': splitext(basename(filename))[0]}
    exec(compile(open(filename, 'rb').read(), filename, 'exec'), globals)

    sys.exit(run(globals, filename))
