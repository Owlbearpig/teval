from PySide6 import QtWidgets, QtCore
from pathlib import Path


class MultiFileSelectorWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Layout setup
        layout = QtWidgets.QVBoxLayout(self)

        # Create List Widget
        self.file_list = QtWidgets.QListWidget()
        # Enable multiple selection (Hold Ctrl / Shift to multi-select)
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.file_list)

        # Add files button
        self.add_btn = QtWidgets.QPushButton("Select Files...")
        self.add_btn.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.add_btn)

    def open_file_dialog(self):
        # QFileDialog method that accepts selecting multiple files at once
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            caption="Select Multiple Files",
            filter="All Files (*)"
        )
        if files:
            for file_path in files:
                # Avoid duplicate entries
                if not self.file_list.findItems(file_path, QtCore.Qt.MatchExact):
                    self.file_list.addItem(file_path)

    def get_selected_files(self) -> list[Path]:
        """Returns Path instances for items currently highlighted/selected in the list."""
        return [Path(item.text()) for item in self.file_list.selectedItems()]

    def get_all_files(self) -> list[Path]:
        """Returns Path instances for all items loaded into the list."""
        return [Path(self.file_list.item(i).text()) for i in range(self.file_list.count())]