from PySide6 import QtCore

class FastNameFilterProxyModel(QtCore.QSortFilterProxyModel):

    def __init__(self, filenames=None, parent=None):
        super().__init__(parent)
        self._allowed_filenames = set(filenames) if filenames else None

    def set_allowed_filenames(self, filenames):
        self.beginResetModel()
        self._allowed_filenames = set(filenames) if filenames else None
        self.endResetModel()

    def filterAcceptsRow(self, source_row, source_parent):
        return True

    def flags(self, index):
        default_flags = super().flags(index)

        if not self._allowed_filenames:
            return default_flags

        source_model = self.sourceModel()
        source_index = self.mapToSource(index)

        if source_model.isDir(source_index):
            return default_flags

        filename = source_model.fileName(source_index)
        if filename not in self._allowed_filenames:
            return default_flags & ~QtCore.Qt.ItemIsSelectable & ~QtCore.Qt.ItemIsEnabled

        return default_flags