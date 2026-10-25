from common.components import ComponentBase
from common.dataset import DataSet
from common.eval_component.q_space_eval import QSpaceEval
from common.eval_component.regression import Regression
from common.eval_component.conductivity_eval import ConductivityEval
from common.eval_component.sparse_deconvolution import SparseDeconvolution
from common.eval_component.effective_medium import EffectiveMedium
from common.settings import Settings
from traitlets import Instance
from common.dataset_plotter import ExtendedDataSetPlotter, ImagePlot
from common.components import action
from common.default_appsettings import ComponentToggle

class AppRoot(ComponentBase):
    settings = Instance(Settings, allow_none=True).tag(priority=1000)
    dataset = Instance(DataSet, allow_none=True).tag(priority=10)
    q_space_eval = Instance(QSpaceEval, allow_none=True).tag(toggle=ComponentToggle.en_q_space_eval, priority=11)
    dataset_plotter = Instance(ExtendedDataSetPlotter, allow_none=True).tag(toggle=ComponentToggle.en_extended_dataset_plots)
    image_plot = Instance(ImagePlot, allow_none=True).tag(toggle=ComponentToggle.en_image_plot)
    regression = Instance(Regression, allow_none=True).tag(toggle=ComponentToggle.en_regression)
    conductivity_eval = Instance(ConductivityEval, allow_none=True).tag(
        toggle=ComponentToggle.en_conductivity_eval)
    sparse_deconvolution = Instance(SparseDeconvolution, allow_none=True).tag(
        toggle=ComponentToggle.en_sparse_deconvolution)
    effective_medium = Instance(EffectiveMedium, allow_none=True).tag(toggle=ComponentToggle.en_effective_medium)

    def __init__(self):
        super().__init__()
        self.settings = Settings(object_name="Settings")
        self.dataset = DataSet(self.settings, object_name="Dataset")

        self.dataset_plotter = ExtendedDataSetPlotter(self.dataset, object_name="DataSet plotter")
        self.image_plot = ImagePlot(self.dataset, object_name="Image plot")
        self.q_space_eval = QSpaceEval(self.dataset, image_plot=self.image_plot, object_name="Q-space evaluation")
        self.regression = Regression(self.dataset, object_name="Regression")
        self.conductivity_eval = ConductivityEval(self.dataset, object_name="Conductivity evaluation")
        self.sparse_deconvolution = SparseDeconvolution(self.dataset, object_name="Sparse deconvolution")
        self.effective_medium = EffectiveMedium(self.dataset, q_space_eval=self.q_space_eval,
                                                object_name="Effective medium")

    def __enter__(self, *args):
        super().__enter__(*args)
        self.hello()

        return self

    @action("Hello")
    def hello(self):
        self.logger.warning("hello")

