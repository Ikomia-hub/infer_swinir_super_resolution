# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_swinir_super_resolution.infer_swinir_super_resolution_process import InferSwinirSuperResolutionParam
from torch.cuda import is_available

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferSwinirSuperResolutionWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferSwinirSuperResolutionParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Cuda
        is_cuda_available = is_available()
        self.check_cuda = pyqtutils.append_check(self.grid_layout, "Cuda",
                                                 self.parameters.cuda and is_cuda_available)
        if not is_cuda_available:
            self.check_cuda.setEnabled(False)
        # Models
        self.combo_model = pyqtutils.append_combo(self.grid_layout, "Model")

        self.combo_model.addItem("GAN")
        self.combo_model.addItem("PSNR")

        self.combo_model.setCurrentText("GAN" if self.parameters.use_gan else "PSNR")

        self.combo_size = pyqtutils.append_combo(self.grid_layout, "Size")

        self.combo_size.addItem("Large")
        self.combo_size.addItem("Medium")

        self.combo_size.setCurrentText("Large" if self.parameters.large_model else "Medium")

        self.tile_spin = pyqtutils.append_spin(self.grid_layout, "Tile (px)", self.parameters.tile)
        self.overlap_spin = pyqtutils.append_double_spin(self.grid_layout, "Overlap between tiles [0,1]",
                                                         self.parameters.overlap_ratio, min=0, max=1, step=0.01)
        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        use_gan = self.combo_model.currentText() == "GAN"
        large_model = self.combo_size.currentText() == "Large"
        if use_gan != self.parameters.use_gan or large_model != self.parameters.large_model:
            self.parameters.update = True
        self.parameters.use_gan = use_gan
        self.parameters.large_model = large_model
        self.parameters.tile = self.tile_spin.value()
        self.parameters.overlap_ratio = self.overlap_spin.value()
        self.parameters.cuda = self.check_cuda.isChecked()
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferSwinirSuperResolutionWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_swinir_super_resolution"

    def create(self, param):
        # Create widget object
        return InferSwinirSuperResolutionWidget(param, None)
