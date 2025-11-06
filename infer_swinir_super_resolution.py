# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate process object
        from infer_swinir_super_resolution.infer_swinir_super_resolution_process import InferSwinirSuperResolutionFactory
        return InferSwinirSuperResolutionFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_swinir_super_resolution.infer_swinir_super_resolution_widget import InferSwinirSuperResolutionWidgetFactory
        return InferSwinirSuperResolutionWidgetFactory()
