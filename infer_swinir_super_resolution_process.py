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
from ikomia.dataprocess import tile_processing
import copy
from infer_swinir_super_resolution.plugin_utils import model_zoo
import requests
import os
import torch
from infer_swinir_super_resolution.SwinIR.main_test_swinir import define_model, setup, get_image_pair, test
import numpy as np
from ikomia.utils import strtobool
from argparse import Namespace


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferSwinirSuperResolutionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.windowSize = 25
        self.update = False
        self.large_model = False
        self.use_gan = True
        self.tile = 256
        self.overlap_ratio = 0.1
        self.scale = 4
        self.cuda = True

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        self.update = strtobool(param_map["cuda"]) != self.cuda or self.large_model != strtobool(param_map['large_model']) or self.scale != int(param_map["scale"])
        self.large_model = strtobool(param_map["large_model"])
        self.use_gan = strtobool(param_map["use_gan"])
        self.tile = int(param_map["tile"])
        self.overlap_ratio = float(param_map["overlap_ratio"])
        self.scale = int(param_map["scale"])
        self.cuda = strtobool(param_map["cuda"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        # Example : paramMap["windowSize"] = str(self.windowSize)
        param_map["large_model"] = str(self.large_model)
        param_map["use_gan"] = str(self.use_gan)
        param_map["tile"] = str(self.tile)
        param_map["overlap_ratio"] = str(self.overlap_ratio)
        param_map["scale"] = str(self.scale)
        param_map["cuda"] = str(self.cuda)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferSwinirSuperResolution(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        # Example :  self.add_input(dataprocess.CImageIO())
        #           self.add_output(dataprocess.CImageIO())
        self.model = None
        self.device =None

        # Create parameters class
        if param is None:
            self.set_param_object(InferSwinirSuperResolutionParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Examples :
        # Get input :
        input = self.get_input(0)

        # Get output :
        output = self.get_output(0)

        # Get parameters :
        param = self.get_param_object()

        assert param.scale in [2, 4], "Scale factor can be only 2 or 4"
        assert not (param.large_model and param.scale==2), "Large models only with scale==4"

        # Get image from input/output (numpy array):
        srcImage = input.get_image()

        if param.update or self.model is None:
            self.device = torch.device('cuda' if param.cuda and torch.cuda.is_available() else 'cpu')
            self.args = Namespace()
            self.args.folder_lq = None
            self.args.folder_gt = None
            self.args.task = 'real_sr'

            self.args.scale = param.scale
            self.args.large_model = param.large_model

            self.args.model_path = os.path.dirname(
                __file__) + "/model_zoo/swinir/" + model_zoo['gan' if param.use_gan else 'psnr']['large' if param.large_model else 'medium'][str(param.scale)]

            # set up model
            if os.path.exists(self.args.model_path):
                print(f'loading model from {self.args.model_path}')
            else:
                os.makedirs(os.path.dirname(self.args.model_path), exist_ok=True)
                url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(
                    os.path.basename(self.args.model_path))
                r = requests.get(url, allow_redirects=True)
                print(f'downloading model {self.args.model_path}')
                open(self.args.model_path, 'wb').write(r.content)

            # setup folder and path
            self.folder, self.save_dir, self.border, self.window_size = setup(self.args)
            self.args.tile = None  # 256 // self.window_size * self.window_size

            self.model = define_model(self.args)
            self.model.eval()
            self.model = self.model.to(self.device)
            param.update = False

        if self.model is not None:
            if srcImage is not None:
                # wrap args in function to work with ikomia.dataprocess.tile_processing.tile_process
                def process(img):
                    return self.infer(self.args, img)

                tile_size = param.tile
                overlap_ratio = param.overlap_ratio / 2
                upscale_ratio = self.args.scale
                divisor = self.window_size
                minimum_size = divisor
                dstImage = tile_processing.tile_process(srcImage, tile_size, overlap_ratio,
                                                        upscale_ratio, divisor, minimum_size, process)

                # Set image of input/output (numpy array):
                output.set_image(np.array(dstImage, dtype='uint8'))
            else:
                print("No input image")
        else:
            print("No model loaded")

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()

    def infer(self, args, img_lq):
        img_lq = np.copy((img_lq.astype(np.float32) / 255.))

        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                              (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, self.model, args, self.window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()

        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

        return output


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferSwinirSuperResolutionFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_swinir_super_resolution"
        self.info.short_description = "Image restoration algorithms with Swin Transformer"
        self.info.description = "Image restoration algorithms with Swin Transformer" \
                                "It includes denoising, deblurring and super resolution"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Super Resolution"
        self.info.icon_path = "icons/swinir.png"
        self.info.version = "1.0.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu"
        self.info.article = "SwinIR: Image Restoration Using Swin Transformer"
        self.info.journal = "arXiv"
        self.info.year = 2022
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentation_link = ""
        # Code source repository
        self.info.repository = "https://github.com/JingyunLiang/SwinIR"
        # Keywords used for search
        self.info.keywords = "swin transformer, super resolution, denoising, deblurring"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "SUPER_RESOLUTION"

    def create(self, param=None):
        # Create process object
        return InferSwinirSuperResolution(self.info.name, param)
