import random
import torch
import os
from diffusers.utils.import_utils import is_xformers_available
import os
from .urbanflow_regression_runner import UrbanFlowRegressionRunner
from tqdm import tqdm
import torch.nn.functional as F
import time
import math
from .urbanflow_diffusion_runner import UrbanflowDiffusionRunner

class UrbanflowDiffusionV1Runner(UrbanflowDiffusionRunner):
    def __init__(self, config, model,compression_model, dataset_info=None):
        super().__init__(config, model, dataset_info)
        self.compression_model =compression_model
        self.compression_model.to(self.accelerator.device)
