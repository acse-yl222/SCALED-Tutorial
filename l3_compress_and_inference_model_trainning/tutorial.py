import os
import sys
import importlib.metadata
original_version = importlib.metadata.version
importlib.metadata.version = lambda name: "1.24.4" if name == "numpy" else original_version(name)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
from omegaconf import OmegaConf
import os
import torch
from scaled.dataset.urban_flow_dataset_h5 import UrbanFlowDatasetV1
from scaled.model.unets.unet_3ds import UNet3DsModel
from scaled.runner.urbanflow_latent_diffusion_runner import UrbanFlowLatentDiffusionV1Runner
import sys
import os
from scaled.model.autoencoders.autoencoder3dv1 import AutoencoderKL
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


if __name__ == "__main__":
    compression_weight_path = 'weight/vae-193000.pth'
    model = UNet3DsModel(in_channels=12,
                        out_channels=4,
                        down_block_types=("DownBlock3D", 
                        "DownBlock3D", 
                        "DownBlock3D", 
                        "DownBlock3D"),
                        up_block_types=(
                            "UpBlock3D", 
                        "UpBlock3D", 
                        "UpBlock3D", 
                        "UpBlock3D"),
                        block_out_channels=(128, 256, 384, 512),
                        add_attention=False)
    
    compression_model = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=
        ["DownEncoderBlock3D",
        "DownEncoderBlock3D",
        "DownEncoderBlock3D"],
        up_block_types=
        ["UpDecoderBlock3D",
        "UpDecoderBlock3D",
        "UpDecoderBlock3D"],
        block_out_channels=[
            128,
            256,
            384
        ],
        latent_channels=4,
    )
    compression_model.load_state_dict(torch.load(compression_weight_path,
                                                 map_location='cpu'))
    
    configure_path = 'config/trainning_diffusion_4x4_latent.yaml'
    config = OmegaConf.load(configure_path)
    
    train_dataset = UrbanFlowDatasetV1(
        data_dir="data/SCALED_dataset/south_kensington",
        skip_timestep=50,
        time_steps_list=[i for i in range(0, 4000)])
    val_dataset = UrbanFlowDatasetV1(
        data_dir="data/SCALED_dataset/south_kensington",
        skip_timestep=50,
        time_steps_list=[i for i in range(4000, 5000)])
    dataset_info = {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
    }
    runner = UrbanFlowLatentDiffusionV1Runner(config,model,compression_model,dataset_info)
    runner.run()