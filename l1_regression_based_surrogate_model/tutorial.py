import os
import sys
import importlib.metadata
original_version = importlib.metadata.version
importlib.metadata.version = lambda name: "1.24.4" if name == "numpy" else original_version(name)
PROJECT_ROOT1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROJECT_ROOT1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT1)
from scaled.model.unets.unet_3ds import UNet3DsModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
from scaled.model.unets.unet_3ds import UNet3DsModel
import torch
import matplotlib.pyplot as plt
import numpy as np
from scaled.model.autoencoders.autoencoder3dv1 import AutoencoderKL
from tqdm import tqdm

width = 128
height = 128
depth = 64

compression_weight = 'weight/compression.pth'
inference_weight = 'weight/inference.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- loading the model ----
compression_model = AutoencoderKL(
    in_channels=3, out_channels=3,
    down_block_types=["DownEncoderBlock3D", "DownEncoderBlock3D", "DownEncoderBlock3D"],
    up_block_types=["UpDecoderBlock3D", "UpDecoderBlock3D", "UpDecoderBlock3D"],
    block_out_channels=[128, 256, 384],
    latent_channels=4,
)
compression_model.load_state_dict(torch.load(compression_weight, map_location="cpu"))
compression_model.to(device).eval()
inference_model = UNet3DsModel(
    in_channels=8, out_channels=4,
    down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D"),
    up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D"),
    block_out_channels=(128, 256, 384, 512),
    add_attention=False
)
inference_model.load_state_dict(torch.load(inference_weight, map_location="cpu"))
inference_model.to(device).eval()

# we first initialize a initial flow speed as zeros
x0 = torch.zeros((1,3,depth,height,width)).to('cuda')/3
geometry_ = torch.zeros(depth,height,width)
geometry_[:,8:-8,8:-8] = torch.tensor(np.load('geo.npy'))[:,8:-8,8:-8]
geometry = geometry_.bool()
xbc = torch.ones((1, 3, depth, height, width), dtype=torch.float32)
xbc[:,:,geometry] = 0
xbc = xbc.to(device)

# compress the intial flow and the boundary condition into latent
latent_x0 = compression_model.encode(x0)/10
latent_xbc = compression_model.encode(xbc)/10

def visualize(data,step=0):
    plt.figure(figsize=(5, 5))
    plt.imshow(data,vmax=1,vmin=-0.5)
    plt.title(f"Timestep {step}")
    plt.colorbar()
    plt.savefig(f'result/{step}.png',dpi=300)
    plt.close()

with torch.no_grad():
    for step in tqdm(range(100)):
        input = torch.cat([latent_x0,latent_xbc],dim=1)
        output = inference_model(input).sample
        latent_x0 = output.clone()
        decode_latent = compression_model.decode(latent_x0*10)
        visualize(decode_latent.detach().cpu().numpy()[0,0,4]*-3,step)