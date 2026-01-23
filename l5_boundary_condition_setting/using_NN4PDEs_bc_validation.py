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
from scaled.pipelines.pipline_ddim_scaled_urbanflow import SCALEDUrbanFlowPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from tqdm import tqdm

def visualize(data,step=0):
    plt.figure(figsize=(5, 5))
    plt.imshow(data,vmax=1,vmin=-0.5)
    plt.title(f"Timestep {step}")
    plt.colorbar()
    plt.savefig(f'result/{step:03d}.png',dpi=300)
    plt.close()
    
def get_velocity_filed(index,start_x,start_y,width,height,device,dtype=torch.float32):
    path = f'data/SCALED_dataset/south_kensington/00{index}.h5'
    import h5py
    with h5py.File(path, 'r') as h5f:
        velocity = h5f['uvw'][:]
    result = torch.tensor(velocity, device=device, dtype=dtype)[:,:,start_x:start_x+width,start_y:start_y+height]/3
    result = result.unsqueeze(0)
    return result

def load_compression_model(weight_path,device):
    model = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=["DownEncoderBlock3D", "DownEncoderBlock3D", "DownEncoderBlock3D"],
        up_block_types=["UpDecoderBlock3D", "UpDecoderBlock3D", "UpDecoderBlock3D"],
        block_out_channels=[128, 256, 384],
        latent_channels=4,
    )
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.to(device).eval()
    return model

def load_inference_model(weight_path,device):
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
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.to(device).eval()
    return model

def load_geometry(path,start_x,start_y,width,height,device,dtype=torch.float32):
    geometry = torch.tensor(np.load(path))[0,0,:,start_x:start_x+width,start_y:start_y+height]/1e8
    geometry = geometry.to(device=device, dtype=dtype)
    return geometry

def pack_boundary_condition(velocity_field,geometry,halo_size=4):
    bc = velocity_field.clone()
    bc[:,:,:,halo_size:-halo_size, halo_size:-halo_size] = 1
    bc[:,:,geometry==1] = 0
    return bc

def get_noise_scheduler():
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        steps_offset=1,
        clip_sample=False,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
        prediction_type="v_prediction",
    )
    return noise_scheduler

width = 256
height = 256
depth = 64
# area start point:
x = 256
y = 256

compression_weight = 'weight/compression.pth'
inference_weight = 'weight/inference.pth'
geometry_path = 'data/SCALED_dataset/south_kensington/sigma.npy'

initial_index = 1000
initial_urbanflow_path = f'data/SCALED_dataset/south_kensington/00{initial_index}.h5'
next_urbanflow_path = f'data/SCALED_dataset/south_kensington/00{initial_index+50}.h5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- loading the model ----
compression_model = load_compression_model(compression_weight, device)
inference_model = load_inference_model(inference_weight, device)
val_noise_scheduler = get_noise_scheduler()

geometry = load_geometry(geometry_path,x,y,width,height,device)
x0 = get_velocity_filed(initial_index,x,y,width,height,device)
latent_x0 = compression_model.encode(x0)/10
pipe = SCALEDUrbanFlowPipeline(inference_model,val_noise_scheduler)

with torch.no_grad():
    for step in tqdm(range(100)):
        x_i = get_velocity_filed(initial_index+(step+1)*50,x,y,width,height,device)
        bc = pack_boundary_condition(x_i,geometry)
        latent_xbc = compression_model.encode(bc)/10
        input = torch.cat([latent_x0,latent_xbc],dim=1)
        output = pipe(
                input,
                num_inference_steps=25,
                guidance_scale=0,
                generator=torch.manual_seed(12580),
                return_dict=False,)
        latent_x0 = output.clone()
        primitive_x0 = compression_model.decode(latent_x0*10)
        visualize(primitive_x0[0,1,3,:, :].cpu().numpy(),step=step)