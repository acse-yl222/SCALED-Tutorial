from scaled.model.unets.unet_3ds import UNet3DsModel
from diffusers import DDIMScheduler
from scaled.dataset.urban_flow_dataset import UrbanFlowDataset
import torch
from scaled.validation.domain_decomposition import DomainDecompositionMultipleGPUs
from scaled.validation.domain_decomposition import predict_model

def main():
    subdomain_size = (64,128,128)
    halo_size = 8
    weight_path = 'weight/denoising_unet-248000.pth'
    save_dir = 'result/scaled_urbanflow_ga'
    dataset_dir = ''
    num_GPUs = 1
    
    model  = UNet3DsModel(
        in_channels=9,
        out_channels=3,
        down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D"),
        block_out_channels=(64, 128, 192, 256),
        add_attention=False,
        )
    
    model.load_state_dict(torch.load(weight_path,map_location='cpu'))
    
    val_noise_scheduler = DDIMScheduler(
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

    model = predict_model(model,
                          val_noise_scheduler,
                          subdomain_size,
                          num_inference_steps=10)
    
    initial_velocity = torch.zeros((1,3,subdomain_size[0],subdomain_size[1],subdomain_size[2]))
    
    for i in range(100):
        bc_velocity = torch.ones(1,3,subdomain_size[0],subdomain_size[1],subdomain_size[2])
        bc_velocity[:,:,32:-32,32:-32] = 0
        bc_velocity[:,:,:,:4] = -1
        bc_velocity = bc_velocity.bool()
        bc_velocity = bc_velocity[:,]
        input = torch.cat([initial_velocity,bc_velocity],dim=1)
        result = model(input)