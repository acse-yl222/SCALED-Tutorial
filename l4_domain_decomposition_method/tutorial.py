import torch
import importlib.metadata
import sys
import os
import numpy as np
original_version = importlib.metadata.version
importlib.metadata.version = lambda name: "1.24.4" if name == "numpy" else original_version(name)
PROJECT_ROOT1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT1)
from scaled.model.unets.unet_3ds import UNet3DsModel
from scaled.runner.single_domain_inference import LatentInferenceOnlyGeometry
from scaled.model.autoencoders.autoencoder3dv1 import AutoencoderKL
import matplotlib.pyplot as plt
import torch.nn.functional as F

def trim_halo(x, hc):
    if hc > 0:
        return x[:, :, :, hc:-hc, hc:-hc]
    else:
        return x

class DataPreprocess:
    def __init__(self,compression_model,geometry,device):
        self.compression_model = compression_model.to(device)
        self.device = device
        self.geometry = geometry

    def make_data_bg(self):
        d,h,w = self.geometry.shape
        data_bg = torch.ones((1,3,d,h,w)).to(self.device)
        data_bg[:,:,geometry.bool()] = 0
        data_0 = torch.zeros((1,3,d,h,w)).to(self.device)
        return data_0,data_bg
    
    def compression(self, subdomain_size, halo_cell):
        d, h, w = self.geometry.shape
        latent_shape = (1, 4, d // 4, h // 4, w // 4)
        result_latent_data_0 = torch.zeros(latent_shape).to(self.device)
        result_latent_data_bg = torch.zeros(latent_shape).to(self.device)
        count_latent_0 = torch.zeros(latent_shape).to(self.device)
        count_latent_bg = torch.zeros(latent_shape).to(self.device)

        data_0, data_bg = self.make_data_bg()
        chunk_h = h // subdomain_size
        chunk_w = w // subdomain_size
        latent_domainsize = subdomain_size // 4
        latent_halo = halo_cell // 4

        with torch.no_grad():
            for i in range(chunk_h):
                for j in range(chunk_w):
                    i_start = i * (subdomain_size - 2 * halo_cell)
                    j_start = j * (subdomain_size - 2 * halo_cell)
                    subdomain_0 = data_0[:, :, :, i_start:i_start + subdomain_size, j_start:j_start + subdomain_size]
                    subdomain_bg = data_bg[:, :, :, i_start:i_start + subdomain_size, j_start:j_start + subdomain_size]
                    latent_subdomain_0 = self.compression_model.encode(subdomain_0)
                    latent_subdomain_bg = self.compression_model.encode(subdomain_bg)
                    i_latent_start = i * (latent_domainsize - 2 * latent_halo)
                    j_latent_start = j * (latent_domainsize - 2 * latent_halo)

                    result_latent_data_0[:, :, :, i_latent_start:i_latent_start + latent_domainsize,
                                            j_latent_start:j_latent_start + latent_domainsize] += latent_subdomain_0
                    result_latent_data_bg[:, :, :, i_latent_start:i_latent_start + latent_domainsize,
                                            j_latent_start:j_latent_start + latent_domainsize] += latent_subdomain_bg
                    count_latent_0[:, :, :, i_latent_start:i_latent_start + latent_domainsize,
                                        j_latent_start:j_latent_start + latent_domainsize] += 1
                    count_latent_bg[:, :, :, i_latent_start:i_latent_start + latent_domainsize,
                                        j_latent_start:j_latent_start + latent_domainsize] += 1

        # aviod 0
        count_latent_0[count_latent_0 == 0] = 1
        count_latent_bg[count_latent_bg == 0] = 1
        result_latent_data_0 /= count_latent_0
        result_latent_data_bg /= count_latent_bg
        return result_latent_data_0, result_latent_data_bg


    def visualize(self,subdomain,timestep):
        os.makedirs('results',exist_ok=True)
        result = subdomain[0,:,128]
        plt.imshow(result,vmin=-0.5,vmax=0.2)
        plt.colorbar()
        plt.savefig(f'result/{timestep:05d}.png')
        plt.close()

    def decode_single_visualize(self,latent,subdomain_size,halo_cell,t):
        latent_domainsize = subdomain_size//4
        latent_halo = halo_cell//4
        result  = torch.zeros((1,3,64,64,1024)).to(self.device)
        latent = F.pad(latent,(latent_halo,latent_halo,latent_halo,latent_halo,0,0))
        with torch.no_grad():
            for i in range(1):
                for j in range(8):
                    hs = i * latent_domainsize
                    ws = j * latent_domainsize
                    he = hs + latent_domainsize + 2 * latent_halo
                    we = ws + latent_domainsize + 2 * latent_halo
                    latent_sub = latent[:, :, :, hs:he, ws:we]
                    subdomain = self.compression_model.decode(latent_sub)  # -> (1,3,D,H,W)
                    subdomain_core = trim_halo(subdomain, halo_cell)
                    Hs = i * subdomain_size
                    Ws = j * subdomain_size
                    He = Hs + subdomain_size
                    We = Ws + subdomain_size
                    result[:, :, :, Hs:He, Ws:We] = subdomain_core
        np.save(f'result/{t:05d}.npy',result.cpu().numpy())
        # self.visualize(result.detach().cpu().numpy()[0], t)

    # def decode_save_single(self,data,subdomain_size,halo_cell,t):
    #     latent_domainsize = subdomain_size//4
    #     latent_halo = halo_cell//4
    #     result  = torch.zeros((1,3,64,256,1024)).to(self.device)
    #     latent = F.pad(latent,(latent_halo,latent_halo,latent_halo,latent_halo,0,0))
    #     with torch.no_grad():
    #         for i in range(1):
    #             for j in range(4):
    #                 hs = i * latent_domainsize
    #                 ws = j * latent_domainsize
    #                 he = hs + latent_domainsize + 2 * latent_halo
    #                 we = ws + latent_domainsize + 2 * latent_halo

    #                 latent_sub = latent[:, :, :, hs:he, ws:we]
    #                 subdomain = self.compression_model.decode(latent_sub)  # -> (1,3,D,H,W)

    #                 # 去掉解码后的 halo
    #                 subdomain_core = trim_halo(subdomain, halo_cell)

    #                 # 放回 result
    #                 Hs = i * subdomain_size
    #                 Ws = j * subdomain_size
    #                 He = Hs + subdomain_size
    #                 We = Ws + subdomain_size

    #                 # 断言尺寸匹配（调试期有用）
    #                 # assert subdomain_core.shape[-2:] == (subdomain_size, subdomain_size)
    #                 # assert subdomain_core.shape[2] == depth
    #                 result[:, :, :, Hs:He, Ws:We] = subdomain_core


    def decode_visualize(self,data_dir,subdomain_size,halo_cell):
        files = os.listdir(data_dir)
        files.sort()
        for t,filename in enumerate(files):
            file_path = os.path.join(data_dir,filename)
            data = torch.load(file_path).to(self.device)
            self.decode_single_visualize(data,subdomain_size,halo_cell,t)
            # print(t)
    # def decode(self,data_dir,subdomain_size,halo_cell):
    #     files = os.listdir(data_dir)
    #     files.sort()
    #     for t,filename in enumerate(files):
    #         file_path = os.path.join(data_dir,filename)
    #         data = torch.load(file_path).to(self.device)
    #         self.decode_single_visualize(data,subdomain_size,halo_cell,t)
    #         print(t)

if __name__ is '__main__':
    device = 'cuda'
    compression_weight_path = 'weight/trainning_autoencoder_ratio4_latentsize4/vae-193000.pth'
    inference_weight_path = 'weight/trainning_latent_scaled_ratio4_latentsize4_regression_v1/model-1003000.pth'
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
    compression_model.load_state_dict(torch.load(compression_weight_path,map_location='cpu'))
    inference_model = UNet3DsModel(
        in_channels=8,
        out_channels=4,
        down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D"),
        block_out_channels=(128, 256, 384, 512),
        add_attention=False)
    inference_model.load_state_dict(torch.load(inference_weight_path,map_location='cpu'))
    
    geometry = np.load('scaled_geometry_64x256x1024.npy')
    # 64x256x1024 -> 16x64x256
    # 64GB H100 64x128x128 40GB A100 
    data_process = DataPreprocess(compression_model,geometry,device)
    initial_information = data_process.compression(128,8)
    inference = LatentInferenceOnlyGeometry(inference_model,initial_information,device)
    inference.inference(100)
    data_process.decode_visualize('saved_latent',128,8)
    # data_process.decode('saved_latent',128,8)
