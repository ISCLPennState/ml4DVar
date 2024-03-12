import os, sys

import torch
import h5py
import numpy as np

from stormer.models.hub.vit_adaln import ViTAdaLN
#from stormer.data.iterative_dataset import ERA5MultiLeadtimeDataset
from iterative_dataset import ERA5MultiLeadtimeDataset
from stormer_utils import StormerWrapper


gpu_num = 0 
device_set = False
if len(sys.argv) > 1:
    gpu_num = sys.argv[1]
    device_set = True
device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")
#device = "cpu"
print('USING DEVICE :',device)

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    save_dir_name = 'stormer_forecasts_2020_val'

    save_dir = '/eagle/MDClimSim/mjp5595/data/stormer/{}/'.format(save_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    from varsStormer import varsStormer
    vars_stormer = varsStormer().vars_stormer

    ckpt_pth = '/eagle/MDClimSim/tungnd/stormer/models/6_12_24_climax_large_2_True_delta_8/checkpoints/epoch_015.ckpt'
    net = ViTAdaLN(
        in_img_size=(128, 256),
        list_variables=vars_stormer,
        patch_size=2,
        embed_norm=True,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
    )
    net.to(device)
    net.eval()
    stormer_wrapper = StormerWrapper(
        root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/',
        variables=vars_stormer,
        net=net,
        list_lead_time=[6],
        ckpt=ckpt_pth,
        device=device,
    )

    ########################################################################################################################
    def run_forecasts(x,
                    x_raw,
                    idx,
                    forecast_steps=2,
                    save_dir=save_dir,
                    stormer_wrapper=stormer_wrapper,
                    vars_stormer=vars_stormer,
                    device=device,
                    ):

        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        norm_preds,raw_preds = stormer_wrapper.eval_multi_step(
            x.to(device),
            vars_stormer,
            steps=forecast_steps)
        
        #print('len(norm_preds), norm_preds[1][0].shape:',len(norm_preds),norm_preds[1].shape)

        hf_norm = h5py.File(os.path.join(save_dir, 'norm_{:0>4d}.h5'.format(idx)),'w')
        hf_raw = h5py.File(os.path.join(save_dir, 'raw_{:0>4d}.h5'.format(idx)),'w')
        hf_norm.create_dataset(str(0), data=x[0,:,:,:])
        hf_raw.create_dataset(str(0), data=x_raw[0,:,:,:])
        for i in range(len(norm_preds)):
            data_norm = norm_preds[i].detach().cpu().numpy()
            data_raw = raw_preds[i].detach().cpu().numpy()
            # data.shape : (1,vars,lat,lon) -> (1,63,128,256)
            hf_norm.create_dataset(str((i+1)*stormer_wrapper.list_lead_time[0]), data=data_norm[0,:,:,:])
            hf_raw.create_dataset(str((i+1)*stormer_wrapper.list_lead_time[0]), data=data_raw[0,:,:,:])
        return
    ########################################################################################################################

    data_test = ERA5MultiLeadtimeDataset(
        #root_dir=os.path.join('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/','train'),
        root_dir=os.path.join('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/','test'),
        variables=vars_stormer,
        list_lead_times=[6],
        transform=stormer_wrapper.inp_transform,
        data_freq=6,
        year_list=[2020]
        #year_list=[2017]
        #year_list=[2014]
        #year_list=[1979]
    )

    for idx, (input_norm, input_raw, _, _, _, _) in enumerate(data_test):
        #print('input_norm shape/min/mean/max :',input_norm.shape,torch.min(input_norm),torch.mean(input_norm),torch.max(input_norm))
        #print('input_raw shape,min/mean/max :',input_raw.shape,torch.min(input_raw),torch.mean(input_raw),torch.max(input_raw))
        #print('idx, device_set, gpu_num, device_count, mod :',idx,device_set,gpu_num,torch.cuda.device_count(),idx%torch.cuda.device_count())

        # Generate init_background for DA
        if idx < 2:
            continue
        if idx == 2:
            np.save('background_init_stormer_norm_hr12',input_norm)
            np.save('background_init_stormer_raw_hr12',input_raw)
            break

        # Run forecast from init background for comparison w/ analysis
        #input_norm = torch.from_numpy(np.load('/eagle/MDClimSim/mjp5595/ml4dvar/stormer/background_init_stormer_norm.npy'))
        #input_raw = np.load('/eagle/MDClimSim/mjp5595/ml4dvar/stormer/background_init_stormer_raw.npy')

        if (device_set == True):
            if (int(idx % torch.cuda.device_count()) != int(gpu_num)):
                continue

        file_to_write = os.path.join(save_dir, 'norm_{:0>4d}.h5'.format(idx))
        if os.path.exists(file_to_write):
            continue

        print('running forecast {}/{}'.format(idx+1,len(data_test)))
        with torch.inference_mode():
            run_forecasts(input_norm,
                        input_raw,
                        idx,
                        forecast_steps=40,
                        save_dir=save_dir,
                        stormer_wrapper=stormer_wrapper,
                        vars_stormer=vars_stormer,
                        device=device,
                        )