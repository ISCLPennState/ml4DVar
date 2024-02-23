import os, sys

import torch
import h5py
import numpy as np

#from stormer.models.hub.vit_adaln import ViTAdaLN
from stormer.data.iterative_dataset import ERA5MultiLeadtimeDataset
from climax_utils import ClimaXWrapper

sys.path.append("/eagle/MDClimSim/mjp5595/ClimaX-v2/src/climax")
from arch import ClimaX

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

    save_dir_name = 'climax_val_forecasts'

    save_dir = '/eagle/MDClimSim/mjp5595/data/climax/{}/'.format(save_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    means = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_mean.npz')
    stds = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_std.npz')

    background_file_np = '/eagle/MDClimSim/mjp5595/ml4dvar/background_starter.npy' # This is just to initialize the model background

    from varsClimaX import varsClimax
    vars_climax = varsClimax().vars_climax
    DEF_VARIABLES = varsClimax().DEF_VARIABLES

    pretrained_path = "/eagle/MDClimSim/tungnd/ClimaX/exps/global_forecast_climax/dinov2_vitl14_iterative_predict_diff_4_steps/checkpoints/epoch_018.ckpt"
    nn_model = ClimaX(
        default_vars=DEF_VARIABLES, 
        img_size=[128, 256], 
        patch_size=4, 
        backbone='dinov2_vitl14', 
        vision_pretrained=False, 
        decoder_depth=2, 
        parallel_patch_embed=True)
    #print('nn_model.state_dict :',nn_model.state_dict())
    net_state_dict = nn_model.state_dict()
    print("Loading pre-trained checkpoint from: %s" % pretrained_path)
    checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
    print('checkpoint.keys() :',checkpoint.keys())
    checkpoint_model = checkpoint["state_dict"]
    if "net.token_embeds.proj_weights" not in checkpoint_model.keys():
        raise ValueError(
            "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
        )
    # checkpoint_keys = list(checkpoint_model.keys())
    for k in list(checkpoint_model.keys()):
        if "channel" in k:
            checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
            del checkpoint_model[k]
    for k in list(checkpoint_model.keys()):
        if 'net.' in k:
            print(f"changing key {k} to {k.replace('net.','')} from pretrained checkpoint")
            #print(" - Shape of model", checkpoint_model[k].shape)
            checkpoint_model[k.replace('net.','')] = checkpoint_model[k]
            del checkpoint_model[k]
    for k in list(checkpoint_model.keys()):
        if k not in net_state_dict.keys() or checkpoint_model[k].shape != net_state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            print(" - Shape of model", checkpoint_model[k].shape)
            print(" - Shape of entry", net_state_dict[k].shape)
            del checkpoint_model[k]
    nn_model.load_state_dict(checkpoint_model, strict=False)

    nn_model.to(device)
    nn_model.eval()

    climax_wrapper = ClimaXWrapper(
        root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/',
        variables=vars_climax,
        net=nn_model,
        base_lead_time=6,
        device=device,
    )

    ########################################################################################################################
    def run_forecasts(x,
                    x_raw,
                    idx,
                    forecast_steps=2,
                    save_dir=save_dir,
                    climax_wrapper=climax_wrapper,
                    vars_climax=vars_climax,
                    device=device,
                    ):

        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        norm_preds,raw_preds,_ = climax_wrapper.forward_multi_step(
            x.to(device),
            vars_climax,
            steps=forecast_steps)
        
        #print('len(norm_preds), norm_preds[1].shape:',len(norm_preds),norm_preds[1].shape)

        hf_norm = h5py.File(os.path.join(save_dir, 'norm_{:0>4d}.h5'.format(idx)),'w')
        hf_raw = h5py.File(os.path.join(save_dir, '{:0>4d}.h5'.format(idx)),'w')
        hf_norm.create_dataset(str(0), data=x[0,:,:,:])
        hf_raw.create_dataset(str(0), data=x_raw[0,:,:,:])
        for i in range(len(norm_preds)):
            data_raw = raw_preds[i].detach().cpu().numpy()
            data_norm = norm_preds[i].detach().cpu().numpy()
            # data.shape : (1,vars,lat,lon) -> (1,63,128,256)
            hf_norm.create_dataset(str((i+1)*climax_wrapper.base_lead_time), data=data_norm[0,:,:,:])
            hf_raw.create_dataset(str((i+1)*climax_wrapper.base_lead_time), data=data_raw[0,:,:,:])
        return
    ########################################################################################################################

    data_test = ERA5MultiLeadtimeDataset(
        root_dir=os.path.join('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/','test'),
        variables=vars_climax,
        list_lead_times=[6],
        transform=climax_wrapper.inp_transform,
        data_freq=6,
        year_list=[2017]
    )

    for idx, (input_norm, input_raw, _, _, _, _) in enumerate(data_test):
        #print('input_norm shape/min/mean/max :',input_norm.shape,torch.min(input_norm),torch.mean(input_norm),torch.max(input_norm))
        #print('input_raw shape,min/mean/max :',input_raw.shape,torch.min(input_raw),torch.mean(input_raw),torch.max(input_raw))
        #print('idx, device_set, gpu_num, device_count, mod :',idx,device_set,gpu_num,torch.cuda.device_count(),idx%torch.cuda.device_count())

        if (device_set == True):
            if (int(idx % torch.cuda.device_count()) != int(gpu_num)):
                continue

        file_to_write = os.path.join(save_dir, '{:0>4d}.h5'.format(idx))
        if os.path.exists(file_to_write):
            continue

        print('running forecast {}/{}'.format(idx+1,len(data_test)))
        with torch.inference_mode():
            run_forecasts(input_norm,
                        input_raw,
                        idx,
                        forecast_steps=40,
                        save_dir=save_dir,
                        climax_wrapper=climax_wrapper,
                        vars_climax=vars_climax,
                        device=device,
                        )