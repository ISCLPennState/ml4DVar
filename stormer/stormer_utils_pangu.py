import torch
import numpy as np
import sys
import os
from torchvision.transforms import transforms

CONSTANTS = [
    "angle_of_sub_gridscale_orography",
    "geopotential_at_surface",
    "high_vegetation_cover",
    "lake_cover",
    "lake_depth",
    "land_sea_mask",
    "low_vegetation_cover",
    "slope_of_sub_gridscale_orography",
    "soil_type",
    "standard_deviation_of_filtered_subgrid_orography",
    "standard_deviation_of_orography",
    "type_of_high_vegetation",
    "type_of_low_vegetation",
    "orography",
    "lattitude",
]

class StormerWrapperPangu:
    def __init__(self,
                 root_dir,
                 variables,
                 net,
                 base_lead_time=6,
                 possible_lead_times=[24,12,6],
                 ckpt=None,
                 device=None,
                 logger=None,
                 ):

        #####################################################################################################
        #####################################################################################################
        normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
        normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
        self.inp_transform = transforms.Normalize(normalize_mean, normalize_std)
        
        self.diff_transforms = {}
        #for l in list_train_lead_time:
        for l in possible_lead_times:
            normalize_diff_std = dict(np.load(os.path.join(root_dir, f"normalize_diff_std_{l}.npz")))
            normalize_diff_std = np.concatenate([normalize_diff_std[v] for v in variables], axis=0)
            self.diff_transforms[l] = transforms.Normalize(np.zeros_like(normalize_diff_std), normalize_diff_std)
        self.diff_transforms = self.diff_transforms

        self.reverse_inp_transform = self.get_reverse_transform(self.inp_transform)
        self.diff_transforms = self.diff_transforms
        
        self.reverse_diff_transform = {}
        for k, v in self.diff_transforms.items():
            self.reverse_diff_transform[k] = self.get_reverse_transform(v)
        self.reverse_diff_transform = self.reverse_diff_transform
        #####################################################################################################
        #####################################################################################################

        self.base_lead_time = base_lead_time
        for i,lt in enumerate(possible_lead_times[1:]):
            assert possible_lead_times[i+1] < possible_lead_times[i]
        self.possible_lead_times = possible_lead_times

        self.variables = variables

        self.logger = logger
        self.device = device
        self.net = net
        if ckpt is not None:
            self.load_model(ckpt)
        print('checkpoint model loaded')
        self.net.to(self.device)
        #self.freeze_model()

    def get_reverse_transform(self, transform):
        mean, std = transform.mean, transform.std
        std_reverse = 1 / std
        mean_reverse = -mean * std_reverse
        return transforms.Normalize(mean_reverse, std_reverse)

    def load_model(self, pretrained_path):

        net_state_dict = self.net.state_dict()
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        checkpoint_model = checkpoint["state_dict"]

        ############################################################################################
        ############################################################################################
        key_dict = {}
        for k in list(checkpoint_model.keys()):
            key_dict[k.replace('net.','')] = 'ckpt'
        for k in list(net_state_dict.keys()):
            if k not in key_dict:
                key_dict[k] = 'ViTAdaLN'
            else:
                del(key_dict[k])
        if len(key_dict) > 0:
            print('Mismatched Keys')
            for key in key_dict:
                print('{} : {}'.format(key_dict[key],key))
        else:
            print('All checkpoint keys match!')
        ############################################################################################
        ############################################################################################

        for k in list(checkpoint_model.keys()):
            if 'net.' in k:
                #print(f"changing key {k} to {k.replace('net.','')} from pretrained checkpoint")
                checkpoint_model[k.replace('net.','')] = checkpoint_model[k]
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if k not in net_state_dict.keys():
                print(f"KEYERROR Removing key {k} from pretrained checkpoint")
                #del checkpoint_model[k]
            elif checkpoint_model[k].shape != net_state_dict[k].shape:
                print(f"SHAPEERROR Removing key {k} from pretrained checkpoint")
                print(" - Shape of ckpt", checkpoint_model[k].shape)
                print(" - Shape of Arch", net_state_dict[k].shape)

        self.net.load_state_dict(checkpoint_model, strict=False)

    def freeze_model(self, ):
        print('freezing model')
        self.net.requires_grad = False

    def replace_constant(self, yhat, out_variables):
        for i in range(yhat.shape[1]):
            # if constant replace with 0.0
            if out_variables[i] in CONSTANTS:
                yhat[:, i] = 0.0
        return yhat

    def forward_given_lead_time(self, x: torch.Tensor, lead_time):
        # x is in the normalized input space
        
        norm_preds = []
        raw_preds = []
        norm_diff = []

        lead_time_tensor = torch.Tensor([lead_time]).to(device=x.device, dtype=x.dtype) / 10.0
        pred_diff = self.net(x, self.variables, lead_time_tensor) # diff in the normalized space
        pred_diff = self.replace_constant(pred_diff, self.variables)
        norm_diff = pred_diff
        pred_diff = self.reverse_diff_transform[lead_time](pred_diff) # diff in the original space
        pred = self.reverse_inp_transform(x.squeeze(1)) + pred_diff # prediction in the original space
        raw_preds = pred
        x = self.inp_transform(pred) # prediction in the normalized space
        norm_preds = x

        return norm_preds, raw_preds, norm_diff

    def eval_to_forecast_time_with_lead_time(self, x: torch.Tensor, forecast_time, lead_time=None, print_steps=True):
        # x is in the normalized input space
        norm_pred_tot = []
        raw_pred_tot = []

        lead_time_combos = self.get_forecast_time_combinations(forecast_time,lead_time)

        if print_steps:
            print('\tlead_time_combos :',lead_time_combos)
            if self.logger:
                self.logger.info('\tlead_time_combos : {}'.format(lead_time_combos))

        norm_pred = x
        for step,lt in enumerate(lead_time_combos):
            if print_steps:
                print('\teval model step {}/{} w/ lead time {}'.format(step+1,len(lead_time_combos),lt))
                if self.logger:
                    self.logger.info('\teval model step {}/{} w/ lead time {}'.format(step+1,len(lead_time_combos),lt))
            norm_pred, raw_pred, _ = self.forward_given_lead_time(norm_pred, lt)
            raw_pred_tot.append(raw_pred[0,:,:,:])
            norm_pred_tot.append(norm_pred[0,:,:,:])
        
        # [(vars, lat, lon)]*steps
        return norm_pred_tot, raw_pred_tot, lead_time_combos

    def get_forecast_time_combinations(self, forecast_time, lead_time=None):
        forecast_time_combinations = []

        for plt in self.possible_lead_times:
            if lead_time:
                if plt != lead_time:
                    continue
            while forecast_time >= plt:
                forecast_time -= plt
                forecast_time_combinations.append(plt)
        assert forecast_time == 0
        
        return forecast_time_combinations