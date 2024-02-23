import torch
import numpy as np
import sys
import os
from torchvision.transforms import transforms

#sys.path.append("/eagle/MDClimSim/mjp5595/ClimaX-v2/src")
#from climax.utils.data_utils import CONSTANTS

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

class ClimaXWrapper:
    def __init__(self,
                 root_dir,
                 variables,
                 net,
                 base_lead_time=6,
                 device=None,
                 ):

        normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
        normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
        self.transforms = transforms.Normalize(normalize_mean, normalize_std)

        #normalize_diff_std = dict(np.load(os.path.join(root_dir, "normalize_diff_std.npz")))
        normalize_diff_std = dict(np.load(os.path.join(root_dir, "normalize_diff_std_{}.npz".format(base_lead_time))))
        normalize_diff_std = np.concatenate([normalize_diff_std[v] for v in variables], axis=0)
        self.diff_transforms = transforms.Normalize(np.zeros_like(normalize_diff_std), normalize_diff_std)

        self.inp_transform = self.transforms
        self.reverse_inp_transform = self.get_reverse_transform(self.transforms)
        self.reverse_diff_transform = self.get_reverse_transform(self.diff_transforms)
        
        self.device = device
        self.base_lead_time = base_lead_time

        self.net = net
        #self.freeze_model()
        self.net.to(self.device)

    def freeze_model(self, ):
        print('freezing model')
        self.net.requires_grad = False

    def get_reverse_transform(self, transform):
        mean, std = transform.mean, transform.std
        std_reverse = 1 / std
        mean_reverse = -mean * std_reverse
        return transforms.Normalize(mean_reverse, std_reverse)

    def replace_constant(self, yhat, out_variables):
        for i in range(yhat.shape[1]):
            # if constant replace with 0.0
            if out_variables[i] in CONSTANTS:
                yhat[:, i] = 0.0
        return yhat

    def forward(self, x: torch.Tensor, in_variables) -> torch.Tensor:
        pred = self.net(x.to(self.device), in_variables)
        return self.replace_constant(pred, in_variables)

    def forward_multi_step(self, x: torch.Tensor, in_variables, steps):
        # x is in the normalized input space
        norm_preds = []
        raw_preds = []
        norm_diff = []
        #print("\tAbout to enter for loop w/ steps: ", steps)
        for j in range(steps):
            pred_diff = self.net(x, in_variables) # diff in the normalized space
            pred_diff = self.replace_constant(pred_diff, in_variables)
            norm_diff.append(pred_diff)
            pred_diff = self.reverse_diff_transform(pred_diff) # diff in the original space
            pred = self.reverse_inp_transform(x.squeeze(1)) + pred_diff # prediction in the original space
            raw_preds.append(pred)
            x = self.inp_transform(pred) # prediction in the normalized space
            norm_preds.append(x)
            #print('\t - pred_steps {}/{} done'.format(j,steps))
        return norm_preds, raw_preds, norm_diff