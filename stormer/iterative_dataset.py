import os
import numpy as np
import torch
import h5py

from torch.utils.data import Dataset
from glob import glob

class ERA5OneStepRandomizedDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variables,
        transform,
        dict_diff_transform,
        list_intervals=[6, 12, 24],
        data_freq=6, # 1-hourly or 3-hourly or 6-hourly data
        year_list=None,
        region_info=None,
        flip_data=False,
    ):
        super().__init__()
        
        for l in list_intervals:
            assert l % data_freq == 0
        
        self.root_dir = root_dir
        self.variables = variables
        self.transform = transform
        self.dict_diff_transform = dict_diff_transform
        self.list_intervals = list_intervals
        self.data_freq = data_freq
        self.year_list = year_list
        self.region_info = region_info
        self.flip_data = flip_data
        if year_list is not None:
            self.year_idx_map = {year: i for i, year in enumerate(year_list)}
       
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        self.file_paths = sorted(file_paths)
        file_paths = []
        if year_list is not None:
            year_str = [str(year) for year in self.year_list]
            for file in self.file_paths:
                for year in year_str:
                    if year in file:
                        print('file added',year,file)
                        file_paths.append(file)
            print('self.file_paths')
            self.file_paths = file_paths

        
    def __len__(self):
        return len(self.file_paths) - max(self.list_intervals) // self.data_freq
    
    def get_out_path(self, year, inp_file_idx, steps):
        out_file_idx = inp_file_idx + steps
        out_path = os.path.join(
            self.root_dir,
            f'{year}_{out_file_idx:04}.h5'
        )
        if not os.path.exists(out_path):
            for i in range(steps):
                out_file_idx = inp_file_idx + i
                out_path = os.path.join(
                    self.root_dir,
                    f'{year}_{out_file_idx:04}.h5'
                )
                if os.path.exists(out_path):
                    max_step_forward = i
            remaining_steps = steps - max_step_forward
            if self.year_list is None:
                next_year = year + 1
            else:
                next_year = self.year_list[self.year_idx_map[year] + 1]
            out_path = os.path.join(
                self.root_dir,
                f'{next_year}_{remaining_steps-1:04}.h5'
            )
        return out_path
    
    def get_data_given_path(self, path):
        with h5py.File(path, 'r') as f:
            data = {
                main_key: {
                    sub_key: np.array(value) for sub_key, value in group.items()
            } for main_key, group in f.items()}
        
        if self.flip_data:
            for main_key, group in data.items():
                for sub_key, value in group.items():
                    if sub_key != 'time':
                        data[main_key][sub_key] = np.flip(value, axis=0)
                    else:
                        data[main_key][sub_key] = value
        
        return data
    
    def __getitem__(self, index):
        path = self.file_paths[index]
        interval = np.random.choice(self.list_intervals)
        
        steps = interval // self.data_freq
        year, inp_file_idx = os.path.basename(path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)
        out_path = self.get_out_path(year, inp_file_idx, steps)
        inp_data = self.get_data_given_path(path)
        out_data = self.get_data_given_path(out_path)
        
        # inp = [inp_data['input'][v] for v in self.variables]
        # inp = np.stack(inp, axis=0)
        inp = []
        for v in self.variables:
            if inp_data['input'][v].shape[0] < inp_data['input'][v].shape[1]:
                inp.append(inp_data['input'][v])
            else: # transpose if long before lat
                inp.append(inp_data['input'][v].T)

            #SSTs have NaNs we are filling them with 270 (basically land sea mask)
            if v == 'sea_surface_temperature':
                inp = np.nan_to_num(inp,nan=270.0)

        inp = np.stack(inp, axis=0)
        
        # out = [out_data['input'][v] for v in self.variables]
        # out = np.stack(out, axis=0)
        out = []
        for v in self.variables:
            if out_data['input'][v].shape[0] < out_data['input'][v].shape[1]:
                out.append(out_data['input'][v])
            else: # transpose if long before lat
                out.append(out_data['input'][v].T)

            #SSTs have NaNs we are filling them with 270 (basically land sea mask)
            if v == 'sea_surface_temperature':
                out = np.nan_to_num(out,nan=270.0)

        out = np.stack(out, axis=0)
    
        diff = out - inp
        inp = torch.from_numpy(inp)
        diff = torch.from_numpy(diff)
        
        interval_tensor = torch.Tensor([interval]) / 10.0
        
        if self.region_info is None:
            return (
                self.transform(inp).unsqueeze(0), # normalized
                inp.unsqueeze(0), # raw
                self.dict_diff_transform[interval](diff), # normalized
                interval_tensor,
                self.variables,
                self.variables
            )
        else:
            return (
                self.transform(inp).unsqueeze(0), # normalized
                inp.unsqueeze(0), # raw
                self.dict_diff_transform[interval](diff), # normalized
                interval_tensor,
                self.variables,
                self.variables,
                self.region_info
            )


class ERA5MultiStepRandomizedDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variables,
        steps,
        transform,
        dict_diff_transform,
        possible_intervals=[6, 12, 24],
        homogeneous_only=False,
        data_freq=6,
        year_list=None,
        region_info=None,
        flip_data=False
    ):
        super().__init__()
        
        for l in possible_intervals:
            assert l % data_freq == 0

        self.root_dir = root_dir
        self.variables = variables
        self.transform = transform
        self.dict_diff_transform = dict_diff_transform
        self.steps = steps
        self.possible_intervals = possible_intervals
        self.homogeneous_only = homogeneous_only
        self.data_freq = data_freq
        self.year_list = year_list
        if year_list is not None:
            self.year_idx_map = {year: i for i, year in enumerate(year_list)}

        self.region_info = region_info
        self.flip_data = flip_data
        
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        # file_paths = sorted(file_paths, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        file_paths = sorted(file_paths)
        self.inp_file_paths = file_paths[:-(steps * max(possible_intervals) // data_freq)] # the last few points do not have ground-truth
        self.file_paths = file_paths
        
    def __len__(self):
        return len(self.inp_file_paths)
    
    def get_data_given_path(self, path):
        with h5py.File(path, 'r') as f:
            data = {
                main_key: {
                    sub_key: np.array(value) for sub_key, value in group.items()
            } for main_key, group in f.items()}
        
        if self.flip_data:
            for main_key, group in data.items():
                for sub_key, value in group.items():
                    if sub_key != 'time':
                        data[main_key][sub_key] = np.flip(value, axis=0)
                    else:
                        data[main_key][sub_key] = value    
        
        x = []
        for v in self.variables:
            if data['input'][v].shape[0] < data['input'][v].shape[1]:
                x.append(data['input'][v])
            else: # transpose if long before lat
                x.append(data['input'][v].T)

            #SSTs have NaNs we are filling them with 270 (basically land sea mask)
            if v == 'sea_surface_temperature':
                x = np.nan_to_num(x,nan=270.0)

        x = np.stack(x, axis=0)
        return x
    
    def get_out_path(self, year, inp_file_idx, steps):
        out_file_idx = inp_file_idx + steps
        out_path = os.path.join(
            self.root_dir,
            f'{year}_{out_file_idx:04}.h5'
        )
        if not os.path.exists(out_path):
            for i in range(steps):
                out_file_idx = inp_file_idx + i
                out_path = os.path.join(
                    self.root_dir,
                    f'{year}_{out_file_idx:04}.h5'
                )
                if os.path.exists(out_path):
                    max_step_forward = i
            remaining_steps = steps - max_step_forward
            if self.year_list is None:
                next_year = year + 1
            else:
                next_year = self.year_list[self.year_idx_map[year] + 1]
            out_path = os.path.join(
                self.root_dir,
                f'{next_year}_{remaining_steps-1:04}.h5'
            )
        return out_path
    
    def __getitem__(self, index):
        inp_path = self.inp_file_paths[index]
        inp_data = self.get_data_given_path(inp_path)
        year, inp_file_idx = os.path.basename(inp_path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)
        
        diffs = []
        mean_diff_transform = []
        std_diff_transform = []
        outs = [inp_data]
        
        if self.homogeneous_only:
            list_intervals = [np.random.choice(self.possible_intervals)] * self.steps
            list_intervals = np.array(list_intervals)
        else:
            list_intervals = np.random.choice(self.possible_intervals, self.steps, replace=True)
        
        last_step_jump = 0
        
        # get ground-truth paths at multiple lead times
        for step in range(1, self.steps + 1):
            interval = list_intervals[step-1]
            out_path = self.get_out_path(year, inp_file_idx, interval // self.data_freq + last_step_jump)
            last_step_jump += interval // self.data_freq
            out = self.get_data_given_path(out_path)
            diff = out - outs[-1]
            diff = torch.from_numpy(diff)
            diffs.append(self.dict_diff_transform[interval](diff))
            outs.append(out)
            mean_diff_transform.append(self.dict_diff_transform[interval].mean)
            std_diff_transform.append(self.dict_diff_transform[interval].std)
        
        diffs = torch.stack(diffs, dim=0)
        inp_data = torch.from_numpy(inp_data)
        mean_diff_transform = torch.from_numpy(np.stack(mean_diff_transform, axis=0)).to(dtype=inp_data.dtype)
        std_diff_transform = torch.from_numpy(np.stack(std_diff_transform, axis=0)).to(dtype=inp_data.dtype)
        list_intervals = torch.from_numpy(list_intervals).to(dtype=inp_data.dtype) / 10.0
        
        if self.region_info is None:
            return (
                self.transform(inp_data).unsqueeze(0), # normalized
                inp_data.unsqueeze(0), # raw
                diffs, # normalized
                mean_diff_transform,
                std_diff_transform,
                list_intervals,
                self.variables,
                self.variables
            )
        else:
            return (
                self.transform(inp_data).unsqueeze(0), # normalized
                inp_data.unsqueeze(0), # raw
                diffs, # normalized
                mean_diff_transform,
                std_diff_transform,
                list_intervals,
                self.variables,
                self.variables,
                self.region_info
            )

# validation and test datasets consist of 1 input and multiple desired outputs at multiple lead times
class ERA5MultiLeadtimeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variables,
        list_lead_times,
        transform,
        data_freq=6,
        year_list=None,
        return_metadata=False,
        region_info=None,
        flip_data=False
    ):
        super().__init__()

        self.root_dir = root_dir
        self.variables = variables
        self.transform = transform
        self.list_lead_times = list_lead_times
        self.data_freq = data_freq
        self.year_list = year_list
        self.return_metadata = return_metadata
        self.region_info = region_info
        self.flip_data = flip_data
        
        #file_paths = glob(os.path.join(root_dir, '*.h5'))
        file_paths = glob(os.path.join(root_dir, '{}*.h5'.format(self.year_list[0])))
        # file_paths = sorted(file_paths, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        file_paths = sorted(file_paths)
        max_lead_time = max(*list_lead_times) if len(list_lead_times) > 1 else list_lead_times[0]
        max_steps = max_lead_time // data_freq
        self.inp_file_paths = file_paths[:-max_steps] # the last few points do not have ground-truth
        self.file_paths = file_paths
        
    def __len__(self):
        return len(self.inp_file_paths)
    
    def get_data_given_path(self, path):
        with h5py.File(path, 'r') as f:
            data = {
                main_key: {
                    sub_key: np.array(value) for sub_key, value in group.items()
            } for main_key, group in f.items()}
        
        if self.flip_data:
            for main_key, group in data.items():
                for sub_key, value in group.items():
                    if sub_key != 'time':
                        data[main_key][sub_key] = np.flip(value, axis=0)
                    else:
                        data[main_key][sub_key] = value
            
        x = []
        for v in self.variables:
            if data['input'][v].shape[0] < data['input'][v].shape[1]:
                x.append(data['input'][v])
            else: # transpose if long before lat
                x.append(data['input'][v].T)

            #SSTs have NaNs we are filling them with 270 (basically land sea mask)
            if v == 'sea_surface_temperature':
                x = np.nan_to_num(x,nan=270.0)

        x = np.stack(x, axis=0)
        return x, data['input']['time']
    
    def get_out_path(self, year, inp_file_idx, lead_time):
        steps = lead_time // self.data_freq
        out_file_idx = inp_file_idx + steps
        out_path = os.path.join(
            self.root_dir,
            f'{year}_{out_file_idx:04}.h5'
        )
        if not os.path.exists(out_path):
            for i in range(steps):
                out_file_idx = inp_file_idx + i
                out_path = os.path.join(
                    self.root_dir,
                    f'{year}_{out_file_idx:04}.h5'
                )
                if os.path.exists(out_path):
                    max_step_forward = i
            remaining_steps = steps - max_step_forward
            if self.year_list is None:
                next_year = year + 1
            else:
                next_year = self.year_list[self.year_idx_map[year] + 1]
            out_path = os.path.join(
                self.root_dir,
                f'{next_year}_{remaining_steps-1:04}.h5'
            )
        return out_path
    
    def __getitem__(self, index):
        inp_path = self.inp_file_paths[index]
        print('inp_path :',inp_path)
        inp_data, inp_time = self.get_data_given_path(inp_path)
        year, inp_file_idx = os.path.basename(inp_path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)
        dict_out = {}
        dict_out_time = {}
        
        # get ground-truth paths at multiple lead times
        for lead_time in self.list_lead_times:
            out_path = self.get_out_path(year, inp_file_idx, lead_time)
            dict_out[lead_time], dict_out_time[lead_time] = self.get_data_given_path(out_path)
            
        inp_data = torch.from_numpy(inp_data)
        dict_out = {lead_time: torch.from_numpy(out) for lead_time, out in dict_out.items()}
        
        if not self.return_metadata:
            if self.region_info is None:
                return (
                    self.transform(inp_data).unsqueeze(0), # normalized
                    inp_data.unsqueeze(0), # raw
                    {lead_time: self.transform(out) for lead_time, out in dict_out.items()}, # normalized
                    {lead_time: out for lead_time, out in dict_out.items()}, # raw
                    self.variables,
                    self.variables
                )
            else:
                return (
                    self.transform(inp_data).unsqueeze(0), # normalized
                    inp_data.unsqueeze(0), # raw
                    {lead_time: self.transform(out) for lead_time, out in dict_out.items()}, # normalized
                    {lead_time: out for lead_time, out in dict_out.items()}, # raw
                    self.variables,
                    self.variables,
                    self.region_info
                )
        else:
            if self.region_info is None:
                return (
                    self.transform(inp_data).unsqueeze(0), # normalized
                    inp_data.unsqueeze(0), # raw
                    inp_time,
                    {lead_time: self.transform(out) for lead_time, out in dict_out.items()}, # normalized
                    {lead_time: out for lead_time, out in dict_out.items()}, # raw
                    dict_out_time,
                    self.variables,
                    self.variables
                )
            else:
                return (
                    self.transform(inp_data).unsqueeze(0), # normalized
                    inp_data.unsqueeze(0), # raw
                    inp_time,
                    {lead_time: self.transform(out) for lead_time, out in dict_out.items()}, # normalized
                    {lead_time: out for lead_time, out in dict_out.items()}, # raw
                    dict_out_time,
                    self.variables,
                    self.variables,
                    self.region_info
                )
            

class ERA5MultiLeadtimeForecastOnlyDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variables,
        list_lead_times,
        transform,
        data_freq=6,
        year_list=None,
        return_metadata=False,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.variables = variables
        self.transform = transform
        self.list_lead_times = list_lead_times
        self.data_freq = data_freq
        self.year_list = year_list
        self.return_metadata = return_metadata
        
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        file_paths = sorted(file_paths)
        self.inp_file_paths = file_paths # don't have to retrieve ground-truth
        self.file_paths = file_paths
        
    def __len__(self):
        return len(self.inp_file_paths)
    
    def get_data_given_path(self, path):
        with h5py.File(path, 'r') as f:
            data = {
                main_key: {
                    sub_key: np.array(value) for sub_key, value in group.items()
            } for main_key, group in f.items()}
        x = []
        for v in self.variables:
            if data['input'][v].shape[0] < data['input'][v].shape[1]:
                x.append(data['input'][v])
            else: # transpose if long before lat
                x.append(data['input'][v].T)

            #SSTs have NaNs we are filling them with 270 (basically land sea mask)
            if v == 'sea_surface_temperature':
                x = np.nan_to_num(x,nan=270.0)


        x = np.stack(x, axis=0)
        return x, data['input']['time']
    
    def __getitem__(self, index):
        inp_path = self.inp_file_paths[index]
        inp_data, inp_time = self.get_data_given_path(inp_path)
        year, inp_file_idx = os.path.basename(inp_path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)
            
        inp_data = torch.from_numpy(inp_data)
        
        if not self.return_metadata:
            return (
                self.transform(inp_data).unsqueeze(0), # normalized
                inp_data.unsqueeze(0), # raw
                None,
                None,
                self.variables,
                self.variables
            )
        else:
            return (
                self.transform(inp_data).unsqueeze(0), # normalized
                inp_data.unsqueeze(0), # raw
                inp_time,
                None,
                None,
                None,
                self.variables,
                self.variables
            )
