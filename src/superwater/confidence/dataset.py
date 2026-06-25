import math
import os
import pickle
import random
from argparse import Namespace

import numpy as np
import torch
import yaml
from torch_geometric.data import Dataset, Data


class ListDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

def get_args(original_model_dir):
    with open(f'{original_model_dir}/model_parameters.yml') as f:
        model_args = Namespace(**yaml.full_load(f))
    return model_args

def confidence_cache_path(cache_path, original_model_dir, split, limit_complexes):
    """Directory holding the sampled candidate cache for one score model + split.

    Keyed by the score-dir basename so candidate caches sampled from different score
    checkpoints (e.g. conf vs large) coexist under the same ``--cache_path``. The
    generator (``superwater.confidence.candidates``) and ``ConfidenceDataset`` MUST
    agree on this layout, so both call this helper."""
    return os.path.join(
        cache_path,
        f'model_{os.path.splitext(os.path.basename(original_model_dir))[0]}'
        f'_split_{split}_limit_{limit_complexes}')

class ConfidenceDataset(Dataset):
    def __init__(self, loader, cache_path, original_model_dir, split, device, limit_complexes,
                 inference_steps, samples_per_complex, all_atoms,
                 args, model_ckpt, balance=False, use_original_model_cache=True, mad_classification_cutoff=2,
                 cache_ids_to_combine=None, cache_creation_id=None, running_mode=None, water_ratio=15, resample_steps=1):

        super(ConfidenceDataset, self).__init__()
        self.loader = loader
        self.device = device
        self.inference_steps = inference_steps
        self.limit_complexes = limit_complexes
        self.all_atoms = all_atoms
        self.original_model_dir = original_model_dir
        self.balance = balance
        self.use_original_model_cache = use_original_model_cache
        self.mad_classification_cutoff = mad_classification_cutoff
        self.cache_ids_to_combine = cache_ids_to_combine
        self.cache_creation_id = cache_creation_id
        self.samples_per_complex = samples_per_complex
        self.model_ckpt = model_ckpt
        self.args = args

        self.running_mode = running_mode
        self.water_ratio = water_ratio
        self.resample_steps = resample_steps

        self.original_model_args = get_args(original_model_dir)

        # Load-only: the candidate cache (water positions + MADs) is sampled ahead of time by
        # superwater.confidence.candidates (via --generate_candidates_only). This dataset never
        # samples — that keeps N DDP ranks from generating concurrently. A missing cache raises
        # below with instructions to run candidate generation first.
        self.full_cache_path = confidence_cache_path(cache_path, original_model_dir, split, limit_complexes)
        print("cache path is ", self.full_cache_path)

        all_mads_unsorted, all_full_water_positions_unsorted, all_names_unsorted = [], [], []
        for idx, cache_id in enumerate(self.cache_ids_to_combine):
            print(f'HAPPENING | Loading positions and MADs from cache_id from the path: {os.path.join(self.full_cache_path, "water_positions_id"+ str(cache_id)+ ".pkl")}')
            if not os.path.exists(os.path.join(self.full_cache_path, f"water_positions_id{cache_id}.pkl")):
                raise FileNotFoundError(
                    f'Candidate cache missing: {os.path.join(self.full_cache_path, f"water_positions_id{cache_id}.pkl")}\n'
                    f'Generate it first (single-process or torchrun-sharded), e.g.:\n'
                    f'  python -m superwater.confidence.train --generate_candidates_only '
                    f'--original_model_dir {original_model_dir} --cache_creation_id {cache_id} ...')
            with open(os.path.join(self.full_cache_path, f"water_positions_id{cache_id}.pkl"), 'rb') as f:
                full_water_positions, mads = pickle.load(f)
            with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order_id{cache_id}.pkl"), 'rb') as f:
                names_unsorted = pickle.load(f)
            all_names_unsorted.append(names_unsorted)
            all_mads_unsorted.append(mads)
            all_full_water_positions_unsorted.append(full_water_positions)

        names_order = list(set(sum(all_names_unsorted, [])))
        all_mads, all_full_water_positions, all_names = [], [], []
        for idx, (mads_unsorted, full_water_positions_unsorted, names_unsorted) in enumerate(zip(all_mads_unsorted,all_full_water_positions_unsorted, all_names_unsorted)):
            name_to_pos_dict = {name: (mad, pos) for name, mad, pos in zip(names_unsorted, full_water_positions_unsorted, mads_unsorted) }
            intermediate_mads = [name_to_pos_dict[name][1] for name in names_order]
            all_mads.append((intermediate_mads))
            intermediate_pos = [name_to_pos_dict[name][0] for name in names_order]
            all_full_water_positions.append((intermediate_pos))
            
        self.full_water_positions, self.mads = [], []
        for positions_tuple in list(zip(*all_full_water_positions)):
            self.full_water_positions.append(np.concatenate(positions_tuple, axis=0))
        for positions_tuple in list(zip(*all_mads)):
            self.mads.append(np.concatenate(positions_tuple, axis=0))
        generated_mad_complex_names = names_order
        
        print('Number of complex graphs: ', len(self.loader.dataset))
            
        print('Number of MADs and positions for the complex graphs: ', len(self.full_water_positions))

        self.all_samples_per_complex = samples_per_complex * (1 if self.cache_ids_to_combine is None else len(self.cache_ids_to_combine))

        self.positions_mads_dict = {name: (pos, mad) for name, pos, mad in zip (generated_mad_complex_names, self.full_water_positions, self.mads)}
        self.dataset_names = list(self.positions_mads_dict.keys())
        if limit_complexes > 0:
            self.dataset_names = self.dataset_names[:limit_complexes]

    def len(self):
        return len(self.dataset_names)

    def get(self, idx):
        complex_name = self.dataset_names[idx]
        # weights_only=False: the cached graph is a full PyG HeteroData object, not a state dict
        # (matches PDBBind.get). torch>=2.6 defaults weights_only=True, which rejects it.
        complex_graph = torch.load(os.path.join(self.loader.dataset.full_cache_path, f"{complex_name}.pt"), weights_only=False)
        positions, mads = self.positions_mads_dict[self.dataset_names[idx]]
        
        assert(complex_graph.name == self.dataset_names[idx])
        complex_graph['ligand'].x =  complex_graph['ligand'].x[-1].repeat(positions.shape[-2], 1)
        if self.balance:
            if isinstance(self.mad_classification_cutoff, list): raise ValueError("a list for --mad_classification_cutoff can only be used without --balance")
            label = random.randint(0, 1)
            success = mads < self.mad_classification_cutoff
            n_success = np.count_nonzero(success)
            if label == 0 and n_success != self.all_samples_per_complex:
                # sample a negative example (predicted water far from any true water)
                sample = random.randint(0, self.all_samples_per_complex - n_success - 1)
                lig_pos = positions[~success][sample]
                complex_graph['ligand'].pos = torch.from_numpy(lig_pos)
            else:
                # sample a positive example
                if n_success > 0:  # if none succeeded, fall back to the matched complex
                    sample = random.randint(0, n_success - 1)
                    lig_pos = positions[success][sample]
                    complex_graph['ligand'].pos = torch.from_numpy(lig_pos)
            complex_graph.y = torch.tensor(label).float()
        else:
            sample = random.randint(0, self.all_samples_per_complex - 1)
            
            complex_graph['ligand'].pos = torch.from_numpy(positions[sample])
            complex_graph.y = torch.tensor(mads < self.mad_classification_cutoff).float()
                
            if isinstance(self.mad_classification_cutoff, list):
                complex_graph.y_binned = torch.tensor(np.logical_and(mads[sample] < self.mad_classification_cutoff + [math.inf],mads[sample] >= [0] + self.mad_classification_cutoff), dtype=torch.float).unsqueeze(0)
                complex_graph.y = torch.tensor(mads[sample] < self.mad_classification_cutoff[0]).unsqueeze(0).float()
            
            complex_graph.mad = torch.tensor(mads).float()

        complex_graph['ligand'].node_t = {'tr': 0 * torch.ones(complex_graph['ligand'].num_nodes)}
        complex_graph['receptor'].node_t = {'tr': 0 * torch.ones(complex_graph['receptor'].num_nodes)}
        if self.all_atoms:
            complex_graph['atom'].node_t = {'tr': 0 * torch.ones(complex_graph['atom'].num_nodes)}
        complex_graph.complex_t = {'tr': 0 * torch.ones(1)}
        return complex_graph



