"""Candidate generation for the confidence model.

The confidence model is trained to score sampled water particles by how far they land from a
true hydration site. The training labels are produced here: for every complex in a split we
randomize ``water_ratio`` particles per residue, run the trained *score* model's reverse
diffusion, and record each particle's mean-absolute-deviation (MAD) to the nearest real water.
The resulting ``(positions, MADs)`` cache is what ``ConfidenceDataset`` trains on.

This is the explicit "generate candidates" step (``--generate_candidates_only``), decoupled from
training so it can be run once — single-process or sharded across GPUs with ``torchrun`` — before
the DDP training job reads the warm cache. Graph construction reuses the *score* model's own
dataset-scope graph cache automatically (it reads ``data_dir``/``cache_path``/graph params from
the score checkpoint's ``model_parameters.yml``), so no per-complex graphs are rebuilt.
"""

import copy
import os
import pickle
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from superwater.confidence.dataset import confidence_cache_path
from superwater.datasets.pdbbind import PDBBind
from superwater.structure_io import structure_path
from superwater.utils.diffusion_utils import get_t_schedule, t_to_sigma as t_to_sigma_compl
from superwater.utils.find_water_pos import find_real_water_pos
from superwater.utils.nearest_point_dist import get_nearest_point_distances
from superwater.utils.sampling import randomize_position_multiple, sampling
from superwater.utils.utils import get_model


def _build_origin_loader(conf_args, score_args, split_path, data_dir=None):
    """One-complex-at-a-time loader over ``split_path``.

    Graph/featurization params come from the *score* model's saved args (so the per-complex
    ``.pt`` graphs resolve to — and reuse — the score model's cache), while the split and the
    preprocessing batch/workers come from the confidence CLI. ``limit_complexes`` uses the
    confidence value so ``--limit_complexes`` actually bounds generation (e.g. for smoke tests).
    ``data_dir`` overrides the structure root (the inference/predict path runs on its own
    structures, not the score model's training data); it defaults to the score model's.
    """
    common_args = {'transform': None, 'root': data_dir or score_args.data_dir, 'limit_complexes': conf_args.limit_complexes,
                   'receptor_radius': score_args.receptor_radius,
                   'c_alpha_max_neighbors': score_args.c_alpha_max_neighbors,
                   'remove_hs': score_args.remove_hs, 'max_lig_size': score_args.max_lig_size,
                   'popsize': score_args.matching_popsize, 'maxiter': score_args.matching_maxiter,
                   'num_workers': score_args.num_workers, 'all_atoms': score_args.all_atoms,
                   'atom_radius': score_args.atom_radius, 'atom_max_neighbors': score_args.atom_max_neighbors,
                   'esm_embeddings_path': score_args.esm_embeddings_path,
                   'cache_scope': getattr(score_args, 'cache_scope', 'split')}
    dataset = PDBBind(cache_path=score_args.cache_path, split_path=split_path, keep_original=True,
                      num_conformers=score_args.num_conformers, **common_args)
    return DataLoader(dataset=dataset, batch_size=conf_args.batch_size_preprocessing,
                      num_workers=conf_args.num_workers, shuffle=False, pin_memory=False)


def _merge_shards(full_cache_path, cid, world_size, canonical_pos, canonical_names):
    """Concatenate the per-rank shards into the canonical cache and delete the shards.

    Order across shards is irrelevant: ``ConfidenceDataset`` re-keys everything by complex name
    on load. Each shard's ``positions``/``mads``/``names`` lists stay element-aligned, so a plain
    ``extend`` preserves the per-complex pairing."""
    all_positions, all_mads, all_names = [], [], []
    for r in range(world_size):
        with open(os.path.join(full_cache_path, f"water_positions_id{cid}_rank{r}.pkl"), 'rb') as f:
            positions, mads = pickle.load(f)
        with open(os.path.join(full_cache_path, f"complex_names_in_same_order_id{cid}_rank{r}.pkl"), 'rb') as f:
            names = pickle.load(f)
        all_positions.extend(positions)
        all_mads.extend(mads)
        all_names.extend(names)
    with open(canonical_pos, 'wb') as f:
        pickle.dump((all_positions, all_mads), f)
    with open(canonical_names, 'wb') as f:
        pickle.dump(all_names, f)
    for r in range(world_size):
        os.remove(os.path.join(full_cache_path, f"water_positions_id{cid}_rank{r}.pkl"))
        os.remove(os.path.join(full_cache_path, f"complex_names_in_same_order_id{cid}_rank{r}.pkl"))
    print(f"[candidates] merged {world_size} shards -> {canonical_pos} ({len(all_names)} complexes)")


def generate_candidates(conf_args, score_args, split_path, split_name, device, rank=0, world_size=1,
                        loader=None, score_model=None, data_dir=None):
    """Sample the candidate cache (per-water positions + MADs) for one split.

    Writes ``water_positions_id<cid>.pkl`` + ``complex_names_in_same_order_id<cid>.pkl`` under
    ``confidence_cache_path(...)``. Under ``torchrun`` each rank samples the complexes where
    ``idx % world_size == rank`` into a per-rank shard; rank 0 merges them. At ``world_size==1``
    this is the plain single-process path (no shards, no barriers).

    The inference/predict path reuses this with a prebuilt ``loader``, a preloaded ``score_model``,
    and an explicit ``data_dir`` (its structures, not the score model's training data). MADs are
    computed against whatever ``<name>_water`` file lives in ``data_dir`` — a real crystal water
    for evaluation, or the placeholder that ``predict`` drops in (the labels are unused at
    inference; only the sampled positions matter).
    """
    cid = conf_args.cache_creation_id
    if cid is None:
        raise ValueError("--cache_creation_id is required for candidate generation (e.g. --cache_creation_id 1).")
    data_dir = data_dir or score_args.data_dir

    full_cache_path = confidence_cache_path(conf_args.cache_path, conf_args.original_model_dir,
                                            split_name, conf_args.limit_complexes)
    os.makedirs(full_cache_path, exist_ok=True)
    canonical_pos = os.path.join(full_cache_path, f"water_positions_id{cid}.pkl")
    canonical_names = os.path.join(full_cache_path, f"complex_names_in_same_order_id{cid}.pkl")
    if os.path.exists(canonical_pos) and os.path.exists(canonical_names):
        if rank == 0:
            print(f"[candidates] {split_name}: cache already exists ({canonical_pos}); skipping.")
        return

    t_to_sigma = partial(t_to_sigma_compl, args=score_args)
    if score_model is not None:
        model = score_model
    else:
        model = get_model(score_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
        state_dict = torch.load(f'{conf_args.original_model_dir}/{conf_args.ckpt}',
                                map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
    model.eval()

    tr_schedule = get_t_schedule(inference_steps=conf_args.inference_steps)
    water_ratio = conf_args.water_ratio
    resample_steps = conf_args.resample_steps
    if rank == 0:
        print(f"[candidates] {split_name}: ckpt={conf_args.ckpt} inference_steps={conf_args.inference_steps} "
              f"water_ratio={water_ratio} resample_steps={resample_steps} "
              f"(total ratio {water_ratio * resample_steps}); world_size={world_size}")

    loader = loader if loader is not None else _build_origin_loader(conf_args, score_args, split_path, data_dir=data_dir)

    mads, full_water_positions, names = [], [], []
    for idx, orig_complex_graph in enumerate(tqdm(loader, disable=(rank != 0))):
        if idx % world_size != rank:
            continue
        # Sample `samples_per_complex` independent particle clouds, repeated `resample_steps` times.
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(conf_args.samples_per_complex)]
        res_num = int(orig_complex_graph[0]['receptor'].pos.shape[0])
        step_num_water = int(res_num * water_ratio)

        prediction_list = []
        for _ in range(resample_steps):
            sample_data_list = copy.deepcopy(data_list)
            randomize_position_multiple(sample_data_list, False, score_args.tr_sigma_max, water_num=step_num_water)
            predictions, _ = sampling(data_list=sample_data_list, model=model,
                                      inference_steps=conf_args.inference_steps, tr_schedule=tr_schedule,
                                      device=device, t_to_sigma=t_to_sigma, model_args=score_args,
                                      save_visualization=False)
            prediction_list.append(predictions)

        _water_name = orig_complex_graph.name[0]
        real_water_pos = find_real_water_pos(
            structure_path(os.path.join(data_dir, _water_name), _water_name, "_water"))

        water_pos_list = [cg['ligand'].pos.cpu().numpy()
                          for complex_graphs in prediction_list for cg in complex_graphs]
        water_pos = np.asarray([np.concatenate(water_pos_list, axis=0)], dtype=np.float32)
        positions_new = water_pos.squeeze(0) + orig_complex_graph.original_center.cpu().numpy()
        mad, _ = get_nearest_point_distances(positions_new, real_water_pos)

        mads.append(mad)
        full_water_positions.append(water_pos)
        names.append(_water_name)

    if world_size > 1:
        shard_pos = os.path.join(full_cache_path, f"water_positions_id{cid}_rank{rank}.pkl")
        shard_names = os.path.join(full_cache_path, f"complex_names_in_same_order_id{cid}_rank{rank}.pkl")
        with open(shard_pos, 'wb') as f:
            pickle.dump((full_water_positions, mads), f)
        with open(shard_names, 'wb') as f:
            pickle.dump(names, f)
        dist.barrier()
        if rank == 0:
            _merge_shards(full_cache_path, cid, world_size, canonical_pos, canonical_names)
        dist.barrier()
    else:
        with open(canonical_pos, 'wb') as f:
            pickle.dump((full_water_positions, mads), f)
        with open(canonical_names, 'wb') as f:
            pickle.dump(names, f)
        print(f"[candidates] {split_name}: wrote {canonical_pos} ({len(names)} complexes)")
