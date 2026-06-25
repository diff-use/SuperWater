import gc
import math
import os
import random
import shutil

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
import yaml
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from superwater.confidence.candidates import generate_candidates
from superwater.confidence.dataset import ConfidenceDataset, get_args
from superwater.datasets.pdbbind import PDBBind
from superwater.utils.parsing import parse_confidence_args
from superwater.utils.training import AverageMeter
from superwater.utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model

torch.multiprocessing.set_sharing_strategy('file_system')


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sigmoid_function(target, scale=4):
    '''Sigmoid-style normalizer mapping a (non-negative) MAD distance into [0, 1].'''
    return (2 / (1 + torch.exp(-(scale * target) / torch.log(torch.tensor(2)))) - 1) ** 2


def _split_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def train_epoch(model, loader, optimizer, args, mad_prediction, device, is_main=True, distributed=False):
    model.train()
    meter = AverageMeter(['confidence_loss'])

    for data in tqdm(loader, total=len(loader), disable=not is_main):
        data = data.to(device)
        optimizer.zero_grad()
        try:
            pred = model(data)
            if mad_prediction:
                labels = sigmoid_function(data.mad)
                confidence_loss = F.mse_loss(pred, labels)
            elif isinstance(args.mad_classification_cutoff, list):
                confidence_loss = F.cross_entropy(pred, data.y_binned)
            else:
                confidence_loss = F.binary_cross_entropy_with_logits(pred, data.y)

            # Always run backward so DDP's gradient all-reduce fires on every rank; decide the
            # optimizer step from a collective finiteness check so all ranks step (or skip) together.
            confidence_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            step_ok = bool(torch.isfinite(confidence_loss).item() and torch.isfinite(grad_norm).item())
            if distributed:
                flag = torch.tensor([1.0 if step_ok else 0.0], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                step_ok = flag.item() > 0

            if step_ok:
                optimizer.step()
                meter.add([confidence_loss.cpu().detach()])
            elif is_main:
                print("| WARNING: non-finite confidence loss/grad; skipping optimizer step")
            optimizer.zero_grad(set_to_none=True)
        except RuntimeError as e:
            # Per-batch recovery is only safe when NOT distributed (a skipped batch on one rank
            # would desync collectives). Under DDP let it propagate so torchrun tears down cleanly.
            if not distributed and 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
                gc.collect()
                continue
            raise

    if distributed:
        meter.all_reduce(device)
    return meter.summary()


def test_epoch(model, loader, args, mad_prediction, device, is_main=True, distributed=False):
    model.eval()
    meter = (AverageMeter(['confidence_loss'], unpooled_metrics=True) if mad_prediction
             else AverageMeter(['confidence_loss', 'accuracy', 'ROC AUC'], unpooled_metrics=True))
    all_labels = []

    for data in tqdm(loader, total=len(loader), disable=not is_main):
        data = data.to(device)
        try:
            with torch.no_grad():
                pred = model(data)
            if mad_prediction:
                labels = data.mad
                confidence_loss = F.mse_loss(pred, sigmoid_function(labels))
                meter.add([confidence_loss.cpu().detach()])
            elif isinstance(args.mad_classification_cutoff, list):
                labels = data.y_binned
                confidence_loss = F.cross_entropy(pred, labels)
                meter.add([confidence_loss.cpu().detach(), torch.tensor(0.0), torch.tensor(0.0)])
            else:
                labels = data.y
                confidence_loss = F.binary_cross_entropy_with_logits(pred, labels)
                accuracy = torch.mean((labels == (pred > 0).float()).float())
                # ROC-AUC is computed on this rank's shard only (it is not a per-batch average
                # that all-reduces correctly); a diagnostic, not the selection metric.
                try:
                    roc_auc = roc_auc_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy())
                except ValueError:
                    roc_auc = 0.0
                meter.add([confidence_loss.cpu().detach(), accuracy.cpu().detach(), torch.tensor(float(roc_auc))])
            all_labels.append(labels.detach())
        except RuntimeError as e:
            if not distributed and 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                torch.cuda.empty_cache()
                continue
            raise

    all_labels = torch.cat(all_labels) if all_labels else torch.tensor([])
    if all_labels.numel() == 0:
        baseline_metric = torch.tensor(float('nan'))
    elif mad_prediction:
        baseline_metric = (all_labels - all_labels.mean()).abs().mean()
    else:
        baseline_metric = all_labels.sum() / len(all_labels)

    if distributed:
        meter.all_reduce(device)
    return meter.summary(), baseline_metric


def train(args, model, optimizer, scheduler, train_loader, val_loader, run_dir, device,
          distributed=False, is_main=True, start_epoch=0, best_val_metric=None, best_epoch=0):
    if best_val_metric is None:
        best_val_metric = math.inf if args.main_metric_goal == 'min' else 0
    bare_model = model.module if hasattr(model, 'module') else model

    if is_main:
        print("Starting training..." if start_epoch == 0 else f"Resuming training from epoch {start_epoch}...")
    for epoch in range(start_epoch, args.n_epochs):
        if distributed and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        logs = {}
        train_metrics = train_epoch(model, train_loader, optimizer, args, args.mad_prediction, device, is_main, distributed)
        val_metrics, baseline_metric = test_epoch(model, val_loader, args, args.mad_prediction, device, is_main, distributed)

        if is_main:
            if args.mad_prediction:
                print("Epoch {}: train loss {:.4f}  val loss {:.4f}".format(
                    epoch, train_metrics['confidence_loss'], val_metrics['confidence_loss']))
            else:
                print("Epoch {}: train loss {:.4f}  val loss {:.4f}  accuracy {:.4f}".format(
                    epoch, train_metrics['confidence_loss'], val_metrics['confidence_loss'], val_metrics['accuracy']))
            if args.wandb:
                logs.update({'valinf_' + k: v for k, v in val_metrics.items()})
                logs.update({'train_' + k: v for k, v in train_metrics.items()})
                logs.update({'mean_mad' if args.mad_prediction else 'fraction_positives': float(baseline_metric),
                             'current_lr': optimizer.param_groups[0]['lr']})
                wandb.log(logs, step=epoch + 1)

        # val_metrics are globally reduced (identical across ranks), so best-model tracking and the
        # scheduler step stay in sync on every rank without a broadcast.
        metric_value = val_metrics[args.main_metric]
        metric_is_finite = math.isfinite(float(metric_value))
        if not metric_is_finite and is_main:
            print(f"| WARNING: non-finite validation metric {args.main_metric}={metric_value}; skipping scheduler/best checkpoint")
        if scheduler and metric_is_finite:
            scheduler.step(metric_value)

        state_dict = bare_model.state_dict() if is_main else None
        improved = metric_is_finite and (
            (args.main_metric_goal == 'min' and metric_value < best_val_metric) or
            (args.main_metric_goal == 'max' and metric_value > best_val_metric))
        if improved:
            best_val_metric = metric_value
            best_epoch = epoch
            if is_main:
                # best_model.pt is a BARE state_dict — the contract inference.py / superwater-infer expect.
                torch.save(state_dict, os.path.join(run_dir, "best_model.pt"))

        if is_main:
            best_path = os.path.join(run_dir, "best_model.pt")
            if args.best_model_save_frequency > 0 and (epoch + 1) % args.best_model_save_frequency == 0 and os.path.exists(best_path):
                shutil.copyfile(best_path, os.path.join(run_dir, f'best_model_epoch{epoch + 1}.pt'))
            save_numbered = ((args.model_save_frequency > 0 and (epoch + 1) % args.model_save_frequency == 0) or
                             (args.checkpoint_freq > 0 and (epoch + 1) % args.checkpoint_freq == 0))
            if save_numbered:
                torch.save(state_dict, os.path.join(run_dir, f'model_epoch{epoch + 1}.pt'))
            torch.save({
                'epoch': epoch,
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'best_val_metric': best_val_metric,
                'best_epoch': best_epoch,
            }, os.path.join(run_dir, 'last_model.pt'))

    if is_main:
        print("Best Validation {} {} on Epoch {}".format(args.main_metric, best_val_metric, best_epoch))


def construct_loader_origin(args_confidence, score_args):
    # One-complex loaders over the confidence splits. Graph/featurization params come from the
    # score model's saved args, so the per-complex .pt graphs resolve to (and reuse) the score
    # model's dataset-scope cache. ConfidenceDataset.get() loads <name>.pt from this loader's
    # full_cache_path.
    common_args = {'transform': None, 'root': score_args.data_dir, 'limit_complexes': args_confidence.limit_complexes,
                   'receptor_radius': score_args.receptor_radius,
                   'c_alpha_max_neighbors': score_args.c_alpha_max_neighbors,
                   'remove_hs': score_args.remove_hs, 'max_lig_size': score_args.max_lig_size,
                   'popsize': score_args.matching_popsize, 'maxiter': score_args.matching_maxiter,
                   'num_workers': score_args.num_workers, 'all_atoms': score_args.all_atoms,
                   'atom_radius': score_args.atom_radius, 'atom_max_neighbors': score_args.atom_max_neighbors,
                   'esm_embeddings_path': score_args.esm_embeddings_path,
                   'cache_scope': getattr(score_args, 'cache_scope', 'split')}
    train_dataset = PDBBind(cache_path=score_args.cache_path, split_path=args_confidence.split_train,
                            keep_original=True, num_conformers=score_args.num_conformers, **common_args)
    val_dataset = PDBBind(cache_path=score_args.cache_path, split_path=args_confidence.split_val,
                          keep_original=True, **common_args)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args_confidence.batch_size_preprocessing,
                              num_workers=args_confidence.num_workers, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args_confidence.batch_size_preprocessing,
                            num_workers=args_confidence.num_workers, shuffle=False)
    return train_loader, val_loader


def construct_loader_confidence(args, device, distributed=False, rank=0, world_size=1):
    score_args = get_args(args.original_model_dir)
    train_origin, val_origin = construct_loader_origin(args, score_args)

    common = {'cache_path': args.cache_path, 'original_model_dir': args.original_model_dir, 'device': device,
              'inference_steps': args.inference_steps, 'samples_per_complex': args.samples_per_complex,
              'limit_complexes': args.limit_complexes, 'all_atoms': args.all_atoms, 'balance': args.balance,
              'mad_classification_cutoff': args.mad_classification_cutoff,
              'use_original_model_cache': args.use_original_model_cache,
              'cache_creation_id': args.cache_creation_id, 'cache_ids_to_combine': args.cache_ids_to_combine,
              'model_ckpt': args.ckpt, 'running_mode': args.running_mode,
              'water_ratio': args.water_ratio, 'resample_steps': args.resample_steps}

    train_dataset = ConfidenceDataset(loader=train_origin, split=_split_name(args.split_train), args=args, **common)
    val_dataset = ConfidenceDataset(loader=val_origin, split=_split_name(args.split_val), args=args, **common)

    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                  num_workers=args.num_dataloader_workers, pin_memory=args.pin_memory, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                                num_workers=args.num_dataloader_workers, pin_memory=args.pin_memory)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_dataloader_workers, pin_memory=args.pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_dataloader_workers, pin_memory=args.pin_memory)
    return train_loader, val_loader


def main():
    args = parse_confidence_args()

    # ---- distributed bootstrap (torchrun sets these env vars) ----
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        dist.init_process_group(backend='nccl', device_id=device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)
    # Offset the seed by rank so each rank draws different diffusion noise / label samples.
    set_seed(42 + rank)

    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    assert (args.main_metric_goal == 'max' or args.main_metric_goal == 'min')

    score_model_args = get_args(args.original_model_dir)

    # The cached per-complex graphs are built with the SCORE model's featurization, so the
    # confidence model (built from confidence args) must agree on the graph-determining flags or
    # its node encoders won't match the cached feature widths. Inherit them from the score model
    # instead of making the user re-specify (and risk mismatching) them. The confidence CLI still
    # controls the confidence model's own capacity (ns/nv/num_conv_layers/embed dims/dropout).
    args.all_atoms = score_model_args.all_atoms
    if args.esm_embeddings_path is None:
        args.esm_embeddings_path = score_model_args.esm_embeddings_path

    # ---- explicit candidate generation step (sample the cache, then exit) ----
    if args.generate_candidates_only:
        for split_path in (args.split_train, args.split_val):
            generate_candidates(args, score_model_args, split_path, _split_name(split_path),
                                device, rank=rank, world_size=world_size)
        if is_main:
            print("Candidate cache prepared; exiting (--generate_candidates_only).")
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    # Build loaders. Rank-0-first barrier: if any per-complex graph .pt is cold, rank 0 builds it
    # while the others wait, instead of every rank writing the same files. The candidate cache must
    # already exist (ConfidenceDataset raises a clear error otherwise — run --generate_candidates_only).
    if distributed and not is_main:
        dist.barrier()
    train_loader, val_loader = construct_loader_confidence(args, device, distributed=distributed, rank=rank, world_size=world_size)
    if distributed and is_main:
        dist.barrier()

    # Build the confidence model bare, restore weights, convert BN -> SyncBatchNorm, then DDP-wrap.
    model = get_model(score_model_args if args.transfer_weights else args, device,
                      t_to_sigma=None, confidence_mode=True, no_parallel=True)

    restart_state = None
    if args.transfer_weights:
        if is_main:
            print("HAPPENING | Transferring weights from original_model_dir into the confidence model.")
        checkpoint = torch.load(os.path.join(args.original_model_dir, args.ckpt), map_location=device)
        model_state_dict = model.state_dict()
        transfer_weights_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        model_state_dict.update(transfer_weights_dict)
        model.load_state_dict(model_state_dict)
    elif args.restart_dir:
        restart_state = torch.load(f'{args.restart_dir}/last_model.pt', map_location=device)
        model.load_state_dict(restart_state['model'], strict=True)
        if is_main:
            print("Restarting from epoch", restart_state['epoch'])

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=args.find_unused_parameters, broadcast_buffers=True)

    optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.main_metric_goal)

    start_epoch = 0
    best_val_metric = math.inf if args.main_metric_goal == 'min' else 0
    best_epoch = 0
    if restart_state is not None:
        optimizer.load_state_dict(restart_state['optimizer'])
        if scheduler is not None and restart_state.get('scheduler') is not None:
            scheduler.load_state_dict(restart_state['scheduler'])
        start_epoch = restart_state['epoch'] + 1
        best_val_metric = restart_state.get('best_val_metric', best_val_metric)
        best_epoch = restart_state.get('best_epoch', restart_state['epoch'])
        if is_main:
            print(f"Resuming at epoch {start_epoch} (n_epochs={args.n_epochs}); "
                  f"best {args.main_metric}={best_val_metric} @ epoch {best_epoch}")

    numel = sum(p.numel() for p in model.parameters())
    if is_main:
        print('Model with', numel, 'parameters')

    run_dir = os.path.join(args.log_dir, args.run_name)
    if is_main:
        if args.wandb:
            wandb.init(entity=args.wandb_entity, settings=wandb.Settings(start_method="fork"),
                       project=args.project, name=args.run_name, config=args)
            wandb.log({'numel': numel})
        os.makedirs(run_dir, exist_ok=True)
        save_yaml_file(os.path.join(run_dir, 'model_parameters.yml'), args.__dict__)
    args.device = device

    train(args, model, optimizer, scheduler, train_loader, val_loader, run_dir, device,
          distributed=distributed, is_main=is_main, start_epoch=start_epoch,
          best_val_metric=best_val_metric, best_epoch=best_epoch)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
