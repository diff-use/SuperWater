import copy
import csv
import math
import os
from functools import partial
import numpy as np
import random
import wandb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import yaml

from superwater.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from superwater.datasets.pdbbind import construct_loader
from superwater.utils.parsing import parse_train_args
from superwater.utils.training import train_epoch, test_epoch, loss_function
from superwater.utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage


def resume_bookkeeping(restart_state):
    """Derive (start_epoch, best_val_loss, best_epoch) from a `last_model.pt` checkpoint dict.

    `start_epoch` is the next epoch to run (saved epoch + 1), so training continues exactly
    where it stopped. `.get()` keeps checkpoints written before scheduler/best-tracking was
    added loadable: they fall back to inf / the resumed epoch."""
    start_epoch = restart_state['epoch'] + 1
    best_val_loss = restart_state.get('best_val_loss', math.inf)
    best_epoch = restart_state.get('best_epoch', restart_state['epoch'])
    return start_epoch, best_val_loss, best_epoch


def train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, infer_loader,
          t_to_sigma, run_dir, device, distributed=False, is_main=True,
          start_epoch=0, best_val_loss=math.inf, best_epoch=0):
    best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_val_inference_epoch = 0
    loss_fn = partial(loss_function, tr_weight=args.tr_weight)

    # The bare (unwrapped) module whose state_dict we checkpoint. Under DDP `model` is the
    # DDP wrapper; otherwise it is the model itself.
    bare_model = model.module if hasattr(model, 'module') else model

    # Only the main rank writes CSV logs; other ranks use a no-op iteration logger.
    if is_main:
        print("Starting training..." if start_epoch == 0 else f"Resuming training from epoch {start_epoch}...")
        # Append (and skip the header) when resuming so the existing loss history is preserved.
        csv_mode = 'w' if start_epoch == 0 else 'a'
        iter_f = open(os.path.join(run_dir, 'losses_iter.csv'), csv_mode, newline='')
        epoch_f = open(os.path.join(run_dir, 'losses_epoch.csv'), csv_mode, newline='')
        iter_writer = csv.writer(iter_f)
        epoch_writer = csv.writer(epoch_f)
        if start_epoch == 0:
            iter_writer.writerow(['epoch', 'iteration', 'loss', 'tr_loss', 'tr_base_loss'])
            epoch_writer.writerow(['epoch', 'split', 'loss', 'tr_loss', 'tr_base_loss'])

        def log_iter(epoch, batch_idx, loss, tr_loss, tr_base_loss):
            iter_writer.writerow([epoch, batch_idx, loss, tr_loss, tr_base_loss])
    else:
        iter_f = epoch_f = None
        log_iter = None

    try:
        for epoch in range(start_epoch, args.n_epochs):
            if is_main and epoch % 5 == 0: print("Run name: ", args.run_name)
            logs = {}
            train_losses = train_epoch(model, train_loader, optimizer, device, t_to_sigma, loss_fn, ema_weights,
                                       epoch=epoch, wandb_enabled=args.wandb, iter_log_fn=log_iter,
                                       distributed=distributed, is_main=is_main)
            if is_main:
                print("Epoch {}: Training loss {:.4f}  tr {:.4f} "
                      .format(epoch, train_losses['loss'], train_losses['tr_loss']))

            # Swap in EMA weights for validation on ALL ranks so the reduced val loss is
            # computed from identical weights everywhere.
            ema_weights.store(model.parameters())
            if args.use_ema: ema_weights.copy_to(model.parameters())
            val_losses = test_epoch(model, val_loader, device, t_to_sigma, loss_fn, args.test_sigma_intervals,
                                    distributed=distributed, is_main=is_main)
            if is_main:
                print("Epoch {}: Validation loss {:.4f}  tr {:.4f} "
                      .format(epoch, val_losses['loss'], val_losses['tr_loss']))

            if not args.use_ema: ema_weights.copy_to(model.parameters())
            ema_state_dict = copy.deepcopy(bare_model.state_dict()) if is_main else None
            ema_weights.restore(model.parameters())

            if is_main:
                epoch_writer.writerow([epoch, 'train', train_losses['loss'], train_losses['tr_loss'], train_losses['tr_base_loss']])
                epoch_writer.writerow([epoch, 'val', val_losses['loss'], val_losses['tr_loss'], val_losses['tr_base_loss']])
                epoch_f.flush()
                iter_f.flush()

                if args.wandb:
                    # Log the same metrics the CSV/console record. Dumping every key instead
                    # would add the 30 per-interval metrics from --test_sigma_intervals
                    # (int0_*..int9_*), whose base-loss values are huge (~800 at epoch 0) and
                    # never appear in the train logs — noisy and misleading on the dashboard.
                    log_keys = ['loss', 'tr_loss', 'tr_base_loss']
                    logs.update({'train_' + k: train_losses[k] for k in log_keys})
                    logs.update({'val_' + k: val_losses[k] for k in log_keys})
                    logs['current_lr'] = optimizer.param_groups[0]['lr']
                    wandb.log(logs, step=epoch + 1)

            # val_losses are globally reduced (identical across ranks), so best-model
            # tracking and the scheduler step stay in sync on every rank without a broadcast.
            state_dict = bare_model.state_dict() if is_main else None
            val_loss_is_finite = math.isfinite(val_losses["loss"])
            if val_loss_is_finite and val_losses["loss"] <= best_val_loss:
                best_val_loss = val_losses["loss"]
                best_epoch = epoch
                if is_main:
                    torch.save(state_dict, os.path.join(run_dir, "best_model.pt"))
                    torch.save(ema_state_dict, os.path.join(run_dir, "best_ema_model.pt"))
            elif not val_loss_is_finite and is_main:
                print("| WARNING: non-finite validation loss ({}); skipping best checkpoint and scheduler step".format(val_losses["loss"]))

            if scheduler and val_loss_is_finite:
                if args.val_inference_freq is not None:
                    scheduler.step(best_val_inference_value)
                else:
                    scheduler.step(val_losses["loss"])

            if is_main:
                torch.save({
                    'epoch': epoch,
                    'model': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'ema_weights': ema_weights.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                }, os.path.join(run_dir, 'last_model.pt'))

                if args.checkpoint_freq > 0 and (epoch + 1) % args.checkpoint_freq == 0:
                    torch.save(state_dict, os.path.join(run_dir, f'model_epoch{epoch + 1}.pt'))
                    torch.save(ema_state_dict, os.path.join(run_dir, f'ema_model_epoch{epoch + 1}.pt'))
    finally:
        if is_main:
            iter_f.close()
            epoch_f.close()

    if is_main:
        print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))

def set_seed(seed: int = 42) -> None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")

def main_function():
    args = parse_train_args()

    # ---- distributed bootstrap (torchrun sets these env vars) ----
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    is_main = (rank == 0)

    # Each rank draws different diffusion noise for its different shard of samples (correct
    # data parallelism); offsetting the seed by rank avoids correlated RNG across ranks.
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
    assert (args.inference_earlystop_goal == 'max' or args.inference_earlystop_goal == 'min')
    if args.val_inference_freq is not None and args.scheduler is not None:
        assert (args.scheduler_patience > args.val_inference_freq) # otherwise we will just stop training after args.scheduler_patience epochs
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # construct loader
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    # Build the graph cache on rank 0 first: non-main ranks wait at the barrier while rank 0
    # preprocesses, then they only read the cache. Without this, every rank writes the same
    # .pt files concurrently (corruption + redundant work) on a fresh/partial cache.
    if distributed and not is_main:
        dist.barrier()
    train_loader, val_loader, infer_loader = construct_loader(
        args, t_to_sigma, distributed=distributed, rank=rank, world_size=world_size)
    if distributed and is_main:
        dist.barrier()

    # Build the bare model, optionally restore weights, then wrap in DDP. Wrapping after the
    # restore means DDP broadcasts the (loaded) rank-0 weights to all ranks at construction.
    model = get_model(args, device, t_to_sigma=t_to_sigma, no_parallel=True)

    restart_state = None
    if args.restart_dir:
        try:
            restart_state = torch.load(f'{args.restart_dir}/last_model.pt', map_location=device)
            model.load_state_dict(restart_state['model'], strict=True)
            if is_main: print("Restarting from epoch", restart_state['epoch'])
        except Exception as e:
            print("Exception", e)
            restart_state = None
            best = torch.load(f'{args.restart_dir}/best_model.pt', map_location=device)
            model.load_state_dict(best, strict=True)
            print("Due to exception had to take the best epoch and no optimiser")

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=args.find_unused_parameters, broadcast_buffers=True)

    optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.inference_earlystop_goal if args.val_inference_freq is not None else 'min')
    ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)

    start_epoch, best_val_loss, best_epoch = 0, math.inf, 0
    if restart_state is not None:
        if args.restart_lr is not None: restart_state['optimizer']['param_groups'][0]['lr'] = args.restart_lr
        optimizer.load_state_dict(restart_state['optimizer'])
        if hasattr(args, 'ema_rate'):
            ema_weights.load_state_dict(restart_state['ema_weights'], device=device)
        # Resume the epoch counter, LR scheduler, and best-checkpoint tracking so training
        # continues exactly where it left off.
        if scheduler is not None and restart_state.get('scheduler') is not None:
            scheduler.load_state_dict(restart_state['scheduler'])
        start_epoch, best_val_loss, best_epoch = resume_bookkeeping(restart_state)
        if is_main:
            print(f"Resuming at epoch {start_epoch} (target n_epochs={args.n_epochs}); "
                  f"best_val_loss={best_val_loss} @ epoch {best_epoch}")
            if start_epoch >= args.n_epochs:
                print(f"WARNING: start_epoch ({start_epoch}) >= n_epochs ({args.n_epochs}); "
                      f"no epochs will run. Increase --n_epochs to continue training.")

    numel = sum([p.numel() for p in model.parameters()])
    if is_main: print('Model with', numel, 'parameters')

    run_dir = os.path.join(args.log_dir, args.run_name)

    if is_main:
        if args.wandb:
            wandb.init(
                entity=args.wandb_entity,
                settings=wandb.Settings(start_method="fork"),
                project=args.project,
                name=args.run_name,
                config=args
            )
            wandb.log({'numel': numel})

        # record parameters
        os.makedirs(run_dir, exist_ok=True)
        yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
        save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, infer_loader,
          t_to_sigma, run_dir, device, distributed=distributed, is_main=is_main,
          start_epoch=start_epoch, best_val_loss=best_val_loss, best_epoch=best_epoch)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main_function()
