import torch
import torch.distributed as dist
from tqdm import tqdm


def loss_function(tr_pred, expand_tr_sigma, data, t_to_sigma, device, tr_weight=1, apply_mean=True):
    mean_dims = (0, 1) if apply_mean else 1

    # translation component. `data` is always a single batched `Batch` (the collating
    # DataLoader concatenates per-graph `tr_score` in ligand-node order, matching tr_pred).
    # The whole loss is computed on CPU to stay numerically identical to the legacy CPU path.
    tr_score = data.tr_score.cpu()
    expand_tr_sigma = expand_tr_sigma.unsqueeze(-1).cpu()


    tr_loss = ((tr_pred.cpu() - tr_score) ** 2 * (expand_tr_sigma ** 2 + 1e-6)).mean(dim=mean_dims)

    # Do not mask non-finite losses here. train_epoch detects and skips them
    # before the optimizer step so unstable batches cannot corrupt model weights.

    tr_base_loss = (tr_score ** 2 *  expand_tr_sigma ** 2).mean(dim=mean_dims).detach()

    loss = tr_loss * tr_weight
    return loss, tr_loss.detach(), tr_base_loss


class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum() if self.unpooled_metrics else v
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    def all_reduce(self, device):
        """Sum the accumulated metric totals and sample counts across all DDP ranks so
        every rank's summary() reflects the global average (used for logging, best-model
        selection and the LR scheduler). No-op when not running distributed."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        if self.intervals == 1:
            count_t = torch.tensor([float(self.count)], device=device)
            dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
            self.count = count_t.item()
        else:
            count_t = self.count.to(device)
            dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
            self.count = count_t.cpu()
        for t in self.types:
            acc_t = self.acc[t].to(device)
            dist.all_reduce(acc_t, op=dist.ReduceOp.SUM)
            self.acc[t] = acc_t.cpu()

    def summary(self):
        if self.intervals == 1:
            if self.count == 0:
                return {k: float('nan') for k in self.acc}
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out


def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weigths, epoch=0,
                wandb_enabled=False, iter_log_fn=None, distributed=False, is_main=True):
    model.train()
    meter = AverageMeter(['loss', 'tr_loss', 'tr_base_loss'])
    num_batches = len(loader)

    # DistributedSampler must be told the epoch so it reshuffles differently each epoch.
    if distributed and hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)

    for batch_idx, data in enumerate(tqdm(loader, total=num_batches, disable=not is_main)):
        data = data.to(device)
        if data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            tr_pred, expand_tr_sigma, expand_batch_idx = model(data)

            loss, tr_loss, tr_base_loss = (
                loss_fn(tr_pred, expand_tr_sigma, data=data, t_to_sigma=t_to_sigma, device=device)
            )

            # Always run backward so DDP's gradient all-reduce fires on every rank every
            # iteration (skipping it on one rank would deadlock the collective). The
            # optimizer step is then gated on a *collective* finiteness check: the step
            # happens only if loss and grad norm are finite on ALL ranks, otherwise no
            # rank steps and the (possibly NaN) grads are discarded. Net effect on weights
            # is identical to the previous "skip the batch" behaviour.
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            step_ok = bool(torch.isfinite(loss).item() and torch.isfinite(grad_norm).item())
            if distributed:
                flag = torch.tensor([1.0 if step_ok else 0.0], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                step_ok = flag.item() > 0

            if step_ok:
                optimizer.step()
                ema_weigths.update(model.parameters())
                meter.add([loss.cpu().detach(), tr_loss, tr_base_loss])
                if iter_log_fn is not None:
                    iter_log_fn(epoch, batch_idx, loss.cpu().detach().item(), tr_loss.item(), tr_base_loss.item())
            elif is_main:
                print(f'| WARNING: non-finite loss/grad at batch {batch_idx}, skipping optimizer step')

            optimizer.zero_grad(set_to_none=True)

        except RuntimeError as e:
            # Graceful per-batch recovery is only safe when NOT distributed: under DDP a
            # rank that fails here never built the backward graph, so silently continuing
            # would desync the collectives. Let it propagate (torchrun tears down all ranks).
            if not distributed and ('out of memory' in str(e) or 'Input mismatch' in str(e)):
                reason = 'ran out of memory' if 'out of memory' in str(e) else 'weird torch_cluster error'
                print(f'| WARNING: {reason}, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    if distributed:
        meter.all_reduce(device)
    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False,
               distributed=False, is_main=True):
    model.eval()
    meter = AverageMeter(['loss', 'tr_loss', 'tr_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'tr_base_loss'],
            unpooled_metrics=True, intervals=10)

    for data in tqdm(loader, total=len(loader), disable=not is_main):
        data = data.to(device)
        try:
            with torch.no_grad():
                tr_pred, expand_tr_sigma, expand_batch_idx = model(data)

            loss, tr_loss, tr_base_loss = \
                loss_fn(tr_pred, expand_tr_sigma, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device)
            meter.add([loss.cpu().detach(), tr_loss, tr_base_loss])

            if test_sigma_intervals > 0:
                complex_t_tr  = data.complex_t['tr'].cpu()
                sigma_index_tr = torch.round(complex_t_tr * (10 - 1)).long()
                expand_sigma_index_tr = torch.index_select(sigma_index_tr, dim=0, index=expand_batch_idx.cpu())
                meter_all.add(
                    [loss.cpu().detach(), tr_loss, tr_base_loss],
                    [expand_sigma_index_tr, expand_sigma_index_tr, expand_sigma_index_tr, expand_sigma_index_tr])

        except RuntimeError as e:
            # Validation has no per-batch DDP collective, so a per-rank skip is safe here.
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    if distributed:
        meter.all_reduce(device)
        if test_sigma_intervals > 0:
            meter_all.all_reduce(device)

    out = meter.summary()
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out
