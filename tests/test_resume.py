"""Tests for training-resume bookkeeping (continue from the exact epoch a run stopped at)."""
import math
import os

import torch

from superwater.train import resume_bookkeeping


def test_resume_continues_from_next_epoch():
    # A run that last saved epoch 99 should resume at epoch 100.
    start_epoch, best_val_loss, best_epoch = resume_bookkeeping(
        {'epoch': 99, 'best_val_loss': 0.42, 'best_epoch': 87}
    )
    assert start_epoch == 100
    assert best_val_loss == 0.42
    assert best_epoch == 87


def test_resume_epoch_range_matches_target_total():
    # With n_epochs as the absolute target, resuming at 100 with target 300
    # runs epochs 100..299 (200 more), never re-running completed epochs.
    start_epoch, _, _ = resume_bookkeeping({'epoch': 99})
    n_epochs = 300
    epochs = list(range(start_epoch, n_epochs))
    assert epochs[0] == 100
    assert epochs[-1] == 299
    assert len(epochs) == 200


def test_resume_no_op_when_target_already_reached():
    # Resuming with n_epochs <= already-completed epochs runs nothing (no negative range).
    start_epoch, _, _ = resume_bookkeeping({'epoch': 99})
    assert list(range(start_epoch, 100)) == []


def test_resume_backward_compatible_with_old_checkpoint():
    # Checkpoints written before scheduler/best tracking existed lack those keys;
    # bookkeeping must still resume cleanly with sane defaults.
    start_epoch, best_val_loss, best_epoch = resume_bookkeeping({'epoch': 50})
    assert start_epoch == 51
    assert best_val_loss == math.inf
    assert best_epoch == 50  # falls back to the resumed epoch


def test_checkpoint_roundtrip_preserves_resume_state(tmp_path):
    # The new checkpoint payload (scheduler + best tracking) must survive save/load,
    # and a ReduceLROnPlateau scheduler must restore its internal counters.
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)

    # Simulate a few plateau steps so the scheduler carries non-trivial state.
    for loss in [1.0, 0.9, 0.9, 0.9]:
        scheduler.step(loss)

    ckpt = {
        'epoch': 99,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_val_loss': 0.9,
        'best_epoch': 1,
    }
    path = os.path.join(tmp_path, 'last_model.pt')
    torch.save(ckpt, path)
    loaded = torch.load(path, map_location='cpu')

    start_epoch, best_val_loss, best_epoch = resume_bookkeeping(loaded)
    assert (start_epoch, best_val_loss, best_epoch) == (100, 0.9, 1)

    # Scheduler state restores exactly onto a fresh scheduler.
    fresh_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    fresh_sched.load_state_dict(loaded['scheduler'])
    assert fresh_sched.state_dict() == scheduler.state_dict()
