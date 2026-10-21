"""Bucle de entrenamiento y evaluación con early stopping y checkpoints."""
import copy
import time
from pathlib import Path

import torch
from torch import nn


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    """Una pasada por el loader. Entrena si se pasa optimizer; si no, solo evalúa."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for step, (x, y) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, optimizer, device, epochs=20, criterion=None, scheduler=None,
        patience=5, ckpt_path=None, max_batches=None, verbose=True):
    """Entrena y conserva los pesos con menor pérdida de validación.

    Si ckpt_path existe se cargan los pesos y el historial en lugar de entrenar.
    """
    criterion = criterion or nn.CrossEntropyLoss()
    if ckpt_path is not None and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if verbose:
            print(f"Checkpoint cargado: {ckpt_path}")
        return ckpt["history"]

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "epoch_time": []}
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, max_batches)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, None, max_batches)
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_loss)
            else:
                scheduler.step()
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc, lr, time.time() - t0)):
            history[k].append(v)
        if verbose:
            print(f"Época {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {lr:.1e} | {history['epoch_time'][-1]:.0f}s")
        if va_loss < best_loss:
            best_loss, best_state, bad_epochs = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"Early stopping en la época {epoch}")
                break

    model.load_state_dict(best_state)
    if ckpt_path is not None:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "history": history}, ckpt_path)
    return history
