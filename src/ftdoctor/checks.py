"""The checks. Each one answers a question people actually ask about a fine-tune.

Every check returns ``None`` when it has nothing to say -- a report that lists
eleven green ticks buries the one line that matters. Every finding carries the
numbers it fired on and a concrete change to make.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from . import stats
from .findings import Finding, Severity
from .history import History, Series
from .metrics import Selection
from .picker import Pick, describe_band

_TRAINING_ARGS = (
    'TrainingArguments(..., eval_strategy="steps", eval_steps=N, save_strategy="steps", '
    'save_steps=N, load_best_model_at_end=True, metric_for_best_model="eval_loss", '
    "save_total_limit=3)"
)


def run_all(history: History, selection: Optional[Selection], p: Optional[Pick]) -> List[Finding]:
    if history.is_rl:
        findings = [rl_run(history), lr_schedule(history), incomplete(history)]
        return [f for f in findings if f is not None]
    findings = [
        no_eval(history, selection),
        overfitting(history, p),
        stopped_improving(history, p),
        metric_disagreement(history, selection),
        loss_degenerate(history),
        not_learning(history),
        loss_spikes(history),
        lr_schedule(history),
        gradients(history),
        eval_below_train(history),
        incomplete(history),
    ]
    return [f for f in findings if f is not None]


# -- RL runs are a different animal ------------------------------------------


def rl_run(history: History) -> Finding:
    evidence = [
        "the log carries reward metrics ("
        + ", ".join(sorted(k for k in history.train if "reward" in k)[:3])
        + "), so this is reinforcement learning, not supervised fine-tuning"
    ]
    loss = history.train.get("loss")
    if loss is not None and len(loss):
        evidence.append(
            f"its logged loss has median {float(np.median(loss.values)):.2g}: in GRPO-style "
            "training that is a policy-gradient surrogate that sits near zero by construction, "
            "not a measure of fit"
        )
    grad = history.train.get("grad_norm")
    if grad is not None and len(grad):
        zero = float(np.mean(grad.values == 0.0))
        if zero > 0:
            evidence.append(
                f"{_pct(zero)} of steps have a gradient norm of exactly zero -- in GRPO that is "
                "groups whose rollouts all scored the same, not a broken setup"
            )
    return Finding(
        "rl_run",
        "This is an RL run",
        Severity.INFO,
        "Training loss means something different in RL, so the loss-based checks are skipped "
        "rather than misread.",
        evidence=evidence,
        fix=[
            "For GRPO / PPO / RLVR diagnosis -- degenerate groups, entropy collapse, reward "
            "hacking, truncation -- use rldoctor on the same file: "
            "pip install rldoctor && rldoctor diagnose trainer_state.json",
        ],
    )


# -- helpers ----------------------------------------------------------------


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, window | 1)  # odd
    if values.size < window:
        return np.full_like(values, np.median(values))
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(view, axis=1)


def _train_loss_near(history: History, step: float, k: int = 3) -> Optional[float]:
    """Median of the training loss logged closest to ``step``."""
    loss = history.train.get("loss")
    if loss is None or not len(loss):
        return None
    order = np.argsort(np.abs(loss.steps - step))[:k]
    return float(np.median(loss.values[order]))


def _after_warmup(n: int) -> int:
    """Index from which a series is past its first few steps.

    The opening steps of almost any fine-tune have large gradients and a fast
    falling loss -- the optimizer has no moment estimates yet and the learning
    rate is still warming up. Flagging those as "spikes" fires on nearly every
    run and teaches people to ignore the check.
    """
    return min(max(int(n * 0.05), 5), max(n - 5, 0))


def _pct(x: float) -> str:
    return f"{x:.0%}" if abs(x) >= 0.01 else f"{x:.1%}"


# -- 1. is there anything to judge "best" by? -------------------------------


def no_eval(history: History, selection: Optional[Selection]) -> Optional[Finding]:
    n_evals = max((len(s) for s in history.evals.values()), default=0)
    if selection is not None and n_evals >= 4:
        return None
    if selection is None:
        return Finding(
            "no_eval",
            "No evaluation",
            Severity.WARNING,
            "This run logged no eval metric, so nothing -- this tool included -- can tell you "
            "whether it overfit or which checkpoint is best. Training loss only ever goes down.",
            fix=[
                "Pass an eval_dataset and evaluate on the same schedule you save on: "
                + _TRAINING_ARGS,
                "Keep the eval set out of training. A few hundred held-out examples are enough "
                "to see a trend.",
            ],
        )
    return Finding(
        "no_eval",
        "Too few evaluations",
        Severity.INFO,
        f"Only {n_evals} evaluation(s) were logged -- too few to tell a trend from noise, so "
        "the checkpoint pick below is a weak one.",
        fix=["Evaluate at least 8-10 times over the run (eval_steps ~ total_steps / 10)."],
        metrics={"n_evals": n_evals},
    )


# -- 2. did it overfit? -----------------------------------------------------


def overfitting(history: History, p: Optional[Pick]) -> Optional[Finding]:
    if p is None or len(p.steps) < 4 or p.final_is_equivalent:
        return None

    sel = p.selection
    evidence = [
        f"best {sel.label} was {p.best_value:.4g} at step {p.best_step}; the last eval, at step "
        f"{p.final_step}, is {p.final_value:.4g} -- {_pct(p.final_gap_rel)} worse",
        f"that is outside the noise band ({describe_band(p)}), so it is a real change, not "
        "eval jitter",
    ]

    after = p.steps > p.best_step
    if int(after.sum()) >= 3:
        tr = stats.trend(p.steps[after], sel.score(p.values[after]))
        if tr.direction == "up":
            evidence.append(
                f"and it is still getting worse: {int(after.sum())} evals after the best trend "
                f"the wrong way (Mann-Kendall p{stats.fmt_p(tr.p_value)})"
            )

    memorising = False
    loss = history.train.get("loss")
    if loss is not None and len(loss) >= 6:
        window = (loss.steps >= p.best_step) & (loss.steps <= p.final_step)
        if int(window.sum()) >= 4:
            tr_train = stats.trend(loss.steps[window], loss.values[window])
            if tr_train.direction == "down":
                evidence.append(
                    "meanwhile training loss kept falling over the same steps -- the classic "
                    "shape of a model learning its training set rather than the task"
                )
                memorising = True
        final_train = float(np.median(loss.values[-3:]))
        if final_train < 0.05 and sel.key == "eval_loss":
            evidence.append(
                f"final training loss is {final_train:.3g}: the training set is effectively "
                "memorised"
            )

    eval_loss = history.evals.get("eval_loss")
    if eval_loss is not None and len(eval_loss) >= 2:
        gap_best = _gap(history, eval_loss, p.best_step)
        gap_final = _gap(history, eval_loss, p.final_step)
        if gap_best is not None and gap_final is not None and gap_final > gap_best:
            evidence.append(
                f"the eval-minus-train loss gap widened from {gap_best:.3g} to {gap_final:.3g}"
            )

    wasted = p.steps_after_best / max(p.final_step, 1)
    if wasted >= 0.1:
        evidence.append(
            f"{_pct(wasted)} of training happened after the best checkpoint"
            + _hours(history, wasted)
        )

    # Outside the noise band means the drop is real; it does not mean it matters.
    # On 89 public runs, a band-only rule called 1% degradations "overfitting".
    if p.final_gap_rel >= 0.10:
        severity = Severity.CRITICAL
    elif p.final_gap_rel >= 0.03:
        severity = Severity.WARNING
    else:
        severity = Severity.INFO
    keep = _keep_line(p)
    fix = [keep] if keep else []
    fix += [
        f"Train for fewer steps next time -- about {p.best_step} rather than {p.final_step}"
        + _epochs_at(history, p.best_step)
        + " -- or let the Trainer stop for you: load_best_model_at_end=True plus "
        "EarlyStoppingCallback(early_stopping_patience=3).",
    ]
    if memorising:
        fix.append(
            "To push the best point later instead of earlier: more (or more varied) data, "
            "fewer epochs over the same data, a lower learning rate, or for LoRA a smaller "
            "rank and some lora_dropout."
        )
    if severity is Severity.INFO:
        title = "Final checkpoint slightly worse"
        summary = (
            f"The final model is {_pct(p.final_gap_rel)} worse on {sel.label} than the one at "
            f"step {p.best_step}: small, but outside the eval's noise, so it is real."
        )
    else:
        title = "Overfitting"
        summary = (
            f"The final model is {_pct(p.final_gap_rel)} worse on {sel.label} than the one at "
            f"step {p.best_step}. Everything after that point made it worse."
        )
    return Finding(
        "overfitting",
        title,
        severity,
        summary,
        evidence=evidence,
        fix=fix,
        metrics={
            "best_step": p.best_step,
            "final_gap_rel": p.final_gap_rel,
            "steps_after_best_frac": wasted,
        },
    )


def _gap(history: History, eval_loss: Series, step: int) -> Optional[float]:
    value = eval_loss.at(float(step))
    train = _train_loss_near(history, step)
    if value is None or train is None:
        return None
    return value - train


def _epochs_at(history: History, step: int) -> str:
    """ " (~1.6 epochs instead of 3)" -- people set num_train_epochs, not steps."""
    epoch = history.meta.get("epoch")
    final = history.final_step
    if not isinstance(epoch, (int, float)) or epoch <= 0 or final <= 0:
        return ""
    at = epoch * step / final
    return f" (~{at:.2g} epochs instead of {epoch:.3g})"


def _hours(history: History, frac: float) -> str:
    runtime = history.meta.get("train_runtime")
    if isinstance(runtime, (int, float)) and runtime > 0:
        hours = runtime * frac / 3600.0
        return f" (about {hours:.1f} h of the {runtime / 3600.0:.1f} h run)"
    return ""


def _keep_line(p: Pick) -> Optional[str]:
    if p.keep_step is not None:
        line = f"Keep checkpoint-{p.keep_step} ({p.selection.label} {p.keep_value:.4g})."
        if p.unsaved_best:
            line += (
                f" The best eval, at step {p.best_step}, was never saved -- set save_steps equal "
                "to eval_steps so the best point is always a checkpoint you can load."
            )
        return line
    if not p.checkpoints_known:
        return (
            f"Use the checkpoint from step {p.best_step} if you saved one -- this log does not say "
            "which checkpoints still exist. Point ftdoctor at the output directory to check."
        )
    return None


# -- 3. did it stop improving long before the end? ---------------------------


def stopped_improving(history: History, p: Optional[Pick]) -> Optional[Finding]:
    if p is None or len(p.steps) < 5 or not p.final_is_equivalent:
        return None
    # First eval that is already within the band of the final best.
    first_equivalent = min(p.equivalent_steps)
    frac = (p.final_step - first_equivalent) / max(p.final_step, 1)
    if frac < 0.35:
        return None
    return Finding(
        "stopped_improving",
        "Stopped improving early",
        Severity.INFO,
        f"{p.selection.label} reached its final level by step {first_equivalent}. The last "
        f"{_pct(frac)} of training changed it by less than the eval's own noise.",
        evidence=[
            f"every eval from step {first_equivalent} on is within the noise band of the best: "
            f"{describe_band(p)}" + _hours(history, frac),
            "nothing got worse, so the final checkpoint is fine to keep",
        ],
        fix=[
            f"Next time, about {first_equivalent} steps buys the same result. Or add "
            "EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=...) and "
            "let the run end itself.",
        ],
        metrics={"first_equivalent_step": first_equivalent, "tail_frac": frac},
    )


# -- 4. do loss and the selection metric disagree? ---------------------------


def metric_disagreement(history: History, selection: Optional[Selection]) -> Optional[Finding]:
    if selection is None or selection.key == "eval_loss":
        return None
    loss = history.evals.get("eval_loss")
    task = history.evals.get(selection.key)
    if loss is None or task is None or len(loss) < 4 or len(task) < 4:
        return None

    loss_best_i = int(np.argmin(loss.values))
    loss_rise = (loss.values[-1] - loss.values[loss_best_i]) / max(abs(loss.values[loss_best_i]), 1e-12)
    task_score = selection.score(task.values)
    task_best_i = int(np.argmin(task_score))
    task_final_is_best = task_best_i == len(task_score) - 1 or (
        task_score[-1] - task_score[task_best_i]
    ) <= max(2 * stats.diff_sigma(task_score), 0.01 * abs(task_score[task_best_i]))
    if loss_rise < 0.10 or not task_final_is_best:
        return None

    return Finding(
        "metric_disagreement",
        "Eval loss and your metric disagree",
        Severity.INFO,
        f"eval_loss rose {_pct(loss_rise)} from its minimum at step {int(loss.steps[loss_best_i])}, "
        f"while {selection.label} kept improving. Judged on {selection.label} -- {selection.reason} "
        "-- the final checkpoint is fine. This is not overfitting in the sense that matters to you.",
        evidence=[
            f"eval_loss: {loss.values[loss_best_i]:.4g} at step {int(loss.steps[loss_best_i])} -> "
            f"{loss.values[-1]:.4g} at step {int(loss.steps[-1])}",
            f"{selection.label}: {task.values[0]:.4g} -> {task.values[-1]:.4g} (best "
            f"{task.values[task_best_i]:.4g})",
            "common in classification fine-tunes: the model grows more confident, so it pays "
            "more loss on the examples it still gets wrong while getting more of them right",
        ],
        fix=[
            f"If you need calibrated probabilities rather than just {selection.label}, the step-"
            f"{int(loss.steps[loss_best_i])} checkpoint is the better one -- or keep the final "
            "model and apply temperature scaling on a held-out set.",
        ],
        metrics={"eval_loss_rise": float(loss_rise)},
    )


# -- 5. is the loss signal broken? ------------------------------------------


def loss_degenerate(history: History) -> Optional[Finding]:
    nan_loss = history.nonfinite.get("loss", [])
    if nan_loss:
        return Finding(
            "loss_degenerate",
            "NaN loss",
            Severity.CRITICAL,
            f"Training loss became NaN or inf at step {min(nan_loss)}. Every update after that "
            "point is garbage, whatever the checkpoints are called.",
            evidence=[f"non-finite loss logged at {len(nan_loss)} step(s)"],
            fix=[
                "Resume from the last checkpoint before that step.",
                "Use bf16 rather than fp16 if the hardware supports it, lower the learning rate, "
                "and keep max_grad_norm=1.0.",
            ],
        )
    loss = history.train.get("loss")
    if loss is None or len(loss) < 5:
        return None
    zero = np.abs(loss.values) < 1e-6
    opening = float(np.median(np.abs(loss.values[: max(loss.values.size // 10, 3)])))
    if float(zero.mean()) >= 0.5 or opening < 1e-4:
        return Finding(
            "loss_degenerate",
            "Loss is exactly zero",
            Severity.CRITICAL,
            f"Training loss is ~0 from the first logged step ({_pct(float(zero.mean()))} of "
            "steps are exactly zero). A fine-tune that starts at zero loss is not training on "
            "anything.",
            evidence=[
                f"{int(zero.sum())} of {zero.size} logged losses are 0.0 (or -0.0); the median "
                f"over the first 10% of the run is {opening:.2g}"
            ],
            fix=[
                "Almost always every label is masked to -100. With completion-only training, "
                "check that the response template actually occurs in your tokenized examples -- "
                "a chat-template mismatch masks the entire sequence silently.",
                "Decode one batch's labels (replace -100 with the pad token) and look at what "
                "is left.",
            ],
        )
    return None


def not_learning(history: History) -> Optional[Finding]:
    loss = history.train.get("loss")
    if loss is None or len(loss) < 20:
        return None
    values = loss.values[loss.values > 0]
    if values.size < 20:
        return None
    head = float(np.median(values[: max(values.size // 10, 3)]))
    tail = float(np.median(values[-max(values.size // 10, 3) :]))
    drop = (head - tail) / max(abs(head), 1e-12)
    if drop >= 0.03:
        return None
    rose = drop <= -0.03
    evidence = [
        f"training loss, median of the first vs the last tenth of the run: {head:.4g} -> "
        f"{tail:.4g} ({'+' if rose else ''}{_pct(-drop)})"
    ]
    lr = history.train.get("learning_rate")
    if lr is not None and len(lr):
        peak = float(np.max(lr.values))
        note = " -- low for most fine-tunes" if peak < 5e-6 else ""
        evidence.append(f"peak learning rate {peak:.2g}{note}")
    epochs = history.meta.get("epoch")
    if rose and isinstance(epochs, (int, float)) and epochs >= 1.5:
        evidence.append(
            f"over {epochs:.3g} epochs of the same data the loss should fall at each epoch "
            "boundary; rising instead means harder examples come later in the data order, or "
            "the updates are making the model worse"
        )
    if head > 9:
        evidence.append(
            f"a loss near {head:.1f} is about ln(vocabulary size): the model is predicting "
            "uniformly, as if no pretrained weights were loaded"
        )
    return Finding(
        "not_learning",
        "Training loss went up" if rose else "Not learning",
        Severity.WARNING,
        (
            f"Training loss rose {_pct(-drop)} from the start of the run to the end."
            if rose
            else f"Training loss barely moved ({_pct(-drop)} from start to end). The run "
            "trained, but the model did not change."
        ),
        evidence=evidence,
        fix=[
            "Check the learning rate actually used (it is logged): typical LoRA rates are "
            "1e-4 to 3e-4, full fine-tuning 1e-5 to 5e-5.",
            "Check something is trainable: model.print_trainable_parameters() for PEFT, and that "
            "LoRA target_modules match your architecture's layer names.",
        ],
        metrics={"loss_drop": drop},
    )


# -- 6. instability ---------------------------------------------------------


def loss_spikes(history: History) -> Optional[Finding]:
    loss = history.train.get("loss")
    if loss is None or len(loss) < 15:
        return None
    baseline = _rolling_median(loss.values, max(loss.values.size // 20, 7))
    resid = loss.values - baseline
    z = stats.robust_z(resid)
    # 1.8x the local loss, not 1.5x: with small batches the logged loss swings
    # by 1.5x routinely, and on 89 public runs a 1.5x rule flagged ordinary
    # noise. Not 2x either: a real run's periodic jumps were 1.95-2.0x.
    candidate = (z > 6.0) & (loss.values > 1.8 * baseline) & (baseline > 1e-3)
    candidate[: _after_warmup(loss.values.size)] = False
    spikes = np.where(candidate)[0]
    if spikes.size == 0:
        return None

    periodic = _periodic_spikes(history, loss.steps[spikes])
    if periodic is not None:
        return periodic

    lr = history.train.get("learning_rate")
    evidence = []
    for i in spikes[:5]:
        step = int(loss.steps[i])
        line = f"step {step}: loss {loss.values[i]:.3g} against a local {baseline[i]:.3g}"
        if lr is not None and len(lr):
            j = int(np.argmin(np.abs(lr.steps - step)))
            line += f" (lr {lr.values[j]:.2g})"
        evidence.append(line)
    if spikes.size > 5:
        evidence.append(f"... and {spikes.size - 5} more")

    severity = Severity.WARNING if spikes.size >= 3 else Severity.INFO
    early = lr is not None and len(lr) and np.all(loss.steps[spikes] <= lr.steps[int(np.argmax(lr.values))] * 1.5)
    fix = []
    if early:
        fix.append(
            "The spikes sit around the learning-rate peak: add or lengthen warmup "
            "(warmup_ratio=0.03-0.1) or lower the peak learning rate."
        )
    fix += [
        "Keep gradient clipping on (max_grad_norm=1.0).",
        "Look at the batches at those steps: recurring spikes usually trace to a few bad "
        "examples -- empty targets, very long sequences, broken encoding.",
    ]
    return Finding(
        "loss_spikes",
        "Loss spikes",
        severity,
        f"{spikes.size} training-loss spike(s), each at least 1.8x the loss around it.",
        evidence=evidence,
        fix=fix,
        metrics={"n_spikes": int(spikes.size)},
    )


def _periodic_spikes(history: History, spike_steps: np.ndarray) -> Optional[Finding]:
    """Spikes at a fixed interval are an artefact of how the loss is computed,
    not instability -- and the usual instability advice is wrong for them.

    Found on a public Qwen2.5-7B SFT whose loss doubled on exactly the first
    logged step of every one of its 10 epochs.
    """
    if spike_steps.size < 4:
        return None
    gaps = np.diff(spike_steps)
    if gaps.mean() <= 0 or gaps.std() / gaps.mean() > 0.15:
        return None
    period = float(np.median(gaps))

    at_boundary = 0
    epoch = history.train.get("epoch")
    if epoch is not None and len(epoch) >= 2:
        for step in spike_steps:
            i = int(np.argmin(np.abs(epoch.steps - step)))
            if i > 0 and np.floor(epoch.values[i]) > np.floor(epoch.values[i - 1]):
                at_boundary += 1
    boundary = at_boundary >= 0.75 * spike_steps.size

    if boundary:
        summary = (
            f"The training loss jumps on the first logged step of every epoch ({spike_steps.size} "
            f"times, every {period:.0f} steps). Periodic, so not instability: the step that "
            "straddles an epoch boundary is computing its loss differently."
        )
        fix = [
            "Most often a gradient-accumulation window that spans the epoch boundary, so the "
            "loss is summed over more micro-batches than it is divided by. Recent transformers "
            "versions normalise this correctly; check yours, and any custom compute_loss that "
            "ignores num_items_in_batch.",
            "Otherwise the last batch of each epoch is unlike the others (a short or padded "
            "final batch): set dataloader_drop_last=True and see whether the jumps go away.",
            "Lowering the learning rate will not help -- this is not that kind of spike.",
        ]
    else:
        summary = (
            f"The training loss jumps every {period:.0f} steps ({spike_steps.size} times). "
            "Periodic, so not instability: something recurs on that schedule."
        )
        fix = [
            f"Look for anything with a period of {period:.0f} steps: an evaluation or "
            "checkpoint save that changes model state, a data shard boundary, or a recurring "
            "very long example.",
            "Lowering the learning rate will not help -- this is not that kind of spike.",
        ]
    return Finding(
        "loss_spikes",
        "Loss jumps at every epoch boundary" if boundary else "Periodic loss spikes",
        Severity.WARNING,
        summary,
        evidence=[
            "spike steps: " + ", ".join(str(int(s)) for s in spike_steps[:10])
            + (" ..." if spike_steps.size > 10 else ""),
            f"interval {period:.0f} steps"
            + (f"; {at_boundary} of {spike_steps.size} fall on an epoch boundary" if boundary else ""),
        ],
        fix=fix,
        metrics={"n_spikes": int(spike_steps.size), "period": period, "epoch_boundary": boundary},
    )


def lr_schedule(history: History) -> Optional[Finding]:
    lr = history.train.get("learning_rate")
    if lr is None or len(lr) < 20:
        return None
    peak = float(np.max(lr.values))
    if peak <= 0:
        return None
    near_zero = lr.values <= 1e-3 * peak
    # Count the dead tail: consecutive near-zero values at the end of the run.
    tail = 0
    for flag in near_zero[::-1]:
        if not flag:
            break
        tail += 1
    if tail < 2:
        # A linear or cosine schedule reaches ~0 at its very last step by design.
        return None
    start = int(lr.steps[-tail])
    frac = (history.final_step - start) / max(history.final_step, 1)
    if frac < 0.05:
        return None
    severity = Severity.WARNING if frac >= 0.10 else Severity.INFO
    return Finding(
        "lr_schedule",
        "Training at zero learning rate",
        severity,
        f"The learning rate reached ~0 at step {start} and stayed there: the last {_pct(frac)} of "
        "steps ran forward and backward passes that changed nothing.",
        evidence=[
            f"peak learning rate {peak:.2g}; {tail} logged steps at or below {1e-3 * peak:.2g}"
            + _hours(history, frac)
        ],
        fix=[
            "The schedule is shorter than the run. This happens when max_steps and "
            "num_train_epochs disagree, or when training is resumed with a fresh step budget. "
            "Set one of them, not both.",
        ],
        metrics={"dead_tail_frac": frac},
    )


def gradients(history: History) -> Optional[Finding]:
    bad = history.nonfinite.get("grad_norm", [])
    grad = history.train.get("grad_norm")
    if bad:
        n_logged = len(bad) + (len(grad) if grad is not None else 0)
        frac = len(bad) / max(n_logged, 1)
        at_end = grad is None or not len(grad) or max(bad) > float(grad.steps[-1])
        loss_died = bool(history.nonfinite.get("loss"))
        if frac < 0.01 and not at_end and not loss_died:
            # Under fp16 mixed precision the loss scaler skips any step whose
            # gradients overflow, logs inf, lowers the scale and carries on.
            # Sporadic infs with a healthy loss are that mechanism working.
            return Finding(
                "gradients",
                "Occasional overflowing gradients",
                Severity.INFO,
                f"Gradient norm was inf/NaN at {len(bad)} scattered step(s) out of {n_logged}, "
                "and training carried on normally. Under fp16 that is the loss scaler skipping "
                "an overflowing step, which is what it is for.",
                evidence=[f"first at step {min(bad)}, last at step {max(bad)}; loss stayed finite"],
                fix=["Nothing to do. bf16, where the hardware supports it, avoids the skips."],
                metrics={"nonfinite_grad_frac": frac},
            )
        return Finding(
            "gradients",
            "NaN gradients",
            Severity.CRITICAL,
            f"Gradient norm was NaN or inf at {len(bad)} step(s), first at step {min(bad)}"
            + (", and it never recovered." if at_end else "."),
            fix=[
                "Resume from a checkpoint before that step, with bf16 (not fp16), a lower "
                "learning rate, and max_grad_norm=1.0.",
            ],
        )
    if grad is None or len(grad) < 10:
        return None
    zero = float(np.mean(grad.values == 0.0))
    if zero >= 0.2:
        return Finding(
            "gradients",
            "Zero gradients",
            Severity.CRITICAL if zero >= 0.7 else Severity.WARNING,
            f"The gradient norm is exactly 0.0 on {_pct(zero)} of logged steps: those updates "
            "moved nothing.",
            fix=[
                "Exactly-zero gradients mean no trainable parameter received a gradient: check "
                "that labels are not all -100, and for PEFT that the adapter is attached "
                "(model.print_trainable_parameters()).",
            ],
            metrics={"zero_grad_frac": zero},
        )
    # Against the *local* level, not the global median: gradient norms fall over
    # a run, so early values look like outliers against a late-run median.
    baseline = _rolling_median(grad.values, max(grad.values.size // 20, 7))
    z = stats.robust_z(grad.values - baseline)
    candidate = (z > 8.0) & (grad.values > 10 * np.maximum(baseline, 1e-12))
    candidate[: _after_warmup(grad.values.size)] = False
    spikes = np.where(candidate)[0]
    if spikes.size >= 3:
        worst = int(spikes[np.argmax(grad.values[spikes])])
        # The Trainer clips to max_grad_norm=1.0 by default, so a gradient spike
        # on its own is usually absorbed. It matters when the loss jumps with it.
        loss_spike = loss_spikes(history)
        coupled = False
        if loss_spike is not None:
            loss = history.train["loss"]
            spike_steps = grad.steps[spikes]
            gap = float(np.median(np.diff(loss.steps))) * 2 if len(loss) > 1 else 0.0
            for line in loss_spike.evidence:
                if line.startswith("step "):
                    step = float(line.split(":", 1)[0].split()[1])
                    if np.any(np.abs(spike_steps - step) <= gap):
                        coupled = True
        return Finding(
            "gradients",
            "Gradient spikes",
            Severity.WARNING if coupled else Severity.INFO,
            f"{spikes.size} gradient-norm spikes over 10x the local level; the worst is "
            f"{grad.values[worst]:.3g} at step {int(grad.steps[worst])} against "
            f"{baseline[worst]:.3g} around it.",
            evidence=[
                "the training loss jumped at the same steps, so these were not absorbed by "
                "gradient clipping"
                if coupled
                else "the loss did not jump with them, so gradient clipping (max_grad_norm, 1.0 "
                "by default) absorbed them"
            ],
            fix=["Keep max_grad_norm=1.0 and check the batches at those steps."],
            metrics={"n_grad_spikes": int(spikes.size), "coupled_to_loss": coupled},
        )
    return None


# -- 7. can the eval be trusted? --------------------------------------------


def eval_below_train(history: History) -> Optional[Finding]:
    eval_loss = history.evals.get("eval_loss")
    if eval_loss is None or len(eval_loss) < 4 or not history.has_train("loss", 10):
        return None
    ratios = []
    for step, value in zip(eval_loss.steps, eval_loss.values, strict=True):
        train = _train_loss_near(history, float(step))
        if train and train > 0:
            ratios.append(value / train)
    if len(ratios) < 4:
        return None
    # Skip the first eval: early on, training loss averages over a moving model.
    ratio = float(np.median(ratios[1:]))
    if ratio >= 0.8:
        return None
    return Finding(
        "eval_below_train",
        "Eval loss below training loss",
        Severity.INFO,
        f"Eval loss sits at about {ratio:.0%} of training loss throughout. Worth knowing why "
        "before you trust the eval numbers.",
        evidence=[f"median eval/train loss ratio over {len(ratios) - 1} evals: {ratio:.2f}"],
        fix=[
            "Benign causes: dropout or regularisation active only during training, label "
            "smoothing, or a training mix that is harder than the eval set.",
            "The one to rule out: eval examples that also appear in training. Deduplicate "
            "(exact and near-duplicate) between the splits.",
        ],
        metrics={"eval_train_ratio": ratio},
    )


def incomplete(history: History) -> Optional[Finding]:
    max_steps = history.max_steps
    final = history.final_step
    if not max_steps or final >= 0.98 * max_steps:
        return None
    frac = final / max_steps
    still = history.meta.get("should_training_stop") is False
    return Finding(
        "incomplete",
        "Run did not finish",
        Severity.INFO,
        f"The log stops at step {final} of {max_steps} ({_pct(frac)}). "
        + (
            "This is an intermediate checkpoint's copy of the history -- the run may still "
            "be going."
            if still
            else "It was stopped, crashed, or early-stopped."
        ),
        fix=[
            "If it crashed, resume with trainer.train(resume_from_checkpoint=True). If it is "
            "still running, run ftdoctor again at the end.",
        ],
        metrics={"completed_frac": frac},
    )
