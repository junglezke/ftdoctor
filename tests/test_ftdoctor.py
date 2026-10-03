"""Tests for ftdoctor.

The checks were calibrated against 89 public fine-tuning logs. Several tests
here pin a shape found in one of them, and say which.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
from helpers import make_output_dir, make_state

from ftdoctor import Severity, diagnose, hub, load, metrics, picker, prune
from ftdoctor.cli import main
from ftdoctor.history import from_trainer_state
from ftdoctor.report import render


def _diag(state, **kw):
    return diagnose(from_trainer_state(state, name="run"), **kw)


def _checks(d, at_least=Severity.INFO):
    return {f.check: f for f in d.findings if f.severity >= at_least}


# -- parsing -----------------------------------------------------------------


def test_train_and_eval_series_are_separated():
    h = from_trainer_state(make_state())
    assert {"loss", "learning_rate", "grad_norm"} <= set(h.train)
    assert "eval_loss" in h.evals
    # timing fields are bookkeeping, never metrics
    assert "eval_runtime" not in h.evals and "eval_samples_per_second" not in h.evals


def test_the_closing_summary_row_is_not_a_point_on_a_curve():
    h = from_trainer_state(make_state())
    assert "train_runtime" not in h.train
    assert h.meta["train_runtime"] == 3600.0


def test_non_finite_values_are_recorded_apart_from_the_series():
    state = make_state(extra_train={500: {"grad_norm": float("inf")}})
    h = from_trainer_state(state)
    assert h.nonfinite["grad_norm"] == [500]
    assert np.all(np.isfinite(h.train["grad_norm"].values))


def test_not_a_trainer_state_is_rejected():
    with pytest.raises(ValueError, match="log_history"):
        from_trainer_state({"foo": 1})


# -- loading from disk ----------------------------------------------------------


def test_output_dir_uses_the_most_complete_history_and_lists_real_checkpoints(tmp_path):
    state = make_state(overfit_after=400)
    root = make_output_dir(str(tmp_path / "out"), state, [300, 400, 500, 1000])
    os.remove(os.path.join(root, "trainer_state.json"))  # only checkpoint copies remain
    h = load(root)
    assert h.final_step == 1000
    assert h.checkpoint_steps() == [300, 400, 500, 1000]
    c = h.checkpoints[0]
    assert c.size_bytes == 4000 + 8000 + 100 + 50 + os.path.getsize(os.path.join(c.path, "trainer_state.json"))
    assert c.resume_only_bytes == 8000 + 100 + 50


def test_a_single_checkpoint_dir_diagnoses_its_run(tmp_path):
    state = make_state()
    root = make_output_dir(str(tmp_path / "out"), state, [500, 1000])
    h = load(os.path.join(root, "checkpoint-500"))
    assert h.checkpoint_steps() == [500, 1000]


def test_missing_target_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(str(tmp_path / "nope" / "deeper"))


# -- which metric ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("eval_loss", -1), ("eval_wer", -1), ("eval_cer_ortho", -1), ("eval_accuracy", 1),
        ("eval_mean_token_accuracy", 1), ("eval_bleu", 1), ("eval_chrf++", 1), ("eval_f1", 1),
        ("eval_rouge1", 1), ("eval_entropy", None),
    ],
)
def test_metric_direction(name, expected):
    assert metrics.direction(name) == expected


def test_the_trainers_own_selection_metric_is_recovered_from_best_metric():
    """A public audio classifier selected on accuracy; trainer_state records the
    value but not the name. Getting this wrong calls its best model overfit."""
    state = make_state(metric="accuracy", best_metric_key="eval_accuracy", overfit_after=300)
    sel = metrics.choose(from_trainer_state(state))
    assert sel.key == "eval_accuracy" and sel.sign == 1
    assert "best_metric" in sel.reason


def test_eval_loss_is_the_default():
    assert metrics.choose(from_trainer_state(make_state())).key == "eval_loss"


def test_override_and_unknown_direction():
    h = from_trainer_state(make_state(metric="accuracy"))
    assert metrics.choose(h, override="accuracy").sign == 1
    with pytest.raises(KeyError, match="not an eval metric"):
        metrics.choose(h, override="bleu")


# -- the pick -------------------------------------------------------------------


def test_noise_band_treats_jitter_as_equivalent():
    state = make_state(n_steps=1000, eval_every=100, eval_noise=0.0005)
    d = _diag(state)
    assert d.pick.final_is_equivalent
    assert "overfitting" not in _checks(d)


def test_real_overfitting_is_outside_the_band():
    d = _diag(make_state(overfit_after=400))
    assert not d.pick.final_is_equivalent
    assert abs(d.pick.best_step - 400) <= 100
    f = _checks(d)["overfitting"]
    assert f.severity >= Severity.WARNING
    assert "~1.2 epochs instead of 3" in " ".join(f.fix)


def test_the_pick_only_recommends_checkpoints_that_exist(tmp_path):
    state = make_state(overfit_after=400)
    root = make_output_dir(str(tmp_path / "out"), state, [300, 500, 1000])  # 400 was never saved
    d = diagnose(load(root))
    assert d.pick.keep_step in (300, 500)
    assert d.pick.unsaved_best
    assert "never saved" in " ".join(_checks(d)["overfitting"].fix)


def test_without_a_checkpoint_list_the_report_says_so():
    d = _diag(make_state(overfit_after=400))
    assert d.pick.keep_step is None and not d.pick.checkpoints_known
    assert "does not say which checkpoints still exist" in " ".join(_checks(d)["overfitting"].fix)


def test_overfitting_severity_tiers():
    """On 89 public runs, a band-only rule called 1% drops 'overfitting'."""
    small = make_state(overfit_after=900, eval_noise=0.0002)
    d = _diag(small)
    f = _checks(d).get("overfitting")
    if f is not None:
        assert f.severity <= Severity.WARNING
        if d.pick.final_gap_rel < 0.03:
            assert f.severity is Severity.INFO and f.title == "Final checkpoint slightly worse"


# -- the checks ------------------------------------------------------------------


def test_loss_and_task_metric_disagreeing_is_not_called_overfitting():
    """The audio-classifier case: eval_loss up 37%, accuracy still improving."""
    state = make_state(metric="accuracy", best_metric_key="eval_accuracy", overfit_after=300)
    d = _diag(state)
    found = _checks(d)
    assert "overfitting" not in found
    assert "metric_disagreement" in found
    assert d.pick.final_is_equivalent


def test_no_eval_warns_with_concrete_settings():
    state = make_state(eval_every=0)
    d = _diag(state)
    f = _checks(d)["no_eval"]
    assert f.severity is Severity.WARNING
    assert "load_best_model_at_end=True" in " ".join(f.fix)
    assert d.pick is None


def test_loss_stuck_at_zero_means_masked_labels():
    """A public gemma fine-tune logged 0.0 / -0.0 from its first step."""
    state = make_state(loss_fn=lambda s: -0.0 if s % 3 else 0.0)
    f = _checks(_diag(state))["loss_degenerate"]
    assert f.severity is Severity.CRITICAL
    assert "-100" in " ".join(f.fix)


def test_loss_going_up_says_so():
    """A public Llama SFT whose loss rose 9% over three epochs at lr 1e-6."""
    state = make_state(loss_fn=lambda s: 0.6 + 0.08 * s / 1000, lr_peak=1e-6)
    f = _checks(_diag(state))["not_learning"]
    assert f.title == "Training loss went up"
    assert "low for most fine-tunes" in " ".join(f.evidence)


def test_sporadic_fp16_overflow_is_not_a_crisis():
    """fp16 loss scaling logs inf and skips the step. Telling someone to roll
    back over that is wrong advice."""
    extra = {s: {"grad_norm": float("inf")} for s in (200, 610)}
    state = make_state(n_steps=4000, extra_train=extra)
    f = _checks(_diag(state))["gradients"]
    assert f.severity is Severity.INFO
    assert "loss scaler" in f.summary


def test_persistent_nan_gradients_are_critical():
    extra = {s: {"grad_norm": float("nan")} for s in range(900, 1001, 10)}
    f = _checks(_diag(make_state(extra_train=extra)))["gradients"]
    assert f.severity is Severity.CRITICAL


def test_opening_steps_are_not_spikes():
    """Large gradients in the first few steps are normal; flagging them fired
    on several public runs."""
    grads = {10: 300.0, 20: 120.0, 30: 50.0}
    state = make_state(grad_fn=lambda s: grads.get(s, 0.5))
    assert "gradients" not in _checks(_diag(state))


def test_real_loss_spikes_fire():
    spikes = {400: 5.0, 600: 5.0, 800: 5.0}
    state = make_state(loss_fn=lambda s: spikes.get(s, 0.5 + 0.0001 * (s % 7)))
    f = _checks(_diag(state))["loss_spikes"]
    assert f.metrics["n_spikes"] == 3


def test_schedule_ending_at_zero_on_the_last_step_is_fine():
    assert "lr_schedule" not in _checks(_diag(make_state()))


def test_a_long_dead_tail_is_flagged():
    state = make_state()
    for e in state["log_history"]:
        if "learning_rate" in e and e["step"] > 700:
            e["learning_rate"] = 0.0
    f = _checks(_diag(state))["lr_schedule"]
    assert f.severity is Severity.WARNING
    assert f.metrics["dead_tail_frac"] > 0.25


def test_incomplete_run():
    state = make_state()
    state["max_steps"] = 4000
    state["stateful_callbacks"]["TrainerControl"]["args"]["should_training_stop"] = False
    f = _checks(_diag(state))["incomplete"]
    assert "intermediate checkpoint" in f.summary


# -- prune -----------------------------------------------------------------------


def test_prune_resume_state_keeps_weights_and_the_kept_checkpoints(tmp_path):
    state = make_state(overfit_after=400)
    root = make_output_dir(str(tmp_path / "out"), state, [300, 400, 500, 1000])
    d = diagnose(load(root))
    plan = prune.plan(d)
    assert set(plan.keep) == {400, 1000}
    assert {a.checkpoint.step for a in plan.actions} == {300, 500}
    freed, _ = prune.apply(plan)
    assert freed == 2 * (8000 + 100 + 50)
    for step in (300, 500):
        ckpt = os.path.join(root, f"checkpoint-{step}")
        assert os.path.exists(os.path.join(ckpt, "model.safetensors"))
        assert not os.path.exists(os.path.join(ckpt, "optimizer.pt"))
    assert os.path.exists(os.path.join(root, "checkpoint-400", "optimizer.pt"))


def test_prune_full_deletes_whole_checkpoints(tmp_path):
    state = make_state(overfit_after=400)
    root = make_output_dir(str(tmp_path / "out"), state, [300, 400, 500, 1000])
    plan = prune.plan(diagnose(load(root)), mode="full")
    prune.apply(plan)
    assert sorted(os.listdir(root)) == ["checkpoint-1000", "checkpoint-400", "trainer_state.json"]


def test_prune_full_is_refused_without_an_eval(tmp_path):
    state = make_state(eval_every=0)
    root = make_output_dir(str(tmp_path / "out"), state, [500, 1000])
    plan = prune.plan(diagnose(load(root)), mode="full")
    assert plan.refused and "resume-state" in plan.refused
    with pytest.raises(RuntimeError):
        prune.apply(plan)


def test_prune_refuses_anything_that_is_not_a_trainer_checkpoint(tmp_path):
    state = make_state(overfit_after=400)
    root = make_output_dir(str(tmp_path / "out"), state, [300, 400, 1000])
    plan = prune.plan(diagnose(load(root)), mode="full")
    os.remove(os.path.join(root, "checkpoint-300", "trainer_state.json"))
    with pytest.raises(RuntimeError, match="not a Trainer checkpoint"):
        prune.apply(plan)
    assert os.path.isdir(os.path.join(root, "checkpoint-300"))


# -- hub (no network) ------------------------------------------------------------


@pytest.mark.parametrize(
    "target, expected",
    [("owner/model", "owner/model"), ("hf://owner/model", "owner/model"),
     ("https://huggingface.co/owner/model/tree/main", "owner/model")],
)
def test_hub_normalise(target, expected):
    assert hub.normalise(target) == expected
    assert hub.looks_like_repo_id(target)


def test_hub_prefers_the_root_state_then_the_highest_checkpoint():
    files = [{"rfilename": n} for n in ("checkpoint-100/trainer_state.json",
                                        "checkpoint-900/trainer_state.json", "README.md")]
    assert hub.choose_state_file(files) == "checkpoint-900/trainer_state.json"
    files.append({"rfilename": "trainer_state.json"})
    assert hub.choose_state_file(files) == "trainer_state.json"


def test_hub_checkpoint_sizes():
    files = [
        {"rfilename": "checkpoint-100/model.safetensors", "size": 1000},
        {"rfilename": "checkpoint-100/optimizer.pt", "size": 2000},
        {"rfilename": "checkpoint-200/model.safetensors", "lfs": {"size": 1000}},
    ]
    found = {c.step: c for c in hub.hub_checkpoints(files)}
    assert found[100].size_bytes == 3000 and found[100].resume_only_bytes == 2000
    assert found[200].size_bytes == 1000 and found[200].where == "hub"


def test_hub_load_end_to_end(monkeypatch):
    state = make_state(overfit_after=400)
    listing = {"siblings": [{"rfilename": "trainer_state.json", "size": 10},
                            {"rfilename": "checkpoint-400/model.safetensors", "size": 5}]}

    def fake(url, timeout=60):
        return json.dumps(listing if "/api/models/" in url else state).encode()

    monkeypatch.setattr(hub, "_request", fake)
    d = diagnose(hub.load("someone/model"))
    assert d.history.source == "hf://someone/model/trainer_state.json"
    assert d.pick.keep_step == 400


# -- reports and CLI ---------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["terminal", "markdown", "json"])
def test_every_format_renders(fmt, tmp_path):
    root = make_output_dir(str(tmp_path / "out"), make_state(overfit_after=400), [400, 1000])
    text = render(diagnose(load(root)), fmt)
    assert len(text) > 200


def test_terminal_report_leads_with_the_checkpoint_to_keep(tmp_path):
    root = make_output_dir(str(tmp_path / "out"), make_state(overfit_after=400), [400, 1000])
    text = render(diagnose(load(root)), "terminal")
    keep_at = text.index("KEEP checkpoint-400")
    assert keep_at < text.index("FINDINGS")


def test_cli_shorthand_and_json(tmp_path, capsys):
    root = make_output_dir(str(tmp_path / "out"), make_state(overfit_after=400), [400, 1000])
    assert main([root, "-f", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pick"]["keep_step"] == 400


def test_cli_fail_on(tmp_path, capsys):
    root = make_output_dir(str(tmp_path / "out"), make_state(overfit_after=300), [300, 1000])
    assert main([root, "-f", "json", "--fail-on", "warning"]) == 1


def test_cli_prune_is_a_dry_run_by_default(tmp_path, capsys):
    root = make_output_dir(str(tmp_path / "out"), make_state(overfit_after=400), [300, 400, 1000])
    assert main(["prune", root]) == 0
    assert "dry run" in capsys.readouterr().out
    assert os.path.exists(os.path.join(root, "checkpoint-300", "optimizer.pt"))
    assert main(["prune", root, "--apply"]) == 0
    assert not os.path.exists(os.path.join(root, "checkpoint-300", "optimizer.pt"))


def test_cli_bad_target(capsys):
    assert main(["no_such_trainer_state.json"]) == 2
    assert "ftdoctor:" in capsys.readouterr().err


def test_headline_is_one_line(tmp_path):
    root = make_output_dir(str(tmp_path / "out"), make_state(overfit_after=400), [400, 1000])
    assert "\n" not in diagnose(load(root)).headline


def test_math_sanity():
    assert math.isclose(picker.REL_FLOOR, 0.01)


def test_loss_jumps_at_every_epoch_boundary_are_not_called_instability():
    """A public Qwen2.5-7B SFT doubled its loss on the first step of each of its
    10 epochs. 'Lower the learning rate' would have been the wrong advice."""
    n, epochs = 700, 10.0

    def loss(step):
        crossed = int(epochs * step / n) > int(epochs * (step - 10) / n)
        return 3.9 if (crossed and step > 10) else 1.95

    f = _checks(_diag(make_state(n_steps=n, epochs=epochs, loss_fn=loss)))["loss_spikes"]
    assert f.title == "Loss jumps at every epoch boundary"
    assert f.metrics["epoch_boundary"] is True
    assert "will not help" in " ".join(f.fix)


def test_rl_runs_are_recognised_and_not_misread():
    """A public GRPO run has loss 0.0 from step 1 -- normal for GRPO, and it was
    once diagnosed as 'trained on nothing, labels masked'."""
    extra = {s: {"reward": 0.6, "frac_reward_zero_std": 0.5, "kl": 0.0} for s in range(10, 1001, 10)}
    state = make_state(loss_fn=lambda s: 0.0, extra_train=extra)
    found = _checks(_diag(state))
    assert "rl_run" in found
    assert "loss_degenerate" not in found and "no_eval" not in found
    assert "rldoctor" in " ".join(found["rl_run"].fix)


def test_dpo_is_not_mistaken_for_rl():
    extra = {s: {"rewards/chosen": 0.4, "rewards/rejected": -0.2, "rewards/margins": 0.6}
             for s in range(10, 1001, 10)}
    assert not from_trainer_state(make_state(extra_train=extra)).is_rl
