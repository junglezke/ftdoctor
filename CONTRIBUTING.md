# Contributing to ftdoctor

## The most valuable thing you can send

**A run where ftdoctor is wrong.** It flagged a fine run, missed a real problem,
or recommended the wrong checkpoint. A public Hub model id is enough -- or attach
the `trainer_state.json`. There is an issue template for it.

Every threshold here was set against public runs, and several were changed
because one of them proved the first version wrong. That is the process
working, and it only works with more runs.

## Changing a threshold

Run the real-run validation before and after:

```bash
python validation/fetch.py
python validation/run.py --check
```

Unit tests show a check *can* fire. The public runs in `validation/runs.json`
show whether it fires on fine-tunes that exist, and -- as importantly -- that it
stays quiet on the ones that are fine. If a verdict moves, either the new verdict
is right (update `runs.json` and say why in the commit) or the change broke
something real.

## Adding a check

A function in `src/ftdoctor/checks.py` that returns a `Finding` or `None`, and an
entry in `run_all`. Return `None` when there is nothing to say: a report full of
green ticks buries the line that matters. Put the numbers you fired on in
`evidence` and a concrete change in `fix`.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests validation tools
```

Keep the only dependency `numpy`. This runs on login nodes and in training
images, next to whatever they already pin.
