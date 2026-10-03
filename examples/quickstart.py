"""Diagnose a run from Python.

    python examples/quickstart.py ./outputs
    python examples/quickstart.py owner/model-on-the-hub
"""

import sys

from ftdoctor import Severity, diagnose, load

target = sys.argv[1] if len(sys.argv) > 1 else "anshpunia8597/mistral-7b-medical-qa-finetuned"
result = diagnose(load(target))

print(result.headline)
if result.pick is not None:
    p = result.pick
    print(f"best {p.selection.label}: {p.best_value:.4g} at step {p.best_step}")
    print(f"checkpoints within noise of it: {p.equivalent_steps}")
    if p.keep_step is not None:
        print(f"keep: checkpoint-{p.keep_step}")

for finding in result.findings:
    if finding.severity >= Severity.WARNING:
        print(f"[{finding.severity.label}] {finding.title}: {finding.fix[0] if finding.fix else ''}")
