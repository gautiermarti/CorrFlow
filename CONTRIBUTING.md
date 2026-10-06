# Contributing

Start with the README's synthetic example. Use Python 3.12 or 3.13 and an isolated
virtual environment. Model development uses a compatible `corrbench` checkout
beside this repository; public installs use its exact Git revision pin.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ../corrbench -e '.[dev,diffeocfm,notebooks]'
python -m pytest -q
ruff check .
python -m build
```

The test suite uses small synthetic fixtures and runs on CPU. TensorFlow and real
research datasets are not required. GPU throughput, full retraining and original
checkpoint parity require the separately documented environments and artifacts.

## Report an issue

Include a minimal example, expected and actual behavior, the full traceback,
package revisions, Python/PyTorch versions, OS/device and random seed. Use the
bug-report template and synthetic inputs. For installation errors, include the
command and dependency resolver output. Check `python -m pip check` and ensure
that Python and pip belong to the same environment. A missing CUDA driver means
you should select `--device cpu` or install a matching PyTorch build.

Do not attach financial source archives, participant records, credentials or
untrusted serialized objects. See [SECURITY.md](SECURITY.md).

## Propose a change

Open a focused branch and pull request describing the concrete behavior change,
why it is needed and how it was checked. Add a regression test for a bug or a
meaningful new behavior; documentation-only changes do not need artificial tests.
Keep public examples executable and update the changelog when behavior changes.
Use clear, respectful discussion and give original authors credit.

Scientific comparisons must record source/split hashes, independent units,
conditions, preprocessing, model configuration, validation-only selection,
all declared seeds, raw and repaired validity, runtime and complete metric reports.
Retain unfavorable outcomes and disclose prior test inspection. New protocols
remain separate from the frozen paper protocol. A submitted result is not an
automatically verified leaderboard entry.

## Reproduce and release

State what you actually ran: installation, tutorial, table aggregation, sampling
or retraining. Recomputing a bundled summary does not establish independent
retraining or an external-person replication. Report missing files and deviations.

A coordinated release builds and installs both wheels and source archives,
checks the installed code against the export, exercises examples/CLI entry points,
and tests the exact CorrBench pin before tagging. CI runs Python 3.12 and 3.13
on CPU. A green local audit is distinct from a completed GitHub Actions run.
