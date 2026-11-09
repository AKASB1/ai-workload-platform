# Contributing

This is a portfolio repository with one maintainer. Issues and pull requests are welcome; this page lists what a change needs before it is merged.

## Environment

- Python 3.12 (`pyproject.toml`). Install the package with its development tools: `python -m pip install -e ".[dev]"`.
- PostgreSQL is optional. Set `AWP_PG_DSN` to run the PostgreSQL tests; without it they are skipped with a message.
- A kind cluster is optional. Set `AWP_KUBECONFIG` to run the tests against it; without it they are skipped with a message. `deploy/README.md` sets the cluster up.
- The stack: FastAPI, Pydantic, SQLite, PostgreSQL (store contract tests and `docker compose`), prometheus-client, the official Kubernetes Python client, Docker, and kind.

## Checks

Run these before opening a pull request. CI runs the same steps (`.github/workflows/ci.yml`):

```bash
ruff check src tests scripts
ruff format --check src tests scripts
python -m ai_workload_platform openapi --check
python -m pytest -q
python -m ai_workload_platform bench --quick --workers 2 --out outputs/ci-results
```

CI also validates the Kubernetes manifests in `deploy/k8s/` with `kubeconform -strict`.

With PostgreSQL and a kind cluster, `pytest` runs 170 tests and skips 2 by design: node loss on a real cluster is a separate demonstration, and the local backend has no `AlreadyExists`.

## Results

The results in the README and in `benchmarks/` come from the committed result files. A few runs are described without one: the in-cluster run, the realized offered load, the 12-worker repeat, and the first full evaluation at `012db44`. The workloads sleep for their run time; the kind timings are wall-clock measurements on one machine. To regenerate the virtual-clock results and their figures:

```bash
python -m ai_workload_platform bench --full
python scripts/plot_results.py --suite full
python scripts/plot_timeline.py --suite full
```

The kind runs, the demonstrations, and the conformance check have their own commands in [benchmarks/README.md](benchmarks/README.md).

A change to code or configuration that the evaluation reads (`src/ai_workload_platform/`, `configs/clusters/`) can change results. Run the affected suite once on the committed change, regenerate the figures and tables, and commit the result files with their manifests in the same pull request. The README's highlights and tables are written by hand: update them from `benchmarks/results/full/table.md` and the suite's CSV files.

## Figures and diagrams

`python scripts/plot_results.py --suite full` redraws the result figures in `docs/figures/` and `benchmarks/results/full/table.md`; `python scripts/plot_timeline.py --suite full` redraws the timeline. The architecture diagram is drawn from `docs/diagrams/architecture.d2` with [D2](https://d2lang.com) (`bash scripts/render_diagrams.sh` or `powershell -ExecutionPolicy Bypass -File scripts\render_diagrams.ps1`); the committed SVG was drawn with D2 0.9.0. `python scripts/make_assets.py` writes the icon. CI needs none of these.

## Style

- The domain code (models, lifecycle, admission, the policy interface and built-ins, the controller rules) imports no web framework and no database driver. It never reads the wall clock: it receives `now_ms` or a `Clock`.
- Every store mutation is one transaction that writes the versioned rows and one event; the event log must replay to the same state.
- Documentation states what was measured, under which assumptions, and names the result file.
