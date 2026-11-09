## What this change does

## Checklist

- [ ] `ruff check src tests scripts` and `ruff format --check src tests scripts` are clean
- [ ] `python -m ai_workload_platform openapi --check` passes
- [ ] `python -m pytest -q` passes
- [ ] `python -m ai_workload_platform bench --quick` runs
- [ ] Tests added or updated for the changed behavior
- [ ] If the change affects results: the affected suite re-run, result files and manifests committed
- [ ] Documentation updated; every new number names its result file
