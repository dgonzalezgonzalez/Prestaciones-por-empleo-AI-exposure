# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python data pipeline for Spanish employment and AI exposure analysis. Core reusable code lives in `src/`: data downloads, taxonomy parsing, embeddings, exposure models, aggregation, database output, and SEPE parsing. Operational scripts live in `scripts/`, including SEPE dataset construction and econometric runs. Tests live in `tests/` and follow `test_*.py` naming. Large or generated assets are organized under `data/`, `models/`, and `analysis/`; avoid committing regenerated large files unless they are intentionally tracked outputs.

## Build, Test, and Development Commands

Install dependencies:

```powershell
py -3 -m pip install -r requirements.txt
```

Run full test suite:

```powershell
py -3 -m unittest discover -s tests -v
```

Build the SEPE CNO4 monthly dataset from cached reports:

```powershell
py -3 scripts/build_sepe_occupation_dataset.py --embedding-model qwen3-embedding:4b --from-cache --workers 8
```

Smoke-test SEPE parsing with one report:

```powershell
py -3 scripts/build_sepe_occupation_dataset.py --embedding-model qwen3-embedding:4b --max-occupations 1 --max-reports 1
```

Run econometric outputs:

```powershell
py -3 scripts/run_ai_exposure_econometrics.py
```

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation, type hints where helpful, and clear snake_case names for functions, variables, and modules. Keep parsing and data-shaping logic in `src/`; keep command-line orchestration in `scripts/`. Prefer `pathlib.Path`, pandas transformations, and structured parsers over ad hoc string handling. No project formatter is configured, so keep diffs focused and match surrounding style.

## Testing Guidelines

Tests use the standard library `unittest` framework. Add focused regression tests in `tests/test_<module>.py` for parser changes, data-shape assumptions, and edge cases. Prefer small synthetic fixtures embedded in tests over large raw files. Before committing, run at least the affected test file; for broader changes, run the full discovery command above.

## Commit & Pull Request Guidelines

Recent history uses short imperative messages, sometimes with conventional prefixes, for example `fix SEPE banner-only total parsing`, `feat: add SEPE econometric event studies`, and `chore: rerun DiD analyses with 100 bootstrap reps`. Keep commits scoped to one logical change. Pull requests should describe the data or code path changed, list validation commands run, mention regenerated outputs, and link any related issue or analysis note. Include screenshots only for visual artifacts.

## Security & Configuration Tips

Do not commit secrets, local credentials, or private API keys. Treat `data/raw/`, large processed CSVs, caches, and model files as heavyweight artifacts; confirm `.gitignore` expectations before adding them. Network-dependent scripts should support cached or resumable workflows when possible.
