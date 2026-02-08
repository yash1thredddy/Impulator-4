# External Diagnostics

These scripts validate live third-party APIs and are intentionally kept outside
the `tests/` tree.

They are **not** part of deterministic CI test runs.

## Scripts

- `scripts/diagnostics/test_external_apis.py`
- `scripts/diagnostics/test_pubchem_api.py`

## Usage

```bash
python scripts/diagnostics/test_external_apis.py --help
python scripts/diagnostics/test_pubchem_api.py --help
```
