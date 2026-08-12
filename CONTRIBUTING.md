# Contributing

Thanks for taking a look.

## Setup

```bash
git clone https://github.com/RenatMursalimov/llm-search-replace
cd llm-search-replace
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest
```

## Ground rules

- **Zero runtime dependencies.** The package must keep importing on a bare stdlib install. `pytest` is the only dev dependency.
- **Python 3.9+.** CI runs 3.9, 3.11 and 3.13.
- **Every behaviour change ships with a test that fails before the change.** A test that is green both before and after proves nothing.
- **When in doubt, refuse.** This library edits people's source files. A wrong edit that parses is worse than an honest failure, so new matching heuristics need a strong argument — see the Design decisions section of the README for the ones already considered and rejected.

## Bug reports

The most useful report is a failing case: the file contents, the SEARCH/REPLACE block, what happened, and what you expected. If you can express it as a `pytest` function, even better — paste it in and I will take it from there.

## Pull requests

Small and focused beats large and comprehensive. If you are planning something structural, open an issue first so we can agree on the shape before you spend the time.
