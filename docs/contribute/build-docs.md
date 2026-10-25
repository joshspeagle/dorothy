# Build the docs

```bash
pip install --group docs                  # pip 25.1 or later
sphinx-build -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. `-W` turns warnings,
including malformed docstrings, into errors, as the CI build does.
