# UTAG (Python 3.12 Compatible Fork)

Unsupervised discovery of Tissue Architecture with Graphs.

This is a local fork of [ElementoLab/utag](https://github.com/ElementoLab/utag) with Python 3.12 compatibility fixes.

## Changes from Original

- Fixed `types.py` to remove deprecated `pathlib._posix_flavour` and `pathlib._windows_flavour` attributes that were removed in Python 3.12.

## Installation

```bash
# From this directory
pip install -e .

# Or from the project root
pip install -e external/utag
```

## Original Authors

- Junbum Kim (juk4007@med.cornell.edu)
- Andre Rendeiro (afrendeiro@gmail.com)

## License

GPL-3.0 (same as original)
