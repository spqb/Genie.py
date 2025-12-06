# Genie 2.0

A Python package template with a standard structure.

## Installation

### From source

```bash
pip install -e .
```

### For development

```bash
pip install -e ".[dev]"
```

## Usage

### As a module

```python
from genie import Genie, hello

# Use the hello function
print(hello())

# Create a Genie instance
genie = Genie(name="MyGenie")
print(genie.greet())
```

### Command-line interface

```bash
# Run as a module
python -m genie

# Or use the installed command
genie
```

## Package Structure

```
Genie2.0.py/
├── src/
│   └── genie/
│       ├── __init__.py
│       ├── __version__.py
│       ├── __main__.py
│       └── core.py
├── tests/
│   └── test_core.py
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── setup.py
└── requirements.txt
```

## Development

### Running tests

```bash
pytest
```

### Code formatting

```bash
black src/
```

### Linting

```bash
flake8 src/
```

## License

MIT License - see LICENSE file for details.

## Author

Roberto Netti