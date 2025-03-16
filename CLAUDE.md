# DeltaRCM Project Guidelines

## Build & Test Commands
```bash
# Install package in development mode
pip install -e .

# Run tests
pytest tests/

# Run a single test
pytest tests/path_to_test.py::test_function_name

# Check type hints
mypy deltarcm/

# Lint code
flake8 deltarcm/
```

## Code Style Guidelines
- **Imports**: Group standard library, third-party, and local imports. Use absolute imports.
- **Formatting**: Follow PEP 8. Line length max 88 characters (Black compatible).
- **Types**: Use type hints for function signatures and class attributes.
- **Naming**: Use snake_case for functions/variables, CamelCase for classes, UPPER_CASE for constants.
- **Documentation**: NumPy docstring format with examples and parameter descriptions.
- **Error Handling**: Use specific exceptions, add context messages.
- **Testing**: Write unit tests for all modules using pytest.
- **Model Implementation**: Follow OOP principles described in docs/esurf-3-67-2015.pdf and docs/esurf-3-87-2015.pdf.