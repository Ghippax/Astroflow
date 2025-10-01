# Contributing to Cosmo-Analysis

Thank you for your interest in contributing to Cosmo-Analysis! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Cosmo-Analysis.git
   cd Cosmo-Analysis
   ```
3. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. Install additional development tools:
   ```bash
   pip install flake8 black isort
   ```

3. Copy and configure the template:
   ```bash
   cp config_template.yaml config.yaml
   # Edit config.yaml as needed
   ```

## Code Style

We follow PEP 8 style guidelines. Before submitting code:

1. Format your code with black:
   ```bash
   black src/ tests/
   ```

2. Sort imports with isort:
   ```bash
   isort src/ tests/
   ```

3. Check for style issues with flake8:
   ```bash
   flake8 src/ tests/
   ```

## Testing

All new code should include tests. We use pytest for testing.

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_config.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=cosmo_analysis --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Include docstrings explaining what is being tested
- Use fixtures from `conftest.py` for common setup

Example:
```python
def test_config_loads_correctly(temp_config_file):
    """Test that configuration loads from file correctly."""
    config = load_config(str(temp_config_file))
    assert config.get('paths.output_directory') == '/tmp/test_output'
```

## Documentation

All new functions, classes, and modules should include docstrings.

### Docstring Style

We use Google-style docstrings:

```python
def my_function(param1, param2):
    """Brief description of function.
    
    Longer description if needed.
    
    Args:
        param1 (str): Description of param1
        param2 (int): Description of param2
        
    Returns:
        bool: Description of return value
        
    Raises:
        ValueError: When param1 is invalid
    """
    pass
```

### Building Documentation

Build the Sphinx documentation:
```bash
cd docs
make html
```

View the built docs in `docs/_build/html/index.html`.

## Pull Request Process

1. Ensure all tests pass:
   ```bash
   pytest tests/
   ```

2. Update documentation if needed

3. Update CHANGELOG.md with your changes

4. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Open a Pull Request on GitHub

6. Ensure all CI checks pass

7. Address any review comments

## Pull Request Guidelines

- **Keep changes focused**: One feature or fix per PR
- **Write clear commit messages**: Use descriptive, present-tense messages
- **Include tests**: All new functionality should be tested
- **Update documentation**: Document new features and changes
- **Follow the roadmap**: Check `roadmap.txt` for planned changes

## Code Review

All submissions require review. We will:
- Check that tests pass
- Review code quality and style
- Ensure documentation is adequate
- Verify changes align with project goals

## Reporting Issues

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and package versions
- Configuration used (if relevant)
- Error messages and stack traces

## Feature Requests

Feature requests are welcome! Please:
- Check if the feature is already in `roadmap.txt`
- Describe the use case and benefits
- Suggest an implementation approach if possible

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the "question" label
- Check the documentation
- Review `roadmap.txt` for planned changes

Thank you for contributing to Cosmo-Analysis!
