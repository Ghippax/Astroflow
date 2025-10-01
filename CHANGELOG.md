# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- YAML-based configuration system (`config.py`) for managing paths, plotting defaults, and analysis parameters
- Configuration template file (`config_template.yaml`) with comprehensive documentation
- pytest test framework with initial test coverage for config and fields modules
- Sphinx documentation infrastructure with RTD theme
- GitHub Actions CI/CD workflows for automated testing and documentation building
- README.md with installation instructions and quick start guide
- CONTRIBUTING.md with development guidelines
- pytest.ini for test configuration
- Comprehensive .gitignore for Python projects

### Changed
- Refactored `constants.py` to use lazy initialization for colormaps to avoid import-time errors
- Updated requirements.txt to include PyYAML, pytest, pytest-cov, sphinx, and sphinx-rtd-theme

### Fixed
- Fixed colormap initialization in `constants.py` that caused import errors when yt colormaps weren't available
- Added fallback to standard matplotlib colormaps when 'algae' colormap is unavailable

### Infrastructure
- Set up pytest framework with fixtures for testing
- Added GitHub Actions workflow for running tests on multiple Python versions (3.8-3.12)
- Added GitHub Actions workflow for building Sphinx documentation
- Configured code linting with flake8, black, and isort in CI

## [0.0.0] - Initial Release

### Added
- Initial codebase for analyzing galaxy simulations
- Support for Gadget/AREPO simulation formats
- Projection and phase space plotting capabilities
- Multiple centering algorithms
- Star formation rate analysis
- Rotation curve and velocity dispersion calculations
