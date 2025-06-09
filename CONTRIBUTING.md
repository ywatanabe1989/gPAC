# Contributing to gPAC

Thank you for your interest in contributing to gPAC! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/gPAC.git`
3. Create a new branch: `git checkout -b feature-name`
4. Make your changes
5. Run tests: `./run_tests.sh`
6. Commit your changes: `git commit -am 'Add feature'`
7. Push to your fork: `git push origin feature-name`
8. Create a Pull Request

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv .env
   source .env/bin/activate  # On Windows: .env\Scripts\activate
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   pip install -r requirements-test.txt
   ```

## Code Style

- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Run `black` for code formatting
- Run `ruff` for linting

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Maintain or improve test coverage
- Run tests with: `pytest tests/`

## Documentation

- Update docstrings for API changes
- Update README.md if needed
- Add examples for new features
- Follow the MNGS framework for examples

## Pull Request Process

1. Update the CHANGELOG.md with your changes
2. Ensure all tests pass
3. Update documentation as needed
4. Request review from maintainers

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help maintain a positive community

## Questions?

Feel free to open an issue for any questions or discussions.