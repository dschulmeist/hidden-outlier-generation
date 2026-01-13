# Contributing to Hidden Outlier Generation

## Development Setup

```bash
# Clone the repository
git clone https://github.com/dschulmeist/hidden-outlier-generation
cd hidden-outlier-generation

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (including dev)
uv sync --all-extras

# Verify installation
uv run pytest
```

## Code Quality

Before submitting changes, ensure your code passes all checks:

```bash
# Run tests
uv run pytest

# Run linter
uv run ruff check

# Run formatter
uv run ruff format

# Run type checker
uv run mypy src/
```

## Project Structure

```
hidden-outlier-generation/
├── src/hog_bisect/          # Library code (pip installable)
├── examples/                 # Usage examples (not pip installed)
├── experiments/              # Research experiments (not pip installed)
├── tests/                    # Test suite
├── pyproject.toml           # Project configuration
└── uv.lock                  # Dependency lock file
```

## Making Changes

1. Create a branch for your changes
2. Make your changes with clear commit messages
3. Ensure all tests pass
4. Submit a pull request

## Versioning

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

The version is defined in `src/hog_bisect/__init__.py`:

```python
__version__ = "1.0.1"
```

---

## Publishing to PyPI

### Prerequisites

1. **PyPI Account**: Create an account at [pypi.org](https://pypi.org)

2. **API Token**: Generate a token at [pypi.org/manage/account/token](https://pypi.org/manage/account/token)
   - Scope: "Entire account" or project-specific

3. **GitHub Secret**: Add the token to your repository:
   - Go to: `github.com/dschulmeist/hidden-outlier-generation/settings/secrets/actions`
   - Create secret named `PYPI_API_TOKEN` with your token value

### Release Process

#### Option 1: Commit Message Trigger

```bash
# 1. Update version in src/hog_bisect/__init__.py
#    Change: __version__ = "1.0.1"
#    To:     __version__ = "1.0.2"

# 2. Commit with "publish" in the message
git add -A
git commit -m "release: publish v1.0.2"

# 3. Push to master
git push origin master
```

#### Option 2: Git Tag Trigger

```bash
# 1. Update version in src/hog_bisect/__init__.py
git add src/hog_bisect/__init__.py
git commit -m "release: bump version to 1.0.2"

# 2. Create and push a version tag
git tag v1.0.2
git push origin master --tags
```

### Verifying the Release

1. Check GitHub Actions: `github.com/dschulmeist/hidden-outlier-generation/actions`
2. Verify on PyPI: `pypi.org/project/hidden-outlier-generation`
3. Test installation: `pip install hidden-outlier-generation==1.0.2`

### Local Build Testing

Before publishing, you can test the build locally:

```bash
# Build the package
uv run python -m build

# Check the contents
unzip -l dist/*.whl

# Test install in a fresh environment
uv venv /tmp/test-env
source /tmp/test-env/bin/activate
pip install dist/*.whl
python -c "from hog_bisect import BisectHOGen; print('Success!')"
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Workflow not triggering | Ensure you're pushing to `master` branch |
| Authentication failed | Verify `PYPI_API_TOKEN` secret is set correctly |
| Version already exists | PyPI doesn't allow re-uploading same version; bump version number |
| Build fails | Run `uv run python -m build` locally to debug |

---

## Running Experiments

```bash
# Run benchmarks
uv run python experiments/benchmarks/scalability.py

# Run examples
uv run python examples/basic_usage.py
```

## Questions?

Open an issue at [github.com/dschulmeist/hidden-outlier-generation/issues](https://github.com/dschulmeist/hidden-outlier-generation/issues)
