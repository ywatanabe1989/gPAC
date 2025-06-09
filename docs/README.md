# gPAC Documentation

This directory contains the Sphinx documentation for the gPAC project.

## Building the Documentation

1. Install the documentation dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build the HTML documentation:
   ```bash
   make html
   ```

3. View the documentation:
   Open `build/html/index.html` in your web browser.

## Documentation Structure

- `source/` - Source files for the documentation
  - `conf.py` - Sphinx configuration
  - `index.rst` - Main documentation page
  - `installation.rst` - Installation guide
  - `quickstart.rst` - Quick start tutorial
  - `user_guide.rst` - Detailed user guide
  - `api_reference.rst` - API documentation
  - `examples.rst` - Example usage

- `build/` - Generated documentation output (not tracked in git)

## Updating the Documentation

1. Edit the `.rst` files in the `source/` directory
2. Rebuild the documentation with `make html`
3. Check for any warnings or errors in the build output

## Cleaning

To clean the build directory:
```bash
make clean
```

## Live Development

For live documentation development with auto-reload:
```bash
pip install sphinx-autobuild
make livehtml
```

This will start a local server and automatically rebuild the documentation when files change.