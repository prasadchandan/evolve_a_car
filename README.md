## Evolve a Car

Use genetic algorithms to evolve a car. The framework currently uses Box2D for the physics engine, pyglet for rendering and imgui for the UI.

The code to 'evolve' the cars from generation to generation has not yet been implemented. This is a work in progress.

Quite a lot of the code in the project is derived from the python Box2D examples.

### Requirements
- Python 3.11 or higher

### Download Pre-built Binaries
Pre-built binaries for all supported platforms are available in the [GitHub Releases](https://github.com/prasadchandan/evolve_a_car/releases) section. Choose the appropriate binary for your platform:

- `evolve_a_car-linux-x86_64` - Linux 64-bit Intel/AMD
- `evolve_a_car-linux-aarch64` - Linux 64-bit ARM
- `evolve_a_car-windows-x86_64.exe` - Windows 64-bit Intel/AMD
- `evolve_a_car-windows-aarch64.exe` - Windows 64-bit ARM (experimental)
- `evolve_a_car-macos-x86_64` - macOS Intel
- `evolve_a_car-macos-aarch64` - macOS Apple Silicon (M1/M2/M3/M4)

### How to run from source
1. Install [UV](https://github.com/astral-sh/uv) - `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Create virtual environment: `uv venv`
3. Install the dependencies: `uv pip install -e .`
4. On macOS follow the instructions below to fix OpenGL issues
5. Run application: `uv run python src/app.py`

### Development

#### Running Tests
```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run all tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

#### Building Binaries
The project uses PyInstaller to create standalone executables for Linux, Windows, and macOS.

```bash
# Install development dependencies (includes PyInstaller)
uv pip install -e ".[dev]"

# Build binary for your platform
uv run pyinstaller evolve_a_car.spec

# Find the binary in the dist/ directory
```

#### CI/CD
The project uses GitHub Actions for continuous integration and deployment:

- **CI Workflow**: Runs tests on every push and pull request across Python 3.11-3.12 on Linux, Windows, and macOS
- **Build Workflow**: Creates standalone binaries for multiple platforms and architectures when a version tag is pushed (e.g., `v1.0.0`). Binaries are uploaded as GitHub release artifacts.

**Supported Platforms and Architectures:**
- **Linux**
  - x86_64 (amd64) - Fully supported
  - ARM64 (aarch64) - Supported using QEMU emulation
- **Windows**
  - x86_64 (amd64) - Fully supported
  - ARM64 (aarch64) - Experimental (cross-compilation)
- **macOS**
  - x86_64 (Intel) - Fully supported
  - ARM64 (Apple Silicon M1/M2/M3/M4) - Fully supported

To trigger a release build:
```bash
git tag v1.0.0
git push origin v1.0.0
```

The build workflow will create binaries for all 6 platform/architecture combinations and attach them to the GitHub release.
### Known Issues

1. On macOS Big Sur OpenGL is not found see discussion on [StackOverflow for details](https://stackoverflow.com/questions/63475461/unable-to-import-opengl-gl-in-python-on-macos).
  - Temporary Fix:
    + Find virtual environment path (should be `.venv` in the project root)
    + Open `.venv/lib/python3.x/site-packages/OpenGL/platform/ctypesloader.py`
    + Replace line `fullName = util.find_library( name )`
    + With line `fullName = "/System/Library/Frameworks/{}.framework/{}".format(name,name)`

### Demo
![Evolve a Car Demo GIF](repo-assets/demo.gif)

### Credits 
1. [Box2D Examples](https://github.com/openai/box2d-py/tree/master/examples)
2. [Hack Font](https://github.com/source-foundry/Hack)
3. [Gidole Font](https://github.com/larsenwork/Gidole)
