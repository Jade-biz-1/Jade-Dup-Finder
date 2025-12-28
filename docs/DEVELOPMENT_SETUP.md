# Development Environment Setup

This document provides detailed instructions for setting up the CloneClean development environment on Windows, macOS, and Linux platforms.

## Prerequisites

Before starting, ensure you have the following:
- Git for version control
- A modern C++17 compatible compiler
- CMake 3.20 or later
- Qt 6.0 or later (Qt 6.5+ recommended)

### Optional: GPU Acceleration Toolchain

If you plan to build with CUDA acceleration:

- NVIDIA GPU with recent Game Ready/Studio drivers (Windows) or proprietary drivers (Linux)
- CUDA Toolkit 12.x (or the version referenced in `config/build_profiles.json`)
- Visual Studio 2022 with MSVC toolset on Windows, or the appropriate host compiler on Linux
- Ensure `nvcc` is accessible (add `CUDA_PATH/bin` to your PATH if necessary)

## Platform-Specific Setup

### Linux Development Setup

#### Ubuntu 20.04 LTS / 22.04 LTS

```bash
# Update package manager
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    pkg-config

# Install Qt6 development packages
sudo apt install -y \
    qt6-base-dev \
    qt6-tools-dev \
    qt6-tools-dev-tools \
    libqt6core6 \
    libqt6widgets6 \
    libqt6concurrent6 \
    libqt6network6

# Install additional development tools (optional)
sudo apt install -y \
    clang-format \
    clang-tidy \
    valgrind \
    gdb
```

#### Fedora 35+ / CentOS Stream 9

```bash
# Update package manager
sudo dnf update -y

# Install essential build tools
sudo dnf groupinstall -y "Development Tools" "Development Libraries"
sudo dnf install -y \
    cmake \
    ninja-build \
    git

# Install Qt6 development packages
sudo dnf install -y \
    qt6-qtbase-devel \
    qt6-qttools-devel \
    qt6-qtconcurrent \
    qt6-qtnetwork

# Install additional development tools (optional)
sudo dnf install -y \
    clang-tools-extra \
    valgrind \
    gdb
```

#### Arch Linux

```bash
# Update system
sudo pacman -Syu

# Install essential build tools
sudo pacman -S \
    base-devel \
    cmake \
    ninja \
    git

# Install Qt6 packages
sudo pacman -S \
    qt6-base \
    qt6-tools

# Install additional development tools (optional)
sudo pacman -S \
    clang \
    valgrind \
    gdb
```

### Windows Development Setup

#### Option 1: Visual Studio 2019/2022 (Recommended)

1. **Install Visual Studio**
   - Download Visual Studio Community/Professional/Enterprise 2019 or 2022
   - During installation, select "Desktop development with C++" workload
   - Ensure Windows 10/11 SDK is included

2. **Install CMake**
   - Download from https://cmake.org/download/
   - Add CMake to your system PATH during installation

3. **Install Qt6**
   - Download Qt Online Installer from https://www.qt.io/download-qt-installer
   - Install Qt 6.4 LTS or later
   - Select MSVC 2019/2022 64-bit compiler during installation
   - Add Qt6 to your system PATH: `C:\Qt\6.x.x\msvc2019_64\bin`

4. **Install Git**
   - Download from https://git-scm.com/download/win
   - Use Git Bash or integrate with Visual Studio

5. **Install Qt Visual Studio Tools (Optional)**
   - In Visual Studio, go to Extensions → Manage Extensions
   - Search for "Qt Visual Studio Tools" and install

#### Option 2: MinGW-w64 + Qt Creator

1. **Install Qt with MinGW**
   - Download Qt Online Installer
   - Install Qt 6.4 LTS with MinGW 11.2.0 64-bit

2. **Install CMake**
   - Download from https://cmake.org/download/
   - Add to system PATH

3. **Install Git**
   - Download from https://git-scm.com/download/win

### macOS Development Setup

#### Using Homebrew (Recommended)

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install essential development tools
brew install cmake ninja git

# Install Qt6
brew install qt6

# Install additional development tools (optional)
brew install clang-format llvm

# Add Qt6 to your PATH (add to ~/.zshrc or ~/.bash_profile)
echo 'export PATH="/opt/homebrew/opt/qt6/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### Using Qt Installer

1. **Install Xcode Command Line Tools**
   ```bash
   xcode-select --install
   ```

2. **Install CMake**
   ```bash
   # Using Homebrew
   brew install cmake
   
   # OR download from https://cmake.org/download/
   ```

3. **Install Qt6**
   - Download Qt Online Installer from https://www.qt.io/download-qt-installer
   - Install Qt 6.4 LTS or later
   - Select macOS compiler during installation

4. **Install Git** (usually pre-installed)
   ```bash
   git --version
   ```

## IDE Setup

### Qt Creator (Cross-Platform)

1. **Install Qt Creator**
   - Usually included with Qt installation
   - Or download separately from https://www.qt.io/product/development-tools

2. **Configure CMake**
   - Go to Preferences → Kits → CMake
   - Ensure CMake path is correctly set

3. **Configure Qt Kit**
   - Go to Preferences → Kits → Qt Versions
   - Add Qt 6.x installation if not auto-detected
   - Go to Preferences → Kits → Kits
   - Create or verify a kit with Qt 6.x and CMake

4. **Open Project**
   - File → Open File or Project
   - Navigate to CloneClean directory and select `CMakeLists.txt`

### Visual Studio Code (Cross-Platform)

1. **Install VS Code**
   - Download from https://code.visualstudio.com/

2. **Install Extensions**
   - C/C++ (Microsoft)
   - CMake Tools (Microsoft)
   - Qt tools for Visual Studio Code
   - GitLens (optional)

3. **Configure CMake**
   - Open CloneClean folder in VS Code
   - Press `Ctrl+Shift+P` and select "CMake: Configure"
   - Select appropriate kit when prompted

### CLion (JetBrains)

1. **Install CLion**
   - Download from https://www.jetbrains.com/clion/

2. **Configure Toolchain**
   - Go to File → Settings → Build, Execution, Deployment → Toolchains
   - Ensure compiler and CMake are detected

3. **Open Project**
   - File → Open → Select CloneClean directory

## Building the Project

### Primary Workflow: Unified Build Script

The repository ships with `scripts/build.py`, which selects the correct generator, configures CMake, builds the chosen target, optionally runs `cpack`, and mirrors finished artifacts into `dist/`.

1. **Configure build profiles (first run only)**
    ```bash
    cp config/build_profiles.example.json config/build_profiles.json
    ```
    Edit the copy to point at your local toolchains (Qt, CUDA, Visual Studio vcvars script, etc.). Remove targets you do not use.

2. **Run the script interactively**
    ```bash
    python scripts/build.py
    ```
    Select the desired target (for example `windows-msvc-cpu` or `windows-msvc-cuda`) and build type (`Debug` or `Release`). The script automatically prepares the environment, runs the build, and copies results into `dist/`.

3. **Non-interactive/CI usage**
    ```bash
    python scripts/build.py \
       --target windows-msvc-cpu \
       --build-type Release \
       --non-interactive --force
    ```

Use `--skip-package` if you only want binaries, or `--list-targets` to inspect available profiles.

### Manual CMake Workflow (Advanced)

Manual invocation remains useful for rapid iteration or debugging:

```bash
# Configure (example: Ninja on Linux)
cmake -S . -B build/linux/x64/manual-ninja -G "Ninja" \
   -DCLONECLEAN_BUILD_VARIANT=cpu \
   -DENABLE_GPU_ACCELERATION=OFF

# Build
cmake --build build/linux/x64/manual-ninja --parallel

# Test
cd build/linux/x64/manual-ninja && ctest --output-on-failure
```

On Windows, open a "x64 Native Tools Command Prompt for VS 2022" and run:

```cmd
cmake -S . -B build/windows/win64/manual-msvc ^
   -G "Visual Studio 17 2022" -A x64 ^
   -DCLONECLEAN_BUILD_VARIANT=cpu
cmake --build build/windows/win64/manual-msvc --config Release --target cloneclean
```

To enable CUDA builds manually, add `-DCLONECLEAN_BUILD_VARIANT=gpu -DENABLE_GPU_ACCELERATION=ON` and ensure CUDA libraries are resolvable.

### IDE-Specific Build

#### Qt Creator
1. Open `CMakeLists.txt`
2. Configure the project with desired kit
3. Press `Ctrl+B` to build
4. Press `Ctrl+R` to run

#### Visual Studio
1. Open the folder containing `CMakeLists.txt`
2. Select appropriate configuration (Debug/Release)
3. Build → Build All
4. Set cloneclean as startup project and run

#### VS Code
1. Open folder in VS Code
2. Press `Ctrl+Shift+P` → "CMake: Build"
3. Press `F5` to debug or `Ctrl+F5` to run

## Troubleshooting

### Common Issues

#### Qt6 Not Found
```bash
# Set Qt6_DIR environment variable
export Qt6_DIR=/path/to/qt6/lib/cmake/Qt6  # Linux/macOS
set Qt6_DIR=C:\Qt\6.x.x\msvc2019_64\lib\cmake\Qt6  # Windows
```

#### CMake Version Too Old
```bash
# Ubuntu: Add Kitware's APT repository for latest CMake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt update && sudo apt install cmake
```

#### Compiler Not Found (Windows)
- Ensure Visual Studio C++ tools are installed
- Run CMake from "Developer Command Prompt for VS"
- Or install MinGW-w64 and ensure it's in PATH

#### CUDA Toolkit Not Detected
- Confirm the toolkit is installed and `nvcc --version` works
- Ensure the profile or environment exports `CUDA_PATH` and adds `CUDA_PATH\bin` to `PATH`
- On Windows, run builds from a Visual Studio Developer Command Prompt so MSVC is discoverable

#### GPU Build Using Wrong Generator
- CUDA requires MSVC on Windows; use the `windows-msvc-cuda` profile or specify `-G "Visual Studio 17 2022"`
- If you see `Error: could create CMAKE_GENERATOR "Visual Studio 17 2022"`, launch the build inside the Visual Studio developer shell or reference the full path to the newer CMake bundled with Qt

### Platform-Specific Issues

#### Linux
- **Missing Qt6 packages**: Use package manager to install missing Qt6 development packages
- **Permission issues**: Ensure user has write permissions to build directory
- **Library linking errors**: Install development packages for system libraries

#### Windows
- **MSVC not found**: Install Visual Studio with C++ development tools
- **Qt DLLs not found at runtime**: Add Qt6 bin directory to PATH or deploy DLLs with executable
- **Long path issues**: Enable long path support in Windows 10/11

#### macOS
- **Xcode command line tools**: Run `xcode-select --install` if compiler not found
- **Homebrew permissions**: Fix with `sudo chown -R $(whoami) /usr/local/share/man/man8`
- **Qt framework issues**: Ensure proper Qt6 installation and DYLD_LIBRARY_PATH if needed

## Development Workflow

### Code Style and Formatting

We use clang-format for consistent code styling:

```bash
# Format all source files
make format  # or ninja format

# Check formatting without modifying files
clang-format --dry-run -Werror src/**/*.{cpp,h}
```

### Testing

```bash
# Build and run all tests
make check  # or ninja check

# Run specific test categories
ctest -L unit       # Unit tests only
ctest -L integration  # Integration tests only

# Run tests with verbose output
ctest --verbose
```

### Continuous Integration

The project uses GitHub Actions for CI/CD. Ensure your changes pass:
- All unit and integration tests
- Code formatting checks
- Cross-platform compilation
- Static analysis (if configured)

### Debug Builds

For debugging, use Debug configuration:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
gdb ./cloneclean  # Linux
lldb ./cloneclean  # macOS
```

## Getting Help

- **Documentation**: Check `/docs` directory for additional documentation
- **Issues**: Report problems on GitHub Issues
- **Community**: Join discussions on GitHub Discussions
- **Qt Documentation**: https://doc.qt.io/qt-6/
- **CMake Documentation**: https://cmake.org/documentation/