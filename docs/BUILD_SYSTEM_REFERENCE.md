# DupFinder Build System Reference

**Version:** 1.0  
**Last Updated:** 2025-10-04  
**Target Audience:** Developers and Contributors  

---

## Overview

DupFinder uses a modern CMake-based build system with Qt6, designed for cross-platform compatibility and developer productivity. The build system includes advanced features for code quality, testing, documentation generation, and packaging.

## Quick Reference

### Essential Commands

```bash
# Basic build
cmake -B build && cmake --build build

# Run tests  
cd build && ctest

# Available targets
make summary                    # Show all available targets
make dupfinder                  # Build main application
make check                      # Run all tests
make package                    # Create distribution packages
```

---

## Build Targets

### Primary Targets

| Target | Description | Usage |
|--------|-------------|--------|
| `dupfinder` | Main application executable | `make dupfinder` |
| `unit_tests` | Unit test executable | `make unit_tests` |
| `integration_tests` | Integration test executable | `make integration_tests` |
| `check` | Run all tests | `make check` |
| `package` | Create distribution packages | `make package` |

### Development Targets

| Target | Description | Usage | Prerequisites |
|--------|-------------|-------|---------------|
| `format` | Format source code | `make format` | clang-format |
| `cppcheck` | Static analysis | `make cppcheck` | cppcheck |
| `memcheck` | Memory analysis | `make memcheck` | valgrind (Linux) |
| `coverage` | Generate coverage report | `make coverage` | lcov, gcov |
| `docs` | Generate API documentation | `make docs` | doxygen |
| `summary` | Show available targets | `make summary` | Always available |

---

## Configuration Options

### Standard CMake Variables

```bash
# Build type (Debug, Release, RelWithDebInfo, MinSizeRel)
-DCMAKE_BUILD_TYPE=Release

# Install prefix
-DCMAKE_INSTALL_PREFIX=/usr/local

# Compiler selection
-DCMAKE_CXX_COMPILER=g++
-DCMAKE_C_COMPILER=gcc

# Generator selection
-G "Ninja"                    # Fast builds
-G "Unix Makefiles"           # Standard makefiles
-G "Visual Studio 17 2022"    # Windows Visual Studio
```

### DupFinder-Specific Options

```bash
# Enable test coverage analysis (Debug builds only)
-DENABLE_COVERAGE=ON

# Enable all compiler warnings (already enabled by default)
-DENABLE_WARNINGS=ON

# Platform-specific options (auto-detected)
-DPLATFORM_NAME=linux         # linux, windows, macos
```

---

## Platform-Specific Build Instructions

### Linux (Ubuntu/Debian)

#### Install Dependencies
```bash
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    qt6-base-dev \
    qt6-tools-dev \
    qt6-base-dev-tools \
    libqt6core6 \
    libqt6gui6 \
    libqt6widgets6 \
    libqt6concurrent6 \
    libqt6network6 \
    libqt6test6
```

#### Build Commands
```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc \
    -GNinja

cmake --build build --parallel
```

### Windows (Visual Studio)

#### Prerequisites
- Visual Studio 2019 or later with C++ support
- Qt6 (6.4 or later) installed
- CMake 3.20 or later

#### Build Commands
```cmd
cmake -B build ^
    -DCMAKE_BUILD_TYPE=Release ^
    -G "Visual Studio 17 2022" ^
    -A x64

cmake --build build --config Release --parallel
```

### macOS

#### Install Dependencies
```bash
# Using Homebrew
brew install cmake ninja qt6
```

#### Build Commands
```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang \
    -GNinja

cmake --build build --parallel
```

---

## Advanced Build Features

### Code Quality Tools

#### 1. Automatic Code Formatting
```bash
# Format all source files
make format

# Check formatting without modifying files
find src include tests -name "*.cpp" -o -name "*.h" | xargs clang-format --dry-run --Werror
```

**Configuration**: Uses `.clang-format` file in project root.

#### 2. Static Analysis
```bash
# Run comprehensive static analysis
make cppcheck

# Custom cppcheck command
cppcheck --enable=all --std=c++17 --suppress=missingIncludeSystem src/ include/
```

**Features**:
- Checks for bugs, performance issues, and style problems
- Configurable suppressions for false positives
- Integrates with CI/CD pipeline

#### 3. Memory Analysis (Linux)
```bash
# Run memory check with valgrind
make memcheck

# Custom valgrind command
valgrind --tool=memcheck --leak-check=full ./dupfinder
```

**Features**:
- Detects memory leaks and memory errors
- Comprehensive heap analysis
- Stack trace reporting

#### 4. Test Coverage Analysis
```bash
# Enable coverage in Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON

# Build and run tests
cmake --build build
cd build && ctest

# Generate coverage report
make coverage
```

**Output**: HTML coverage report in `build/coverage/html/`

### Documentation Generation

#### API Documentation with Doxygen
```bash
# Generate documentation
make docs

# Custom doxygen command
doxygen docs/Doxyfile
```

**Features**:
- Automatic API documentation from source comments
- Class hierarchy diagrams
- Cross-referenced source code
- HTML and LaTeX output formats

---

## Build System Architecture

### CMake Module Structure

```
cmake/
├── FindQt6.cmake              # Qt6 detection (built-in)
├── CompilerFlags.cmake         # Compiler-specific settings
├── PlatformDetection.cmake     # Platform-specific configurations  
└── Testing.cmake              # Test configuration helpers
```

### File Organization

```
CMakeLists.txt                 # Main build configuration
├── src/CMakeLists.txt         # Source files configuration
├── tests/CMakeLists.txt       # Test configuration
├── docs/CMakeLists.txt        # Documentation configuration
└── resources/CMakeLists.txt   # Resource files configuration
```

### Target Dependencies

```
dupfinder (main executable)
├── Qt6::Core
├── Qt6::Widgets  
├── Qt6::Concurrent
├── Qt6::Network
└── Platform libraries (Linux: dl, pthread)

unit_tests
├── Qt6::Test
├── Core source files (recompiled)
└── Test framework

integration_tests  
├── Qt6::Test
├── Core source files (recompiled)
└── Integration test framework
```

---

## Build Optimization

### Compiler Optimizations

#### Release Build Optimizations
```bash
# Enabled automatically in Release builds:
-DCMAKE_BUILD_TYPE=Release
```

Features:
- Link Time Optimization (LTO) enabled
- Symbol visibility controls
- Optimized binary size
- Debug information stripped

#### Debug Build Features
```bash
-DCMAKE_BUILD_TYPE=Debug
```

Features:
- Full debug symbols
- Runtime checks enabled
- Address sanitizer support (GCC/Clang)
- Coverage instrumentation available

### Parallel Building

#### Make/Ninja Parallel Builds
```bash
# Automatic parallel detection
make -j$(nproc)                 # Linux
make -j$(sysctl -n hw.ncpu)     # macOS

# Or use cmake --build with --parallel
cmake --build build --parallel
```

#### MSBuild Parallel (Windows)
```cmd
cmake --build build --config Release --parallel
```

### Build Caching

#### CMake Build Cache
```bash
# Use ccache for compilation caching (Linux/macOS)
export CC="ccache gcc"
export CXX="ccache g++"

cmake -B build
```

#### Qt6 Action Caching (CI)
Uses `jurplel/install-qt-action@v3` with caching enabled in GitHub Actions.

---

## Packaging and Distribution

### Linux Packages

#### Available Formats
- `.tar.gz` - Generic Linux archive
- `.deb` - Debian/Ubuntu package  
- `.rpm` - RedHat/Fedora package

```bash
# Generate all packages
make package

# Generate specific package type
cpack -G DEB
cpack -G RPM  
cpack -G TGZ
```

### Windows Installer

#### NSIS Installer
```cmd
# Build installer (requires NSIS)
cmake --build build --config Release
cd build
cpack -C Release
```

**Output**: `dupfinder-1.0.0-win64.exe`

### macOS Bundle

#### DMG Creation
```bash
# Build macOS bundle and DMG
make package
```

**Output**: `dupfinder-1.0.0-Darwin.dmg`

---

## Troubleshooting

### Common Build Issues

#### Qt6 Not Found
```
CMake Error: Could not find a package configuration file provided by "Qt6"
```

**Solutions**:
```bash
# Set Qt6 path manually
export Qt6_DIR=/path/to/qt6/lib/cmake/Qt6

# Or specify during cmake configuration
cmake -DQt6_DIR=/path/to/qt6/lib/cmake/Qt6 -B build
```

#### Compiler Warnings as Errors
```
error: conversion from 'qsizetype' to 'int' may change value [-Werror=conversion]
```

**Solutions**:
```bash
# Disable warnings as errors (not recommended for production)
cmake -DENABLE_WARNINGS=OFF -B build

# Fix the actual warnings (recommended)
# Use static_cast<int>() for safe conversions
```

#### Test Linking Failures
```
undefined reference to `FileScanner::scanStarted()'
```

**Solutions**:
- Ensure MOC processing is working (AUTOMOC=ON)
- Check that header files are included in CMakeLists.txt
- Verify Q_OBJECT macro is present in header files

#### Memory Issues During Build
```
c++: fatal error: Killed signal terminated program cc1plus
```

**Solutions**:
```bash
# Reduce parallel jobs
make -j2  # instead of -j$(nproc)

# Or disable parallel building temporarily
cmake --build build -j1
```

### Debug Build System Issues

#### Verbose Build Output
```bash
cmake --build build --verbose
# or
make VERBOSE=1
```

#### CMake Configuration Debug
```bash
cmake --debug-output -B build
cmake --trace -B build          # Very verbose
```

#### Generator Expressions Debug
```bash
cmake --build build -- VERBOSE=1
```

---

## CI/CD Integration

### GitHub Actions

The build system integrates with GitHub Actions for automated CI/CD:

- **Linux builds**: Ubuntu latest with GCC
- **Windows builds**: Visual Studio 2022  
- **macOS builds**: Xcode latest with Clang
- **Code quality**: Static analysis, formatting, testing
- **Artifacts**: Automatic package generation and upload
- **Release automation**: Tagged releases with multi-platform binaries

### Local CI Simulation

```bash
# Simulate CI environment locally
docker run -it ubuntu:latest
apt-get update && apt-get install -y git cmake qt6-base-dev

# Clone and build as CI would
git clone <repository>
cd dupfinder
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cd build && ctest
```

---

## Performance Considerations

### Build Times

| Component | Typical Time | Optimization |
|-----------|-------------|--------------|
| CMake Configure | 5-15s | Use Ninja generator |
| Main Application | 30-60s | Parallel builds, ccache |
| Unit Tests | 15-30s | Shared compilation units |
| Integration Tests | 10-20s | Minimal dependencies |
| Documentation | 10-30s | Incremental builds |

### Binary Sizes

| Build Type | Size | Notes |
|------------|------|-------|
| Debug | ~50MB | Full symbols, no optimization |
| Release | ~15MB | Optimized, stripped |
| MinSizeRel | ~10MB | Size-optimized |

### Memory Usage

| Operation | RAM Usage | Notes |
|-----------|-----------|-------|
| Compilation | 2-4GB | Peak during linking |
| Testing | 500MB | Per test executable |
| Documentation | 100MB | Doxygen processing |

---

## Contributing to Build System

### Adding New Targets

1. **Edit CMakeLists.txt**:
```cmake
add_custom_target(new_target
    COMMAND your_command
    COMMENT "Description of new target"
    VERBATIM
)
```

2. **Add to summary target**:
```cmake
# In summary target definition
COMMAND ${CMAKE_COMMAND} -E echo "  new_target     - Your description"
```

### Adding Dependencies

1. **Find package**:
```cmake
find_package(NewLibrary REQUIRED)
```

2. **Link to targets**:
```cmake
target_link_libraries(dupfinder NewLibrary::NewLibrary)
```

### Platform-Specific Code

```cmake
if(WIN32)
    # Windows-specific configuration
elseif(APPLE)  
    # macOS-specific configuration
elseif(UNIX)
    # Linux-specific configuration
endif()
```

---

This completes the comprehensive build system reference. The build system is designed to be robust, efficient, and developer-friendly while maintaining high code quality standards.