# DupFinder Build System Guide

**Last updated:** 2025-11-04  
**Scope:** Desktop (CPU + optional GPU) builds on Windows, Linux, and macOS

---

## 1. Architecture At A Glance

The build pipeline now has three cooperating layers:

| Layer | Purpose | Key Files/Locations |
|-------|---------|---------------------|
| **CMake project** | Describes targets, Qt6 AUTOGEN rules, packaging, and tool integrations. | `CMakeLists.txt`, `cmake/` modules, `resources/` |
| **Build profiles** | Capture per-machine generators, toolchain paths, environment tweaks, and output routing. | `config/build_profiles.json` (copy of `build_profiles.example.json`) |
| **Unified script** | Detects the host, selects a profile, runs configure/build/package, and copies artifacts to `dist/`. | `scripts/build.py` |

Supporting directories:

- `build/` — generator-specific build trees (e.g., `build/windows/win64/windows-msvc-cpu`).
- `dist/` — normalized distribution output (`Win64/Release`, `Linux/Debug`, etc.).
- `docs/` — build references and operational guides (this document plus topical guides).

---

## 2. Preparing A Machine

1. Install the platform toolchain:
   - **Windows (MSVC)**: Visual Studio 2022 + Desktop development tools.
   - **Windows (MinGW)**: MinGW-w64 toolchain matching the chosen Qt build.
   - **Linux**: GCC/Clang, CMake ≥ 3.24, Ninja (recommended), Qt6 dev packages.
   - **macOS**: Xcode command-line tools, CMake, Ninja, Homebrew Qt6.
2. Install Qt6 binaries matching the compiler (e.g., `msvc2022_64`, `mingw_64`, `/opt/qt6`).
3. (Optional) Install CUDA Toolkit if you plan GPU builds.
4. Copy the example build profile and edit it for local paths:

```bash
cp config/build_profiles.example.json config/build_profiles.json
```

Update each target you intend to use:

- `generator` and `architecture` (Visual Studio / Ninja / Makefiles).
- `setup_scripts` pointing to toolchain environment scripts (`vcvars64.bat`).
- `environment.Qt6_DIR` and optional `CUDA_PATH` / `CUDA_HOME`.
- `path` entries so auxiliary tools (e.g., `nvcc`) are discoverable.

You can delete entries you do not need; the script only considers profiles that match the current OS/arch.

---

## 3. Using `scripts/build.py`

Key features:

- Detects OS/architecture and auto-filters valid targets.
- Detects CUDA availability and highlights GPU-only profiles.
- Supports interactive or non-interactive operation.
- Runs configure, build, optional `cpack`, then mirrors artifacts into `dist/`.

### 3.1 Interactive Flow (recommended the first time)

```bash
python scripts/build.py
```

You will be prompted to:

1. Choose a matching build target (e.g., `windows-msvc-cpu`).
2. Select the build type (Debug or Release).
3. Confirm the summary (generator, GPU requirement, notes).

The script automatically:

- Creates `build/<os>/<arch>/<target>/` and configures CMake.
- Runs the correct generator (`cmake --build ...`).
- Invokes `cpack` unless `--skip-package` is passed.
- Copies new artifacts into `dist/`.

### 3.2 CLI Flags (for automation/CI)

```
python scripts/build.py \
  --target windows-msvc-cpu \
  --build-type Release \
  --non-interactive --force
```

Useful options:

- `--list-targets` — print available profiles.
- `--clean` — wipe the selected build directory before configuring.
- `--skip-package` — build binaries only.
- `--dry-run` — show planned commands without executing.

Environment variables declared in a profile (`environment`) are applied before invoking any commands, and `setup_scripts` are `call`ed automatically on Windows shells.

---

## 4. Manual CMake Workflows (Fallback)

You can still interact with CMake directly when needed (e.g., custom debugging).

### 4.1 Configure

```bash
cmake -S . -B build/windows/win64/manual-msvc ^
  -G "Visual Studio 17 2022" -A x64 ^
  -DDUPFINDER_BUILD_VARIANT=cpu ^
  -DENABLE_GPU_ACCELERATION=OFF
```

Linux/macOS use `-GNinja` or `-G "Unix Makefiles"` and the appropriate Qt, compiler, and GPU flags.

### 4.2 Build

```bash
cmake --build build/windows/win64/manual-msvc --config Release --target dupfinder
```

> **Tip:** Run the above from a Visual Studio Developer Command Prompt so the MSVC generator is available; otherwise you will see errors like `Error: could create CMAKE_GENERATOR "Visual Studio 17 2022"`.

### 4.3 Package

```bash
cmake --build build/... --target package
```

Manual invocations will not copy outputs into `dist/`; do that yourself or re-run `scripts/build.py --skip-package --dry-run` to reuse its copy step.

---

## 5. CPU vs GPU Builds

| Variant | Flag | Behavior |
|---------|------|----------|
| CPU only | `-DDUPFINDER_BUILD_VARIANT=cpu` and `-DENABLE_GPU_ACCELERATION=OFF` | Disables CUDA/OpenCL code paths; fastest compile; compatible everywhere. |
| GPU (CUDA) | `-DDUPFINDER_BUILD_VARIANT=gpu` and `-DENABLE_GPU_ACCELERATION=ON` | Enables CUDA sources, links GPU libraries, and requires a CUDA-ready toolchain/profile. |

When a GPU profile is chosen, the script verifies CUDA availability (`nvidia-smi`, `nvcc`, environment variables). Use `--force` to override detection (not recommended unless you know CUDA is available).

---

## 6. Distribution Layout
After a successful run you will find artifacts such as:
```

---

## 8. Additional Reference Material

### 8.1 Built Targets and Developer Commands

Primary targets:

| Target | Description | Invocation |
|--------|-------------|------------|
| `dupfinder` | Main application executable | `cmake --build ... --target dupfinder` or `make dupfinder` |
| `unit_tests` | Unit test executable | `cmake --build ... --target unit_tests` |
| `integration_tests` | Integration test executable | `cmake --build ... --target integration_tests` |
| `check` | Runs all registered tests (`ctest`) | `cmake --build ... --target check` |
| `package` | Creates installer/package artifacts | `cmake --build ... --target package` |
| `summary` | Prints the CMake build summary | `cmake --build ... --target summary` |

Developer utilities:

| Target | Purpose | Notes |
|--------|---------|-------|
| `format` | Run clang-format over the codebase | Requires `clang-format` in PATH |
| `cppcheck` | Run static analysis | Requires `cppcheck` |
| `memcheck` | Run Valgrind memcheck (Linux) | Requires `valgrind` |
| `coverage` | Generate coverage report (Debug + ENABLE_COVERAGE) | Requires `gcov`, `lcov`, `genhtml` |
| `docs` | Build API docs with Doxygen | Requires `doxygen` |

### 8.2 Key CMake Options

Standard variables:

```bash
-DCMAKE_BUILD_TYPE=Release        # Debug, Release, RelWithDebInfo, MinSizeRel
-DCMAKE_INSTALL_PREFIX=/usr/local # Installation prefix
-DCMAKE_CXX_COMPILER=g++          # Override compiler
-G "Ninja"                         # Explicit generator
```

DupFinder-specific toggles:

```bash
-DDUPFINDER_BUILD_VARIANT=cpu|gpu
-DENABLE_GPU_ACCELERATION=ON|OFF   # Auto-toggled by build profiles
-DDUPFINDER_WARNINGS_AS_ERRORS=ON  # Default ON for MSVC /W4 + /WX
-DDUPFINDER_PACKAGE_SUFFIX=<label> # Labels copied into dist outputs
```

### 8.3 Platform Quick Starts

**Ubuntu / Debian:**

```bash
sudo apt update -qq
sudo apt install -y build-essential cmake ninja-build \
  qt6-base-dev qt6-tools-dev qt6-tools-dev-tools \
  libqt6widgets6 libqt6concurrent6 libqt6network6
```

**Windows (MSVC - Recommended):**

1. Install Visual Studio 2022 Community/Professional/Enterprise
   - Include "Desktop development with C++" workload
   - Ensure Windows 10/11 SDK is selected
2. Install Qt 6.x (6.5+ recommended) MSVC 64-bit using Qt Online Installer
   - Default path: `C:\Qt\6.x.x\msvc2022_64\`
3. Install CMake ≥3.24 and add to PATH
4. Install Git for version control
5. Optional: Install NSIS for packaging (`cpack -C Release`)

**Windows (MinGW - Alternative):**

1. Install Qt 6.x with MinGW 11.2.0 64-bit using Qt Online Installer
   - Includes Qt Creator IDE
2. Install CMake ≥3.24 and add to PATH
3. Install Git

**macOS:**

```bash
brew install cmake ninja qt6
```

### 8.4 Code Quality & Documentation Commands

```bash
# Formatting
cmake --build <build-dir> --target format

# Static analysis
cmake --build <build-dir> --target cppcheck

# Memory analysis (Linux)
cmake --build <build-dir> --target memcheck

# Coverage (Debug + ENABLE_COVERAGE=ON)
cmake --build <build-dir> --target coverage

# API documentation
cmake --build <build-dir> --target docs
```

  Win64/
    Release/
      dupfinder-1.0.0-win64-msvc-cpu.exe
  Linux/
    Release/
      dupfinder-1.0.0-linux-x86_64-cpu.deb
      dupfinder-1.0.0-linux-x86_64-cpu.tgz
  MacOS/
    X64/
      Release/dupfinder-1.0.0-macos-x86_64.dmg
```

The exact suffix is controlled by `DUPFINDER_PACKAGE_SUFFIX` in each profile. Multi-config generators place binaries under the usual `build/.../<Config>/` folder; the script harvests results before copying to `dist/`.

---

## 7. Troubleshooting Checklist

- **Generator errors:** Ensure you are running CMake from the same environment referenced in the profile (`vcvars64.bat`, `Qt6_DIR`).
- **Missing Qt DLLs on Windows:** The post-build step calls `windeployqt` automatically when the script drives the build; verify the Qt path if DLLs are missing.
- **CUDA detection fails:** Confirm `nvcc` (or `nvidia-smi`) is reachable; amend the profile `path` entries or set `CUDA_PATH`.
- **AUTOGEN symbols unresolved:** Confirm you are on the updated CMake that adds the configuration-specific `mocs_compilation_*.cpp` sources (already handled in the repository’s `CMakeLists.txt`).

For a comprehensive option-by-option breakdown, see Section 8 below.

---

## 9. Windows Development Deep Dive

### Windows-Specific CMake Configuration

DupFinder's CMakeLists.txt includes Windows-specific settings:

```cmake
# Platform detection
if(WIN32)
    set(PLATFORM_NAME "windows")
    target_link_libraries(dupfinder shell32 ole32)
    set_target_properties(dupfinder PROPERTIES WIN32_EXECUTABLE TRUE)  # No console window
endif()

# Windows packaging
if(WIN32)
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "DupFinder")
endif()
```

### Required Windows Platform Files

For full Windows integration, implement these platform-specific files:

#### `src/platform/windows/platform_file_ops.cpp`
- Long path support (>260 chars)
- NTFS permissions handling
- Windows file attributes

#### `src/platform/windows/trash_manager.cpp`
- Windows Recycle Bin integration
- Move files to Recycle Bin (not permanent delete)
- Restore and empty Recycle Bin operations

#### `src/platform/windows/system_integration.cpp`
- Windows Explorer context menus
- File associations
- Desktop notifications
- Start menu integration

### Windows-Specific Features Roadmap
1. **Recycle Bin API Integration**
2. **Windows Explorer Context Menus**
3. **File Association Registration**
4. **Windows Toast Notifications**
5. **UWP/App Store Packaging Option**

### Windows Testing Setup

#### Automated Testing
```powershell
# Run tests after build
cd build
ctest --output-on-failure

# Run specific categories
ctest -L unit
ctest -L integration
```

#### Manual Testing Checklist
- [ ] Application launches without console window
- [ ] Qt6 GUI renders correctly
- [ ] File scanning works on Windows paths
- [ ] Basic duplicate detection functions
- [ ] Settings persistence works
- [ ] Window state management works

### Windows Development Workflow

#### Daily Development Cycle
```powershell
# Update code
git pull origin main

# Build and test
cd build
cmake --build . --config Debug --parallel
ctest

# Run application
.\Debug\dupfinder.exe
```

#### Release Build Process
```powershell
# Clean build
rmdir /s /q build
mkdir build
cd build

# Release configuration
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release

# Build optimized version
cmake --build . --config Release --parallel

# Create installer
cpack -C Release
```

### Windows CI/CD Setup

#### GitHub Actions Windows Workflow
```yaml
name: Windows Build
on: [push, pull_request]

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup MSVC
        uses: microsoft/setup-msbuild@v1

      - name: Install Qt6
        uses: jurplel/install-qt-action@v3
        with:
          version: '6.6.*'
          host: 'windows'
          target: 'desktop'
          arch: 'win64_msvc2022_64'

      - name: Configure CMake
        run: cmake -B build -G "Visual Studio 17 2022" -A x64

      - name: Build
        run: cmake --build build --config Release --parallel

      - name: Test
        run: cd build && ctest --output-on-failure

      - name: Package
        run: cd build && cpack -C Release
```

### Common Windows Build Issues

#### Qt6 Not Found
```
CMake Error: Could not find a package configuration file
```
**Solution:**
```powershell
# Set Qt6_DIR environment variable
$env:Qt6_DIR = "C:\Qt\6.x.x\msvc2022_64\lib\cmake\Qt6"
cmake .. -G "Visual Studio 17 2022" -A x64
```

#### MSVC Compiler Issues
```
fatal error C1083: Cannot open include file: 'windows.h'
```
**Solution:** Run from "Developer Command Prompt for VS 2022"

#### DLL Dependencies at Runtime
**Solution:** Use `windeployqt` to copy dependencies:
```powershell
& "C:\Qt\6.x.x\msvc2022_64\bin\windeployqt.exe" .\Release\dupfinder.exe
```

#### NSIS Installer Creation Fails
```
CPack Error: Cannot find NSIS
```
**Solution:**
```powershell
# Add NSIS to PATH or set in CPackConfig.cmake
set(CPACK_NSIS_EXECUTABLE "C:/Program Files (x86)/NSIS/makensis.exe")
cpack -C Release
```

### Windows Performance Considerations

- **MSVC:** Faster compilation than MinGW
- **Parallel Builds:** Use `--parallel` flag
- **Incremental Builds:** CMake detects changes automatically
- **NTFS Optimization:** Use Windows file APIs for large drives
- **Memory Management:** Windows memory limits vs Linux
- **Threading:** Windows thread pool vs Qt threading

### Windows Development Next Steps

#### Immediate Tasks
1. **Install Development Environment**
   - Set up Visual Studio 2022 + Qt6
   - Verify CMake installation

2. **Validate Core Build**
   - Build application successfully
   - Run basic functionality tests
   - Verify Qt6 GUI works on Windows

3. **Implement Platform Integration**
   - Create Windows platform files
   - Add Recycle Bin support
   - Test Windows-specific features

#### Medium-term Goals
1. **Full Windows Testing**
   - Complete test suite on Windows
   - Performance benchmarking
   - User acceptance testing

2. **Windows Packaging**
   - NSIS installer creation
   - Microsoft Store preparation (optional)
   - Auto-update system

3. **Windows-specific Features**
   - Explorer integration
   - Windows notifications
   - File association support

---

## 10. Next Steps

1. Customize `config/build_profiles.json` on each developer workstation.
2. Validate a CPU Release build with `python scripts/build.py --target <profile> --build-type Release`.
3. If GPU hardware is available, repeat with the matching GPU profile.
4. Commit new or updated documentation (like this file) alongside any profile templates you modify.
