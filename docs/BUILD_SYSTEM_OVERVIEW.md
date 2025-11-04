# DupFinder Build System Guide

**Last updated:** 2025-11-04
**Scope:** Desktop (CPU + optional GPU) builds on Windows, Linux, and macOS

---

## 1. Architecture At A Glance

The build pipeline has three cooperating layers:

| Layer | Purpose | Key Files/Locations |
|-------|---------|---------------------|
| **CMake project** | Describes targets, Qt6 AUTOGEN rules, packaging, and tool integrations. | `CMakeLists.txt`, `cmake/` modules, `resources/` |
| **Build profiles** | Capture per-machine generators, toolchain paths, environment tweaks, and output routing. | `config/build_profiles_*.json` (individual per-target configs) OR `config/build_profiles.json` (single-file legacy mode) |
| **Unified script** | Detects the host, selects a profile, runs configure/build/package, and copies artifacts to `dist/`. | `scripts/build.py` |

Supporting directories:

- `build/` — generator-specific build trees (e.g., `build/windows/win64/windows-msvc-cpu`, `build/linux/x64/linux-ninja-cpu`).
- `dist/` — normalized distribution output (see Section 1.1 for complete structure).
- `docs/` — build references and operational guides (this document plus topical guides).

### 1.1 Distribution Folder Structure

All build artifacts are organized in `dist/` following this structure:

```
dist/
├── Win64/
│   ├── Debug/
│   │   └── dupfinder-<version>-win64-<variant>.exe
│   └── Release/
│       └── dupfinder-<version>-win64-<variant>.exe
├── Linux/
│   ├── Debug/
│   │   ├── dupfinder-<version>-linux-x86_64-<variant>.deb
│   │   ├── dupfinder-<version>-linux-x86_64-<variant>.rpm
│   │   └── dupfinder-<version>-linux-x86_64-<variant>.tgz
│   └── Release/
│       ├── dupfinder-<version>-linux-x86_64-<variant>.deb
│       ├── dupfinder-<version>-linux-x86_64-<variant>.rpm
│       └── dupfinder-<version>-linux-x86_64-<variant>.tgz
└── MacOS/
    ├── X64/
    │   ├── Debug/
    │   │   └── dupfinder-<version>-macos-x86_64.dmg
    │   └── Release/
    │       └── dupfinder-<version>-macos-x86_64.dmg
    └── ARM/
        ├── Debug/
        │   └── dupfinder-<version>-macos-arm64.dmg
        └── Release/
            └── dupfinder-<version>-macos-arm64.dmg
```

Where `<variant>` is one of: `msvc-cpu`, `msvc-cuda`, `mingw-cpu` (Windows) or `cpu`, `gpu` (Linux).

### 1.2 How the Three Layers Work Together

Understanding the interaction between CMakeLists.txt, build profiles, and build.py:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER RUNS: python scripts/build.py                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. build.py DISCOVERS AND LOADS BUILD PROFILES                 │
│    - Reads all config/build_profiles_*.json files              │
│    - Detects OS, architecture, GPU availability                │
│    - Filters compatible targets for current system             │
│    - User selects target and build type                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. build.py PREPARES ENVIRONMENT                               │
│    - Sets environment variables from profile.environment       │
│    - Adds profile.path entries to PATH                         │
│    - On Windows: Calls setup_scripts (vcvars64.bat)            │
│    - Creates build directory: build/<os>/<arch>/<target>       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. build.py INVOKES CMAKE CONFIGURE                            │
│    Command: cmake -S . -B build/<path>                         │
│             -G "Visual Studio 17 2022" -A x64                  │
│             -DDUPFINDER_BUILD_VARIANT=cpu                      │
│             -DENABLE_GPU_ACCELERATION=OFF                      │
│             -DDUPFINDER_PACKAGE_SUFFIX=win64-msvc-cpu          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. CMakeLists.txt PROCESSES CONFIGURATION                      │
│    - Finds Qt6 using Qt6_DIR environment variable              │
│    - Detects platform (Windows/Linux/macOS)                    │
│    - Checks ENABLE_GPU_ACCELERATION flag                       │
│    - If GPU enabled: includes GPU source files                 │
│    - If GPU enabled: finds and links CUDA/OpenCL               │
│    - Configures CPack generators (NSIS/DEB/RPM/DMG)            │
│    - Sets package filename using DUPFINDER_PACKAGE_SUFFIX      │
│    - Generates build system files                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. build.py INVOKES CMAKE BUILD                               │
│    Command: cmake --build build/<path>                         │
│             --target dupfinder --parallel                      │
│             --config Release                                   │
│    - Compiles source files                                     │
│    - Links libraries (Qt6, CUDA if enabled)                    │
│    - Post-build: windeployqt copies Qt DLLs (Windows)          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. build.py INVOKES CMAKE PACKAGE                             │
│    Command: cmake --build build/<path> --target package        │
│    - CPack creates platform-specific packages:                 │
│      • Windows: NSIS installer (.exe)                          │
│      • Linux: DEB, RPM, TGZ archives                           │
│      • macOS: DMG disk image                                   │
│    - Package names include version and suffix                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. build.py COPIES ARTIFACTS TO dist/                         │
│    - Detects new packages in build/ directory                  │
│    - Determines target location:                               │
│      • dist/Win64/<build_type>/                                │
│      • dist/Linux/<build_type>/                                │
│      • dist/MacOS/<arch>/<build_type>/                         │
│    - Copies all artifacts to appropriate location              │
│    - Reports success with artifact list                        │
└─────────────────────────────────────────────────────────────────┘
```

**Key Takeaways**:
1. **build.py** = Orchestrator (detects environment, selects profiles, drives build)
2. **build_profiles_*.json** = Configuration (toolchain paths, CMake flags, packaging options)
3. **CMakeLists.txt** = Build logic (source files, dependencies, compilation rules, packaging)

**Data Flow**:
- Build profile JSON → CMake variables → CMakeLists.txt logic → Compiled binaries → Packages → dist/

### 1.3 Build System Requirements Addressed

The DupFinder build system has been designed to meet the following requirements:

✅ **Requirement 1: Organized Build Folders**
- Separate build folders per platform: `build/windows/`, `build/linux/`, `build/macos/`
- Sub-folders for architecture: `build/windows/win64/`, `build/macos/x64/`, `build/macos/arm64/`
- Target-specific folders: `build/windows/win64/windows-msvc-cpu/`, etc.
- Prevents overwriting artifacts between different build configurations
- Enables repeatable builds by preserving build state

✅ **Requirement 2: Linux Multi-Format Packaging**
- Linux CPU build produces: DEB, RPM, and TGZ packages
- Linux GPU build produces: DEB, RPM, and TGZ packages
- All formats generated automatically via CPack
- Configured in `CMakeLists.txt` and `build_profiles_linux-*.json`

✅ **Requirement 3: Windows CPU/GPU Support**
- Windows CPU build: Available via MSVC or MinGW toolchains
- Windows GPU build: Available via MSVC + NVIDIA CUDA
- Separate configuration files for each variant

✅ **Requirement 4: Windows GPU MSVC+CUDA Support**
- Dedicated `build_profiles_windows-msvc-cuda.json` configuration
- Integrated CUDA detection via `nvidia-smi` and `nvcc`
- Automatic environment setup with `vcvars64.bat` and CUDA paths

✅ **Requirement 5: Windows CPU Multi-Toolchain Support**
- MSVC toolchain: `build_profiles_windows-msvc-cpu.json`
- MinGW toolchain: `build_profiles_windows-mingw-cpu.json`
- Users can choose based on preference and licensing needs

✅ **Requirement 6: macOS x86/ARM Support**
- Intel (x86_64): `build_profiles_macos-x86_64.json`
- Apple Silicon (ARM64): `build_profiles_macos-arm64.json`
- Native compilation for optimal performance on each architecture

✅ **Requirement 7: OS/GPU Detection with User Confirmation**
- Automatic OS and architecture detection
- CUDA GPU detection via `nvidia-smi`, `nvcc`, and environment variables
- Interactive mode shows detected environment and asks for confirmation
- User sees: OS, architecture, GPU status, build configuration before proceeding

✅ **Requirement 8: Organized Distribution Folder**
- Structured `dist/` folder matching specification:
  - `dist/Win64/{Debug,Release}/`
  - `dist/Linux/{Debug,Release}/` (contains DEB, RPM, TGZ)
  - `dist/MacOS/{X64,ARM}/{Debug,Release}/`
- Automatic artifact copying after successful builds

✅ **Requirement 9: Configuration Management**
- Build dependencies stored in configuration files (not hardcoded)
- Paths for Qt6, CUDA, Visual Studio, etc. in JSON profiles
- Multi-file approach: One file per target for easy management
- Legacy single-file mode also supported
- No need to edit build scripts for different environments

✅ **Requirement 10: One-Command Build Process**
- Single command: `python scripts/build.py`
- Auto-detects platform and shows compatible targets
- Handles configuration, build, packaging, and artifact distribution
- Non-interactive mode for CI/CD: `python scripts/build.py --target <id> --build-type Release --non-interactive`
- Platform-specific builds require only changing the target ID

---

## 2. Preparing A Machine

### 2.1 Install Platform Toolchain

1. **Windows (MSVC)**:
   - Visual Studio 2022 (Community/Professional/Enterprise) with "Desktop development with C++" workload
   - Ensure Windows 10/11 SDK is selected
   - Note the vcvars64.bat path (typically `C:\Program Files\Microsoft Visual Studio\2022\<Edition>\VC\Auxiliary\Build\vcvars64.bat`)

2. **Windows (MinGW)** - Alternative to MSVC:
   - MinGW-w64 toolchain (11.2.0 or later)
   - Ensure `mingw64\bin` is in your PATH or noted for configuration

3. **Linux**:
   - GCC/Clang compiler
   - CMake ≥ 3.24
   - Ninja build system (recommended): `sudo apt install ninja-build`
   - Qt6 development packages
   - Packaging tools: `sudo apt install dpkg-dev rpm`

4. **macOS**:
   - Xcode command-line tools: `xcode-select --install`
   - CMake: `brew install cmake`
   - Ninja: `brew install ninja`
   - Homebrew Qt6: `brew install qt@6`

5. **GPU Builds (Optional)**:
   - **NVIDIA CUDA**: Install CUDA Toolkit 12.x or later from nvidia.com
   - Note the CUDA installation path (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5` on Windows)
   - Verify installation: `nvidia-smi` and `nvcc --version`

### 2.2 Install Qt6

Install Qt6 binaries matching your compiler:
- **Windows MSVC**: Qt6 with MSVC 2022 64-bit (e.g., `C:\Qt6\6.9.3\msvc2022_64`)
- **Windows MinGW**: Qt6 with MinGW 64-bit (e.g., `C:\Qt\6.5.0\mingw_64`)
- **Linux**: System packages or custom install (e.g., `/opt/qt6`)
- **macOS**: Homebrew (`/opt/homebrew/opt/qt` on ARM, `/usr/local/opt/qt` on Intel)

### 2.3 Configure Build Profiles

**Quick Start**: Check `LOCAL_SETTINGS.md` in the repository root for reference configurations showing what needs to be set up on each platform. This file contains example paths from working development machines and is maintained by the team.

The build system supports two configuration modes:

#### Option A: Multi-File Mode (Recommended)

Create individual profile files for each target you'll use. Pre-configured templates are provided in `config/`:

```bash
# Windows users - Edit these files with your local paths:
config/build_profiles_windows-msvc-cpu.json
config/build_profiles_windows-msvc-cuda.json
config/build_profiles_windows-mingw-cpu.json

# Linux users - Edit these files:
config/build_profiles_linux-cpu.json
config/build_profiles_linux-gpu.json

# macOS users - Edit these files:
config/build_profiles_macos-x86_64.json  # Intel Macs
config/build_profiles_macos-arm64.json   # Apple Silicon
```

Update the paths in each file you'll use:
- `setup_scripts`: Path to `vcvars64.bat` (Windows MSVC only)
- `environment.Qt6_DIR`: Your Qt6 installation path
- `environment.CUDA_PATH` or `CUDA_HOME`: CUDA installation path (GPU builds only)
- `path`: Additional paths for tools (CUDA bin, etc.)

#### Option B: Single-File Mode (Legacy)

```bash
cp config/build_profiles.example.json config/build_profiles.json
# Edit build_profiles.json with your local paths
```

### 2.4 Path Configuration Quick Reference

| Platform | Setting | Typical Value | Notes |
|----------|---------|---------------|-------|
| Windows MSVC | `Qt6_DIR` | `C:\Qt6\6.9.3\msvc2022_64` | Must match your Qt installation |
| Windows MSVC | `setup_scripts` | `C:\Program Files\Microsoft Visual Studio\2022\<Edition>\VC\Auxiliary\Build\vcvars64.bat` | Adjust for Community/Professional/Enterprise |
| Windows MSVC GPU | `CUDA_PATH` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5` | Required for GPU builds |
| Windows MinGW | `Qt6_DIR` | `C:\Qt\6.5.0\mingw_64` | Must match MinGW Qt build |
| Windows MinGW | `path` | `C:\mingw64\bin` | MinGW toolchain location |
| Linux | `Qt6_DIR` | `/opt/qt6` or `/usr/lib/x86_64-linux-gnu/cmake/Qt6` | System packages or custom install |
| Linux GPU | `CUDA_HOME` | `/usr/local/cuda` | CUDA toolkit path |
| macOS Intel | `Qt6_DIR` | `/usr/local/opt/qt` | Homebrew Intel path |
| macOS ARM | `Qt6_DIR` | `/opt/homebrew/opt/qt` | Homebrew ARM path |

---

## 3. Using `scripts/build.py`

Key features:

- Auto-discovers and loads all `build_profiles_*.json` files (or single `build_profiles.json`)
- Detects OS/architecture and auto-filters valid targets
- Detects CUDA availability and highlights GPU-only profiles
- Shows user the detected environment and selected configuration before building
- Supports interactive or non-interactive operation
- Runs configure, build, optional `cpack`, then mirrors artifacts into `dist/`

### 3.1 List Available Targets

Before building, see what targets are configured for your machine:

```bash
python scripts/build.py --list-targets
```

Example output:
```
Configured build targets:
- windows-msvc-cpu: Windows 64-bit (MSVC, CPU build) (Windows) [from build_profiles_windows-msvc-cpu.json]
- windows-msvc-cuda: Windows 64-bit (MSVC, CUDA GPU build) [GPU] (Windows) [from build_profiles_windows-msvc-cuda.json]
- windows-mingw-cpu: Windows 64-bit (MinGW, CPU build) (Windows) [from build_profiles_windows-mingw-cpu.json]
```

### 3.2 Interactive Flow (Recommended for First Build)

```bash
python scripts/build.py
```

You will be prompted to:

1. Choose a matching build target (only targets compatible with your OS/arch are shown)
2. Select the build type (Debug or Release)
3. Review the environment summary:
   - Operating system and architecture
   - GPU availability (detected via nvidia-smi/nvcc)
   - Selected target and generator
   - Build configuration
4. Confirm to proceed

The script automatically:

- Creates `build/<os>/<arch>/<target>/` and configures CMake
- Runs the correct generator (`cmake --build ...`)
- Invokes `cpack` to create installers (unless `--skip-package` is passed)
- Copies new artifacts into `dist/<platform>/<build_type>/`

Example session:
```
Environment summary:
  Operating system : Windows
  Architecture     : AMD64
  GPU              : Detected (required for this build)
  GPU details      : GPU 0: NVIDIA GeForce RTX 3080

Build configuration:
  Target           : Windows 64-bit (MSVC, CUDA GPU build) [windows-msvc-cuda]
  Notes            : Requires NVIDIA CUDA toolkit with nvcc and nvidia-smi available.
  Build type       : Release
  Generator        : Visual Studio 17 2022
  Multi-config     : Yes

Proceed with this configuration? [Y/n]:
```

### 3.3 CLI Flags (for Automation/CI)

For automated builds or CI/CD pipelines:

```bash
python scripts/build.py \
  --target windows-msvc-cpu \
  --build-type Release \
  --non-interactive --force
```

Useful options:

| Flag | Description |
|------|-------------|
| `--list-targets` | List all configured build targets and exit |
| `--target <id>` | Explicitly select a build target by ID |
| `--build-type <type>` | Override build type (Debug or Release) |
| `--clean` | Delete the build directory before configuring |
| `--skip-package` | Build binaries only, skip CPack packaging |
| `--dry-run` | Show planned commands without executing |
| `--non-interactive` | Use defaults, don't prompt for input |
| `--force` | Proceed without confirmation prompts |

Environment variables declared in a profile (`environment`) are applied before invoking any commands, and `setup_scripts` are executed automatically on Windows shells.

### 3.4 Platform-Specific Examples

#### Windows MSVC CPU Build
```bash
# Interactive
python scripts/build.py

# Non-interactive
python scripts/build.py --target windows-msvc-cpu --build-type Release --non-interactive
```

#### Windows MSVC CUDA GPU Build
```bash
# Requires CUDA toolkit installed and nvidia-smi/nvcc available
python scripts/build.py --target windows-msvc-cuda --build-type Release --non-interactive
```

#### Linux CPU Build
```bash
# Interactive
python3 scripts/build.py

# Non-interactive - produces DEB, RPM, and TGZ
python3 scripts/build.py --target linux-ninja-cpu --build-type Release --non-interactive
```

#### macOS Apple Silicon Build
```bash
# Interactive
python3 scripts/build.py

# Non-interactive - produces DMG
python3 scripts/build.py --target macos-ninja-arm64 --build-type Release --non-interactive
```

---

## 4. Supported Platforms and Build Variants

### 4.1 Platform Support Matrix

| Platform | Architecture | CPU Build | GPU Build | Package Formats | Build Profile Files |
|----------|--------------|-----------|-----------|-----------------|---------------------|
| **Windows** | x64 (AMD64) | ✅ MSVC<br>✅ MinGW | ✅ MSVC + CUDA | NSIS Installer (.exe) | `build_profiles_windows-msvc-cpu.json`<br>`build_profiles_windows-msvc-cuda.json`<br>`build_profiles_windows-mingw-cpu.json` |
| **Linux** | x86_64 | ✅ GCC/Clang | ✅ GCC/Clang + CUDA | DEB, RPM, TGZ | `build_profiles_linux-cpu.json`<br>`build_profiles_linux-gpu.json` |
| **macOS** | x86_64 (Intel)<br>ARM64 (Apple Silicon) | ✅ Clang<br>✅ Clang | ❌<br>❌ | DMG, PKG | `build_profiles_macos-x86_64.json`<br>`build_profiles_macos-arm64.json` |

### 4.2 Windows Build Variants

#### MSVC + CPU (Recommended for most users)
- **Target ID**: `windows-msvc-cpu`
- **Toolchain**: Visual Studio 2022 MSVC compiler
- **GPU Support**: No
- **Advantages**: Best Windows integration, official Microsoft toolchain, easier debugging
- **Requirements**: Visual Studio 2022 with "Desktop development with C++" workload

#### MSVC + CUDA (For NVIDIA GPU acceleration)
- **Target ID**: `windows-msvc-cuda`
- **Toolchain**: Visual Studio 2022 MSVC compiler + NVIDIA CUDA
- **GPU Support**: Yes (NVIDIA CUDA 12.x or later)
- **Advantages**: GPU-accelerated hash calculations, significantly faster for large files
- **Requirements**:
  - Visual Studio 2022 with "Desktop development with C++" workload
  - NVIDIA CUDA Toolkit 12.x or later
  - NVIDIA GPU with CUDA support

#### MinGW + CPU (Alternative open-source toolchain)
- **Target ID**: `windows-mingw-cpu`
- **Toolchain**: MinGW-w64 GCC compiler
- **GPU Support**: No
- **Advantages**: Open-source toolchain, no Visual Studio license required
- **Requirements**: MinGW-w64 toolchain (11.2.0 or later)
- **Note**: MinGW builds may have different runtime dependencies than MSVC builds

### 4.3 Linux Build Variants

#### CPU Build (Standard)
- **Target ID**: `linux-ninja-cpu`
- **Toolchain**: System GCC or Clang
- **GPU Support**: No
- **Package Formats**: DEB (Debian/Ubuntu), RPM (RedHat/Fedora/CentOS), TGZ (universal)
- **Requirements**: GCC/Clang, CMake, Ninja, Qt6 dev packages, dpkg-dev, rpm

#### GPU Build (NVIDIA CUDA)
- **Target ID**: `linux-ninja-gpu`
- **Toolchain**: System GCC or Clang + NVIDIA CUDA
- **GPU Support**: Yes (NVIDIA CUDA 11.x or later)
- **Package Formats**: DEB, RPM, TGZ
- **Requirements**:
  - All CPU build requirements
  - NVIDIA CUDA Toolkit 11.x or later
  - NVIDIA GPU drivers

### 4.4 macOS Build Variants

#### Intel (x86_64)
- **Target ID**: `macos-ninja-x86_64`
- **Architecture**: x86_64 (Intel processors)
- **Toolchain**: Xcode Clang
- **GPU Support**: No (macOS doesn't support CUDA)
- **Package Formats**: DMG (disk image), PKG (installer package)
- **Requirements**: Xcode command-line tools, Homebrew, Qt6

#### Apple Silicon (ARM64)
- **Target ID**: `macos-ninja-arm64`
- **Architecture**: ARM64 (M1, M2, M3, M4 processors)
- **Toolchain**: Xcode Clang (ARM native)
- **GPU Support**: No (macOS doesn't support CUDA)
- **Package Formats**: DMG, PKG
- **Requirements**: Xcode command-line tools, Homebrew, Qt6 ARM build
- **Note**: Uses native ARM compilation for optimal performance on Apple Silicon

### 4.5 Build Variant Selection Guide

**Choose your build variant based on:**

1. **Operating System**: Automatically detected by `build.py`
2. **GPU Availability**:
   - If you have NVIDIA GPU → Use GPU variant for better performance
   - No GPU or non-NVIDIA → Use CPU variant
3. **Windows Toolchain Preference**:
   - Professional development → MSVC (better debugging, Microsoft support)
   - Open-source preference → MinGW (no Visual Studio license needed)
4. **macOS Architecture**:
   - Intel Mac → x86_64 variant
   - Apple Silicon Mac → ARM64 variant for native performance

**Performance Comparison** (approximate, varies by hardware):

| Operation | CPU Build | GPU Build (CUDA) | Speedup |
|-----------|-----------|------------------|---------|
| SHA-256 hashing (1 GB file) | ~2-3 seconds | ~0.3-0.5 seconds | 4-10x faster |
| Scanning 10,000 files | ~1-2 minutes | ~10-30 seconds | 2-6x faster |

---

## 5. Manual CMake Workflows (Fallback)

You can still interact with CMake directly when needed (e.g., custom debugging).

### 5.1 Configure

```bash
cmake -S . -B build/windows/win64/manual-msvc ^
  -G "Visual Studio 17 2022" -A x64 ^
  -DDUPFINDER_BUILD_VARIANT=cpu ^
  -DENABLE_GPU_ACCELERATION=OFF
```

Linux/macOS use `-GNinja` or `-G "Unix Makefiles"` and the appropriate Qt, compiler, and GPU flags.

### 5.2 Build

```bash
cmake --build build/windows/win64/manual-msvc --config Release --target dupfinder
```

> **Tip:** Run the above from a Visual Studio Developer Command Prompt so the MSVC generator is available; otherwise you will see errors like `Error: could create CMAKE_GENERATOR "Visual Studio 17 2022"`.

### 5.3 Package

```bash
cmake --build build/... --target package
```

Manual invocations will not copy outputs into `dist/`; do that yourself or re-run `scripts/build.py --skip-package --dry-run` to reuse its copy step.

---

## 6. CPU vs GPU Builds

| Variant | Flag | Behavior |
|---------|------|----------|
| CPU only | `-DDUPFINDER_BUILD_VARIANT=cpu` and `-DENABLE_GPU_ACCELERATION=OFF` | Disables CUDA/OpenCL code paths; fastest compile; compatible everywhere. |
| GPU (CUDA) | `-DDUPFINDER_BUILD_VARIANT=gpu` and `-DENABLE_GPU_ACCELERATION=ON` | Enables CUDA sources, links GPU libraries, and requires a CUDA-ready toolchain/profile. |

When a GPU profile is chosen, the script verifies CUDA availability (`nvidia-smi`, `nvcc`, environment variables). Use `--force` to override detection (not recommended unless you know CUDA is available).

### 6.1 How Build Variants Work

The build system uses CMake variables to control build variants:

1. **Build Profile Sets Variables**: Each `build_profiles_*.json` file specifies CMake arguments:
   ```json
   "cmake_args": [
     "-DDUPFINDER_BUILD_VARIANT=cpu",
     "-DENABLE_GPU_ACCELERATION=OFF",
     "-DDUPFINDER_PACKAGE_SUFFIX=win64-msvc-cpu"
   ]
   ```

2. **CMakeLists.txt Processes Variables**: The root `CMakeLists.txt` uses these to:
   - Enable/disable GPU source files
   - Link appropriate libraries (CUDA, OpenCL, or none)
   - Configure compiler flags
   - Set package naming via `DUPFINDER_PACKAGE_SUFFIX`

3. **Conditional Source Compilation**:
   ```cmake
   if(ENABLE_GPU_ACCELERATION)
       list(APPEND CORE_SOURCES
           src/gpu/gpu_detector.cpp
           src/gpu/gpu_hash_calculator.cpp
           src/gpu/gpu_context.cpp
       )
   endif()
   ```

4. **Package Naming**: The `DUPFINDER_PACKAGE_SUFFIX` from build profiles becomes part of the filename:
   - `dupfinder-1.0.0-windows-win64-msvc-cpu.exe`
   - `dupfinder-1.0.0-linux-x86_64-gpu.deb`

---

## 7. Packaging and Distribution

### 7.1 CPack Integration

The build system uses CPack (part of CMake) to generate platform-specific installers and packages. Configuration is in `CMakeLists.txt`:

#### Windows Packaging (NSIS Installer)

```cmake
# CMakeLists.txt configures Windows installer
set(CPACK_GENERATOR "NSIS")
set(CPACK_NSIS_DISPLAY_NAME "DupFinder")
set(CPACK_NSIS_MENU_LINKS
    "bin/dupfinder.exe" "DupFinder"
    "https://github.com/Jade-biz-1/Jade-Dup-Finder" "DupFinder on GitHub"
)
# Creates Start Menu shortcuts and desktop icon
set(CPACK_NSIS_CREATE_ICONS_EXTRA
    "CreateShortCut '$DESKTOP\\\\DupFinder.lnk' '$INSTDIR\\\\bin\\\\dupfinder.exe'"
)
```

**Requirements**: NSIS (Nullsoft Scriptable Install System) must be installed
- Download from: https://nsis.sourceforge.io/
- CMake auto-detects NSIS in standard locations
- If NSIS not found, falls back to ZIP packaging

**Output**: `dupfinder-<version>-windows-<suffix>.exe` (self-extracting installer)

#### Linux Packaging (DEB, RPM, TGZ)

```cmake
# CMakeLists.txt configures multi-format Linux packages
set(CPACK_GENERATOR "DEB;RPM;TGZ")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "DupFinder Team")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
```

**Requirements**:
- DEB format: `dpkg-deb` (install with `sudo apt install dpkg-dev`)
- RPM format: `rpmbuild` (install with `sudo apt install rpm` or `sudo yum install rpm-build`)
- TGZ format: Built-in tar/gzip (always available)

**Output**: Three packages per build:
- `dupfinder-<version>-linux-<suffix>.deb` (Debian/Ubuntu)
- `dupfinder-<version>-linux-<suffix>.rpm` (RedHat/Fedora/CentOS)
- `dupfinder-<version>-linux-<suffix>.tgz` (Universal tarball)

#### macOS Packaging (DMG)

```cmake
# CMakeLists.txt configures macOS disk image
set(CPACK_GENERATOR "DragNDrop")
set(CPACK_DMG_VOLUME_NAME "DupFinder")
```

**Requirements**: macOS native tools (included with Xcode)

**Output**: `dupfinder-<version>-macos-<suffix>.dmg` (disk image for drag-and-drop installation)

### 7.2 Package File Naming

Package names are automatically generated using this pattern:

```
<PackageName>-<Version>-<Platform>-<Suffix>.<Extension>
```

Components:
- **PackageName**: `dupfinder` (set by `CPACK_PACKAGE_NAME`)
- **Version**: `1.0.0` (from `project(DupFinder VERSION 1.0.0)`)
- **Platform**: `windows`, `linux`, or `macos` (auto-detected by `CMakeLists.txt`)
- **Suffix**: From `DUPFINDER_PACKAGE_SUFFIX` in build profile (e.g., `win64-msvc-cpu`, `linux-x86_64-gpu`)
- **Extension**: `.exe`, `.deb`, `.rpm`, `.tgz`, or `.dmg`

Example filenames:
```
dupfinder-1.0.0-windows-win64-msvc-cpu.exe
dupfinder-1.0.0-windows-win64-msvc-cuda.exe
dupfinder-1.0.0-linux-x86_64-cpu.deb
dupfinder-1.0.0-linux-x86_64-gpu.rpm
dupfinder-1.0.0-macos-arm64.dmg
```

### 7.3 Distribution Layout

After a successful build, artifacts are automatically copied to the `dist/` directory organized by platform and build type (see Section 1.1 for complete structure).

Example artifacts:
```
dist/
├── Win64/Release/dupfinder-1.0.0-win64-msvc-cpu.exe
├── Linux/Release/
│   ├── dupfinder-1.0.0-linux-x86_64-cpu.deb
│   ├── dupfinder-1.0.0-linux-x86_64-cpu.rpm
│   └── dupfinder-1.0.0-linux-x86_64-cpu.tgz
└── MacOS/ARM/Release/dupfinder-1.0.0-macos-arm64.dmg
```

The `build.py` script automatically:
1. Builds the application
2. Invokes CPack to create packages
3. Detects new artifacts in `build/` directory
4. Copies them to the appropriate `dist/` subdirectory

### 7.4 Customizing Package Content

To modify what gets included in packages, edit `CMakeLists.txt`:

```cmake
# Install rules (what goes into the package)
install(TARGETS dupfinder
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION .
)

# Install Qt6 dependencies (Windows)
if(WIN32)
    # windeployqt automatically called post-build
    # Copies required Qt DLLs, plugins, and resources
endif()

# Install additional resources
install(FILES README.md LICENSE DESTINATION .)
install(DIRECTORY resources/ DESTINATION resources)
```

### 7.5 Package Installation Behavior

**Windows NSIS Installer**:
- Default install location: `C:\Program Files\DupFinder\`
- Creates Start Menu shortcuts: "DupFinder" group
- Creates Desktop shortcut (optional during install)
- Adds to Windows "Add/Remove Programs"
- Uninstaller included: `uninstall.exe`

**Linux DEB Package**:
- Installs to: `/usr/local/bin/dupfinder`
- Integration with system package manager: `dpkg -i dupfinder*.deb`
- Remove with: `sudo apt remove dupfinder`

**Linux RPM Package**:
- Installs to: `/usr/local/bin/dupfinder`
- Integration with system package manager: `sudo rpm -i dupfinder*.rpm` or `sudo yum install dupfinder*.rpm`
- Remove with: `sudo rpm -e dupfinder`

**Linux TGZ Archive**:
- Manual extraction: `tar -xzf dupfinder*.tgz`
- Portable installation, run from any directory
- No system integration

**macOS DMG**:
- Opens disk image showing application icon
- User drags to Applications folder
- Standard macOS application bundle
- Remove by dragging to Trash

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

## 9. Troubleshooting Checklist

- **Generator errors:** Ensure you are running CMake from the same environment referenced in the profile (`vcvars64.bat`, `Qt6_DIR`).
- **Missing Qt DLLs on Windows:** The post-build step calls `windeployqt` automatically when the script drives the build; verify the Qt path if DLLs are missing.
- **CUDA detection fails:** Confirm `nvcc` (or `nvidia-smi`) is reachable; amend the profile `path` entries or set `CUDA_PATH`.
- **AUTOGEN symbols unresolved:** Confirm you are on the updated CMake that adds the configuration-specific `mocs_compilation_*.cpp` sources (already handled in the repository’s `CMakeLists.txt`).

For a comprehensive option-by-option breakdown, see Section 8 below.

---

## 10. Windows Development Deep Dive

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

## 11. Next Steps

### For New Developers

1. **Read LOCAL_SETTINGS.md**: Start by reading `LOCAL_SETTINGS.md` in the repository root. It contains reference configurations from working development machines for all platforms and explains what needs to be installed.

2. **Choose your configuration approach**:
   - **Multi-file mode (recommended)**: Edit individual `config/build_profiles_<target>.json` files for platforms you'll build on
   - **Single-file mode**: Copy `config/build_profiles.example.json` to `config/build_profiles.json` and edit

3. **Update paths** in your chosen configuration files to match YOUR installation:
   - Qt6 installation path (`Qt6_DIR`)
   - Visual Studio vcvars64.bat path (Windows MSVC only)
   - CUDA toolkit path (GPU builds only)

4. **Validate your configuration**:
   ```bash
   # List available targets
   python scripts/build.py --list-targets

   # Run your first build
   python scripts/build.py
   ```

5. **Test a CPU Release build**:
   ```bash
   python scripts/build.py --target <your-platform-target> --build-type Release
   ```

6. **If you have NVIDIA GPU**, test a GPU build:
   ```bash
   python scripts/build.py --target <gpu-target> --build-type Release
   ```

### For CI/CD Setup

1. Create platform-specific CI jobs for each target platform
2. Use non-interactive mode with explicit targets:
   ```bash
   python scripts/build.py --target <target-id> --build-type Release --non-interactive --force
   ```
3. Archive `dist/` artifacts after successful builds

### For Build System Maintenance

1. Keep build profile templates (`build_profiles_*.json`) in version control
2. Update this documentation when adding new build targets or platforms
3. Test all supported platforms before major releases
4. Monitor build times and artifact sizes across platforms

---

## 12. Build System Enhancements (November 2025)

### 12.1 Multi-File Configuration System

**Enhancement**: Transitioned from single `build_profiles.json` to individual per-target configuration files.

**Benefits**:
- **Easier Management**: Each platform/variant has its own dedicated configuration file
- **Version Control Friendly**: Team members can update only the profiles they use
- **Reduced Conflicts**: Multiple developers can work on different platform configs simultaneously
- **Better Organization**: Clear separation between Windows, Linux, and macOS configurations
- **Selective Deployment**: CI/CD pipelines can use only the profiles they need

**Files Added**:
```
config/
├── build_profiles_windows-msvc-cpu.json     # Windows MSVC CPU-only
├── build_profiles_windows-msvc-cuda.json    # Windows MSVC + NVIDIA CUDA
├── build_profiles_windows-mingw-cpu.json    # Windows MinGW CPU-only
├── build_profiles_linux-cpu.json            # Linux CPU-only (DEB/RPM/TGZ)
├── build_profiles_linux-gpu.json            # Linux NVIDIA CUDA (DEB/RPM/TGZ)
├── build_profiles_macos-x86_64.json         # macOS Intel (DMG)
└── build_profiles_macos-arm64.json          # macOS Apple Silicon (DMG)
```

**Backward Compatibility**: The legacy single-file `build_profiles.json` continues to work. The system auto-detects which mode is in use.

### 12.2 Enhanced build.py Script

**Improvements**:
- Auto-discovery of all `build_profiles_*.json` files in `config/` directory
- Shows which file each target is loaded from when using `--list-targets`
- Better error messages when no configuration is found
- Supports both multi-file and single-file configuration modes simultaneously

**Example Output**:
```bash
$ python scripts/build.py --list-targets
Configured build targets:
- windows-msvc-cpu: Windows 64-bit (MSVC, CPU build) (Windows) [from build_profiles_windows-msvc-cpu.json]
- windows-msvc-cuda: Windows 64-bit (MSVC, CUDA GPU build) [GPU] (Windows) [from build_profiles_windows-msvc-cuda.json]
- linux-ninja-cpu: Linux x86_64 (Ninja, CPU build) (Linux) [from build_profiles_linux-cpu.json]
- linux-ninja-gpu: Linux x86_64 (Ninja, CUDA GPU build) [GPU] (Linux) [from build_profiles_linux-gpu.json]
- macos-ninja-x86_64: macOS x86_64 (Ninja) (Darwin) [from build_profiles_macos-x86_64.json]
- macos-ninja-arm64: macOS arm64 (Ninja) (Darwin) [from build_profiles_macos-arm64.json]
```

### 12.3 Comprehensive Documentation Updates

**Additions to BUILD_SYSTEM_OVERVIEW.md**:
1. **Section 1.1**: Complete distribution folder structure specification
2. **Section 1.2**: Visual flow diagram showing how all three layers interact
3. **Section 1.3**: Checklist of all 10 requirements addressed
4. **Section 2**: Expanded platform setup guide with path reference table
5. **Section 3**: Enhanced usage examples for all platforms
6. **Section 4**: Platform support matrix with all variants
7. **Section 6.1**: Explanation of how build variants work internally
8. **Section 7**: Complete CPack packaging guide with installation behavior

### 12.4 Configuration Path Reference

Quick reference table added (Section 2.4) documenting typical paths for:
- Qt6 installations on all platforms
- Visual Studio vcvars64.bat locations
- CUDA toolkit paths
- Homebrew Qt6 paths (macOS)

### 12.5 Verification Checklist

All 10 original requirements verified and documented:

| # | Requirement | Status | Implementation |
|---|-------------|--------|----------------|
| 1 | Organized build folders | ✅ Complete | `build/{platform}/{arch}/{target}/` |
| 2 | Linux DEB/RPM/TGZ | ✅ Complete | CPack multi-generator |
| 3 | Windows CPU/GPU | ✅ Complete | Separate MSVC profiles |
| 4 | Windows MSVC+CUDA | ✅ Complete | `build_profiles_windows-msvc-cuda.json` |
| 5 | Windows MSVC/MinGW | ✅ Complete | Two CPU profile options |
| 6 | macOS x86/ARM | ✅ Complete | Separate architecture profiles |
| 7 | OS/GPU detection + confirmation | ✅ Complete | Interactive build flow |
| 8 | Organized dist/ folder | ✅ Complete | Automatic artifact copying |
| 9 | Configuration management | ✅ Complete | JSON profiles, no hardcoding |
| 10 | One-command build | ✅ Complete | `python scripts/build.py` |

### 12.6 Migration Guide

**From Single-File to Multi-File Configuration**:

If you're currently using `build_profiles.json`, you can migrate to multi-file mode:

1. **Identify targets you use**: Look at your current `build_profiles.json` to see which targets you've configured
2. **Copy to individual files**: Split each target into its own `build_profiles_<target>.json` file
3. **Keep or remove build_profiles.json**: You can keep both; the system loads from all files
4. **Test**: Run `python scripts/build.py --list-targets` to verify all targets are detected

**No migration required**: Your existing `build_profiles.json` continues to work as-is.
