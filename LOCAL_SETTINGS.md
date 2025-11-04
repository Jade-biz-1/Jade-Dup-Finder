# Local Development Machine Settings

**Purpose**: This document serves as a reference for configuring build environments across different platforms (Windows, Linux, macOS). It documents what settings need to be configured and provides examples from actual development machines.

**How to Use**:
1. **New developers**: Read this to understand what needs to be installed and configured on your machine
2. **Existing developers**: Update your platform section when you set up a new development machine
3. **Private notes**: Create `LOCAL_SETTINGS.local.md` for your personal notes (not tracked by git)

**What to Commit**:
- ✅ Example paths and configurations (helps others understand what's needed)
- ✅ Platform-specific setup instructions
- ✅ Troubleshooting tips that benefit the team

**What NOT to Commit**:
- ❌ Sensitive information (passwords, tokens, private keys)
- ❌ Personal/proprietary paths if working in a restricted environment

---

## Windows Development Machine (Example Configuration)

**Last Updated**: 2025-11-04
**Operating System**: Windows 11
**Purpose**: Reference configuration showing typical Windows development setup

> **Note**: These are example paths from a working Windows development machine. Adjust paths in your local `config/build_profiles_*.json` files to match your installation locations.

### Qt Installation

| Component | Path | Notes |
|-----------|------|-------|
| Qt 6.9.3 MSVC | `C:\Qt\6.9.3\msvc2022_64` | For Visual Studio 2022 builds |
| Qt 6.9.3 MinGW | `C:\Qt\6.9.3\llvm-mingw_64` | LLVM-MinGW variant |
| Qt Environment Script | `C:\Qt\6.9.3\msvc2022_64\bin\qtenv2.bat` | Sets Qt environment variables |
| Qt CMake Tools | `C:\Qt6\Tools\CMake_64\bin` | Qt-provided CMake (alternative) |

**Usage Note**: For MSVC builds, use `C:\Qt\6.9.3\msvc2022_64` as `Qt6_DIR` in build profiles.

### Visual Studio 2022 Professional

| Component | Path | Notes |
|-----------|------|-------|
| Edition | Professional | VS 2022 v17.x |
| vcvars64.bat | `C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat` | **Must run before building** |
| Visual Studio CMake | `C:\Program Files\CMake\bin\cmake` | Installed with VS |

**Critical Setup Step**: Always run `vcvars64.bat` in your shell before invoking CMake commands. The `build.py` script does this automatically via the `setup_scripts` configuration.

### CMake Installations

**⚠️ IMPORTANT CMake Path Consideration**:

This machine has **two CMake installations**:

1. **Visual Studio CMake** (Recommended for MSVC builds):
   - Path: `C:\Program Files\CMake\bin\cmake`
   - Version: Matches Visual Studio requirements
   - Use this for: Windows MSVC builds

2. **Qt CMake** (Alternative):
   - Path: `C:\Qt6\Tools\CMake_64\bin\cmake`
   - Version: Bundled with Qt
   - Use this for: MinGW builds or if VS CMake has issues

**Best Practice**: Let `build.py` use the CMake from PATH (set by vcvars64.bat for MSVC builds). If you encounter CMake-related errors, verify which CMake is being used:

```powershell
where cmake
```

If needed, explicitly add the desired CMake to the `path` array in your build profile.

### NVIDIA CUDA Toolkit

| Component | Path | Notes |
|-----------|------|-------|
| CUDA 12.5 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA` | Complete installation |
| CUDA Version | 12.5 | Verified working |
| nvcc Compiler | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe` | CUDA compiler |
| CUDA Libraries | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\lib\x64` | Link libraries |

**GPU Detection**: Run `nvidia-smi` to verify GPU availability.

### Environment Setup Sequence

For manual builds (when not using `build.py`):

```powershell
# 1. Set up Visual Studio environment
& "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"

# 2. (Optional) Set up Qt environment
& "C:\Qt\6.9.3\msvc2022_64\bin\qtenv2.bat"

# 3. Verify CMake is accessible
where cmake

# 4. Verify Qt is accessible
echo $env:Qt6_DIR

# 5. (For GPU builds) Verify CUDA
nvcc --version
nvidia-smi
```

**Note**: The `build.py` script handles steps 1-2 automatically when you use the configured build profiles.

### Build Profile Configuration

Update the following build profile files with your paths:

#### For MSVC CPU builds:
File: `config/build_profiles_windows-msvc-cpu.json`
```json
{
  "setup_scripts": [
    "C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\VC\\Auxiliary\\Build\\vcvars64.bat"
  ],
  "environment": {
    "Qt6_DIR": "C:\\Qt\\6.9.3\\msvc2022_64"
  }
}
```

#### For MSVC CUDA builds:
File: `config/build_profiles_windows-msvc-cuda.json`
```json
{
  "setup_scripts": [
    "C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\VC\\Auxiliary\\Build\\vcvars64.bat"
  ],
  "environment": {
    "Qt6_DIR": "C:\\Qt\\6.9.3\\msvc2022_64",
    "CUDA_PATH": "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5"
  },
  "path": [
    "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\bin"
  ]
}
```

#### For MinGW builds:
File: `config/build_profiles_windows-mingw-cpu.json`
```json
{
  "environment": {
    "Qt6_DIR": "C:\\Qt\\6.9.3\\llvm-mingw_64"
  },
  "path": [
    "C:\\Qt\\6.9.3\\llvm-mingw_64\\bin"
  ]
}
```

---

## Linux Development Machine (To Be Documented)

**Status**: ⏳ Awaiting Linux environment setup
**How to Add**: When you set up a Linux development machine, add the configuration here following the Windows example above.

**Information to Document**:

### Distribution and Tools
- Linux distribution and version (e.g., Ubuntu 22.04, Fedora 38)
- GCC/Clang version
- CMake version
- Ninja build system installation

### Qt Installation
- Qt6 installation method (system packages, Qt installer, or compiled from source)
- Qt6 installation path (e.g., `/opt/qt6`, `/usr/lib/x86_64-linux-gnu/cmake/Qt6`)
- Qt6 version

### CUDA Installation (for GPU builds)
- CUDA version
- CUDA installation path (typically `/usr/local/cuda`)
- CUDA bin path (for nvcc)
- GPU verification commands: `nvidia-smi`, `nvcc --version`

### Build Profile Configuration
Update `config/build_profiles_linux-cpu.json` and `config/build_profiles_linux-gpu.json` with your paths.

---

## macOS Development Machine (To Be Documented)

**Status**: ⏳ Awaiting macOS environment setup
**How to Add**: When you set up a macOS development machine, add the configuration here following the Windows example above.

**Information to Document**:

### System Information
- macOS version (e.g., macOS 14 Sonoma)
- Xcode version
- Architecture (Intel x86_64 or Apple Silicon ARM64)

### Homebrew and Tools
- Homebrew installation path (`/opt/homebrew` for ARM, `/usr/local` for Intel)
- CMake installation: `brew install cmake`
- Ninja installation: `brew install ninja`

### Qt Installation
- Qt6 installation method (Homebrew or Qt installer)
- Qt6 installation path:
  - Homebrew Intel: `/usr/local/opt/qt@6`
  - Homebrew ARM: `/opt/homebrew/opt/qt@6`
  - Qt Installer: Custom path
- Qt6 version

### Build Profile Configuration
Update `config/build_profiles_macos-x86_64.json` or `config/build_profiles_macos-arm64.json` with your paths.

---

## Quick Build Commands Reference

### Windows MSVC CPU Build
```powershell
python scripts\build.py --target windows-msvc-cpu --build-type Release
```

### Windows MSVC CUDA Build (GPU)
```powershell
python scripts\build.py --target windows-msvc-cuda --build-type Release
```

### Windows MinGW Build
```powershell
python scripts\build.py --target windows-mingw-cpu --build-type Release
```

### List All Available Targets
```powershell
python scripts\build.py --list-targets
```

---

## Troubleshooting Tips

### Issue: "CMake Error: Could not find Qt6"
**Solution**: Verify Qt6_DIR environment variable or path in build profile:
```powershell
echo $env:Qt6_DIR
# Should output: C:\Qt\6.9.3\msvc2022_64
```

### Issue: "cl.exe not found"
**Solution**: vcvars64.bat not executed. Run it manually or let build.py do it:
```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
```

### Issue: Wrong CMake version used
**Solution**: Check which CMake is first in PATH:
```powershell
where cmake
# First result should be: C:\Program Files\CMake\bin\cmake (for MSVC builds)
```

### Issue: CUDA not detected for GPU build
**Solution**: Verify CUDA_PATH and nvcc accessibility:
```powershell
echo $env:CUDA_PATH
nvcc --version
nvidia-smi
```

---

## Notes for Team Members

1. **This file is version controlled** - It serves as a team reference showing working configurations for each platform
2. **Update when you set up a new platform** - When you configure Linux or macOS, add your working configuration to this file
3. **Use LOCAL_SETTINGS.local.md for private notes** - If you need to keep personal notes that shouldn't be shared, create `LOCAL_SETTINGS.local.md` (ignored by git)
4. **Keep it helpful** - Document what worked for you, including versions and any non-obvious setup steps
5. **Reference before building** - Check this file when setting up a new development machine or troubleshooting build issues

---

## Version History

| Date | Machine | Changes |
|------|---------|---------|
| 2025-11-04 | Windows 11 | Initial configuration documented |
| TBD | Linux | To be added |
| TBD | macOS | To be added |
