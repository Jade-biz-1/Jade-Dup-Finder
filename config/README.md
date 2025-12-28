# Build Profile Configuration

This directory contains build profile configurations for the CloneClean multi-platform build system.

## Quick Start

1. **Check the reference settings**: See `../LOCAL_SETTINGS.md` for example configurations and what needs to be set up
2. **Edit the profiles you need**: Update paths in the `build_profiles_*.json` files to match YOUR installation locations
3. **Verify configuration**: Run `python scripts/build.py --list-targets` to see available targets

> **Important**: The `build_profiles_*.json` files contain example paths. You MUST update them to match your actual installation locations before building.

## Files in This Directory

### Build Profile Files (Edit These!)

| File | Platform | Description |
|------|----------|-------------|
| `build_profiles_windows-msvc-cpu.json` | Windows | MSVC CPU-only build |
| `build_profiles_windows-msvc-cuda.json` | Windows | MSVC + NVIDIA CUDA GPU build |
| `build_profiles_windows-mingw-cpu.json` | Windows | MinGW CPU-only build (LLVM variant) |
| `build_profiles_linux-cpu.json` | Linux | CPU-only, produces DEB/RPM/TGZ |
| `build_profiles_linux-gpu.json` | Linux | NVIDIA CUDA GPU build |
| `build_profiles_macos-x86_64.json` | macOS | Intel Mac (x86_64) |
| `build_profiles_macos-arm64.json` | macOS | Apple Silicon (ARM64) |

### Template Files (Reference Only)

| File | Purpose |
|------|---------|
| `build_profiles.example.json` | Single-file configuration template (legacy) |

**Note**: The individual `build_profiles_*.json` files above already contain configuration for the primary development machine. You may need to adjust paths for your local environment.

## Configuration Modes

### Multi-File Mode (Current/Recommended)
- Each platform/variant has its own JSON file
- Easier to manage and version control
- Edit only the files for platforms you build on
- The `build.py` script auto-discovers all `build_profiles_*.json` files

### Single-File Mode (Legacy)
- All targets in one `build_profiles.json` file
- Copy `build_profiles.example.json` to `build_profiles.json`
- Still supported for backward compatibility

**Both modes can coexist**: The build system loads targets from both approaches.

## What Needs to Be Configured

Each build profile contains paths that are **machine-specific** and must be updated:

### Windows MSVC Builds
```json
{
  "setup_scripts": [
    "C:\\Program Files\\Microsoft Visual Studio\\2022\\<Edition>\\VC\\Auxiliary\\Build\\vcvars64.bat"
  ],
  "environment": {
    "Qt6_DIR": "C:\\Qt\\<version>\\msvc2022_64"
  }
}
```

**Update**:
- Path to your Visual Studio edition (Community/Professional/Enterprise)
- Path to your Qt MSVC installation

### Windows MSVC CUDA Builds
Additionally requires:
```json
{
  "environment": {
    "CUDA_PATH": "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x"
  },
  "path": [
    "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin"
  ]
}
```

**Update**:
- Path to your CUDA installation (match version number)

### Windows MinGW Builds
```json
{
  "environment": {
    "Qt6_DIR": "C:\\Qt\\<version>\\mingw_64"
  },
  "path": [
    "C:\\Qt\\<version>\\mingw_64\\bin"
  ]
}
```

**Update**:
- Path to your Qt MinGW installation

### Linux Builds
```json
{
  "environment": {
    "Qt6_DIR": "/opt/qt6"
  }
}
```

**Update**:
- Path to your Qt6 installation (system packages or custom)
- For GPU builds: Add `CUDA_HOME` environment variable

### macOS Builds
```json
{
  "environment": {
    "Qt6_DIR": "/opt/homebrew/opt/qt"
  }
}
```

**Update**:
- Path to Homebrew Qt (different for Intel vs ARM Macs)

## Important Notes

### CMake Path Issues (Windows)

Some machines have multiple CMake installations:
- **Visual Studio CMake**: `C:\Program Files\CMake\bin\cmake` (recommended for MSVC)
- **Qt CMake**: `C:\Qt6\Tools\CMake_64\bin` (alternative)

The `vcvars64.bat` script sets the correct CMake in PATH. If you encounter CMake errors:
1. Check which CMake is first in PATH: `where cmake`
2. The Visual Studio CMake should be preferred for MSVC builds
3. For MinGW builds, ensure MinGW's `mingw32-make` is accessible

### Profile Validation

After editing profiles, verify they're detected:
```bash
python scripts/build.py --list-targets
```

You should see your platform's targets listed with their source files.

### Path Format

**Windows**: Use double backslashes `\\` in JSON:
```json
"Qt6_DIR": "C:\\Qt\\6.9.3\\msvc2022_64"
```

**Linux/macOS**: Use forward slashes:
```json
"Qt6_DIR": "/opt/qt6"
```

## Troubleshooting

### "No build targets found" Error
- No `build_profiles_*.json` files exist, or they're invalid JSON
- Check for syntax errors in your JSON files
- Verify files have the naming pattern: `build_profiles_<target>.json`

### "Qt6 not found" Error
- Verify `Qt6_DIR` path is correct in your profile
- Check that Qt6 is actually installed at that location
- On Linux: May need to install Qt6 development packages

### "vcvars64.bat not found" (Windows)
- Verify Visual Studio installation path in `setup_scripts`
- Check your Visual Studio edition: Community/Professional/Enterprise
- Ensure "Desktop development with C++" workload is installed

### "CUDA not detected" (GPU builds)
- Verify CUDA installation path in `CUDA_PATH`
- Run `nvidia-smi` and `nvcc --version` to verify CUDA is working
- Ensure CUDA bin directory is in `path` array

## See Also

- **LOCAL_SETTINGS.md**: Machine-specific settings reference (in repository root)
- **docs/BUILD_SYSTEM_OVERVIEW.md**: Complete build system documentation
- **scripts/build.py**: The unified build script

## Version Control

**Do NOT commit**:
- `build_profiles.json` (if you create it from the example)
- Any files with your personal/machine-specific paths

These files are already in `.gitignore`.

**Do commit**:
- The template files (`build_profiles_*.json` with placeholder/example paths)
- This README
- Any improvements to the build profile structure
