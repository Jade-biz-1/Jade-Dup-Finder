# DupFinder - Windows Build Guide

**Date:** November 2, 2025  
**Status:** Build tested with Ninja generator; installer creation requires NSIS path configuration ✅  
**Prerequisites:** Core source files now available ✅

---

## 🎯 Windows Build Status

### ✅ Prerequisites Met
- **Core Source Files:** All 25+ missing .cpp files now available in `src/core/`
- **CMake Configuration:** Windows-specific settings present in CMakeLists.txt
- **Qt6 Integration:** Windows linking configured (shell32, ole32)

### 🔧 Required Windows Development Environment

#### Option 1: Visual Studio 2022 + MSVC (Recommended)

##### 1. Install Visual Studio 2022
```powershell
# Download and install Visual Studio 2022 Community (free)
# Include workloads:
# - Desktop development with C++
# - Windows 10/11 SDK (latest)
```

##### 2. Install Qt6 for Windows
```powershell
# Download Qt Online Installer: https://www.qt.io/download-qt-installer
# Install Qt 6.6.x LTS with MSVC 2022 64-bit compiler
# Default installation path: C:\Qt\6.6.x\msvc2022_64\
```

##### 3. Install CMake
```powershell
# Download from: https://cmake.org/download/
# Or use winget: winget install Kitware.CMake
# Add to PATH during installation
```

##### 4. Install Git
```powershell
# Download from: https://git-scm.com/download/win
# Or use winget: winget install Git.Git
```

#### Option 2: MinGW-w64 + Qt Creator

##### 1. Install Qt Creator with MinGW
```powershell
# Download Qt Online Installer
# Install Qt 6.6.x with MinGW 11.2.0 64-bit
# Includes Qt Creator IDE
```

##### 2. Install CMake
```powershell
# Same as above
```

---

## 🏗️ Windows Build Instructions

### Step 1: Clone and Setup
```powershell
# Clone repository
git clone https://github.com/Jade-biz-1/Jade-Dup-Finder.git
cd Jade-Dup-Finder

# Create build directory
mkdir build
cd build
```

### Step 2: Configure with CMake (MSVC)
```powershell
# Using Visual Studio Developer Command Prompt
cmake .. -G "Visual Studio 17 2022" -A x64

# Alternative: Using PowerShell with VS environment
cmake .. -G "Visual Studio 17 2022" -A x64
```

### Step 3: Build Application
```powershell
# Build Release configuration
cmake --build . --config Release --parallel

# Alternative: Open DupFinder.sln in Visual Studio and build
```

### Step 4: Run Application
```powershell
# From build directory
.\Release\dupfinder.exe
```

---

## 🔧 CMakeLists.txt Windows Analysis

### Windows-Specific Configuration
```cmake
# Platform detection
if(WIN32)
    set(PLATFORM_NAME "windows")
elseif(APPLE)
    set(PLATFORM_NAME "macos")
elseif(UNIX)
    set(PLATFORM_NAME "linux")
endif()

# Windows-specific linking
if(WIN32)
    target_link_libraries(dupfinder shell32 ole32)
    set_target_properties(dupfinder PROPERTIES
        WIN32_EXECUTABLE TRUE  # No console window
    )
endif()
```

### Platform Source Files (Currently Missing)
```cmake
# These are commented out - need implementation
# set(PLATFORM_SOURCES
#     src/platform/${PLATFORM_NAME}/platform_file_ops.cpp
#     src/platform/${PLATFORM_NAME}/trash_manager.cpp
#     src/platform/${PLATFORM_NAME}/system_integration.cpp
# )
```

### Windows Packaging
```cmake
# Windows installer generation
if(WIN32)
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "DupFinder")
    set(CPACK_PACKAGE_NAME "DupFinder")
endif()
```

---

## 📦 Windows-Specific Implementation Requirements

### Required Platform Files
Create these files for full Windows integration:

#### 1. `src/platform/windows/platform_file_ops.cpp`
```cpp
#include "platform_file_ops.h"
#include <windows.h>
#include <shlwapi.h>

// Windows-specific file operations
// - Long path support (>260 chars)
// - NTFS permissions handling
// - Windows file attributes
```

#### 2. `src/platform/windows/trash_manager.cpp`
```cpp
#include "trash_manager.h"
#include <windows.h>
#include <shellapi.h>

// Windows Recycle Bin integration
// - Move files to Recycle Bin (not permanent delete)
// - Restore from Recycle Bin
// - Empty Recycle Bin operations
```

#### 3. `src/platform/windows/system_integration.cpp`
```cpp
#include "system_integration.h"
#include <windows.h>
#include <shlobj.h>

// Windows system integration
// - Windows Explorer context menus
// - File associations
// - Desktop notifications
// - Start menu integration
```

### Windows-Specific Features Needed
1. **Recycle Bin API Integration**
2. **Windows Explorer Context Menus**
3. **File Association Registration**
4. **Windows Toast Notifications**
5. **UWP/App Store Packaging Option**

---

## 🧪 Testing Windows Build

### Automated Testing Setup
```powershell
# Run tests after build
cd build
ctest --output-on-failure

# Run specific test categories
ctest -L unit
ctest -L integration
```

### Manual Testing Checklist
- [ ] Application launches without console window
- [ ] Qt6 GUI renders correctly
- [ ] File scanning works on Windows paths
- [ ] Basic duplicate detection functions
- [ ] Settings persistence works
- [ ] Window state management works

---

## 📋 Windows Development Workflow

### Daily Development Cycle
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

### Release Build Process
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

### Resuming an Existing Build
If the build directory already exists and is configured (e.g., with Ninja generator):
```powershell
cd build

# Build or rebuild Release configuration
cmake --build . --config Release

# If needed, reconfigure (e.g., after CMakeLists.txt changes)
cmake ..

# Create installer
cpack -C Release
```

**Note:** If using Ninja generator (default in some setups), ensure NSIS is in PATH or set CPACK_NSIS_EXECUTABLE in CPackConfig.cmake:
```cmake
set(CPACK_NSIS_EXECUTABLE "C:/Program Files (x86)/NSIS/makensis.exe")
```

---

## 🚀 Windows CI/CD Setup

### GitHub Actions Windows Workflow
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

---

## 🐛 Common Windows Build Issues

### Qt6 Not Found
```
CMake Error: Could not find a package configuration file
```
**Solution:**
```powershell
# Set Qt6_DIR environment variable
$env:Qt6_DIR = "C:\Qt\6.6.x\msvc2022_64\lib\cmake\Qt6"
cmake .. -G "Visual Studio 17 2022" -A x64
```

### MSVC Compiler Issues
```
fatal error C1083: Cannot open include file: 'windows.h'
```
**Solution:** Run from "Developer Command Prompt for VS 2022"

### DLL Dependencies at Runtime
**Solution:** Copy Qt6 DLLs to executable directory or use windeployqt:
```powershell
# Use windeployqt to copy dependencies
& "C:\Qt\6.6.x\msvc2022_64\bin\windeployqt.exe" .\Release\dupfinder.exe
```

### NSIS Installer Creation Fails
```
CPack Error: Cannot find NSIS
```
**Solution:**
```powershell
# Add NSIS to PATH
$env:PATH += ";C:\Program Files (x86)\NSIS"

# Or set in CPackConfig.cmake
set(CPACK_NSIS_EXECUTABLE "C:/Program Files (x86)/NSIS/makensis.exe")

# Then run
cpack -C Release
```

---

## 📊 Windows Performance Considerations

### Build Performance
- **MSVC:** Faster compilation than MinGW
- **Parallel Builds:** Use `--parallel` flag
- **Incremental Builds:** CMake detects changes automatically

### Runtime Performance
- **NTFS Optimization:** Use Windows file APIs for large drives
- **Memory Management:** Windows memory limits vs Linux
- **Threading:** Windows thread pool vs Qt threading

---

## 🎯 Next Steps for Windows Development

### Immediate Tasks (This Week)
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

### Medium-term Goals (1-2 Months)
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

## ❓ Windows Development Questions

1. **Qt6 Version:** Use 6.6.x LTS or latest 6.7.x?
2. **Target Windows Versions:** Windows 10 only, or 10+11?
3. **Installer Type:** NSIS only, or also MSI/Windows Store?
4. **Code Signing:** Required for distribution?
5. **Windows Store:** Target UWP packaging?

---

**Prepared By:** AI Assistant (Grok)  
**Date:** November 2, 2025  
**Status:** Ready for Windows development implementation</content>
<parameter name="filePath">c:\Public\Jade-Dup-Finder\docs\WINDOWS_BUILD_GUIDE.md