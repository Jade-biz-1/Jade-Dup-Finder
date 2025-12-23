# CloneClean Development Bootstrap Guide

## Overview
**CloneClean** - One File. One Place. An intelligent duplicate file finder and cleaner.

This document provides essential information to get you started with CloneClean development. It covers project structure, constraints, best practices, and key resources.

## Project Navigation

### Core Documentation
- **[PRD (Product Requirements Document)](docs/PRD.md)** - Complete feature specifications and requirements
- **[Architecture Design](docs/ARCHITECTURE_DESIGN.md)** - System architecture and design decisions  
- **[Implementation Tasks](docs/IMPLEMENTATION_TASKS.md)** - Current development tasks and status
- **[Build System Overview](docs/BUILD_SYSTEM_OVERVIEW.md)** - Comprehensive build system documentation
- **[Development Workflow](docs/DEVELOPMENT_WORKFLOW.md)** - Process and practices for contributing
- **[UI Design Specification](docs/UI_DESIGN_SPECIFICATION.md)** - Complete user interface design

### Current Status Tracking
- **[Implementation Tasks](docs/IMPLEMENTATION_TASKS.md)** - Active development tasks and status
- **[Pending Tasks](PendingForNov27.md)** - Recently identified pending work
- **[Dec23.md](Dec23.md)** - Immediate tasks identified on December 23, 2025

## Project Structure
```
cloneclean/
├── src/                   # Source code
│   ├── core/              # Core duplicate detection algorithms
│   ├── gui/               # Qt6-based user interface
│   ├── gpu/               # GPU acceleration (CUDA/OpenCL)
│   ├── platform/          # Platform-specific implementations
│   └── main.cpp           # Application entry point
├── include/               # Header files
├── tests/                 # Unit and integration tests (place all tests here only)
├── config/                # Build configuration files (individual per-target JSON files)
├── docs/                  # Documentation
├── resources/             # Application resources
├── scripts/               # Build and deployment scripts
├── build/                 # Build output (generated)
└── dist/                  # Distribution packages (generated)
```

## Technology Stack

### Core Technologies
- **C++17** - Primary implementation language
- **Qt6** - Cross-platform GUI framework (Widgets, Concurrent, Network)
- **CMake 3.20+** - Build system
- **Ninja/Make** - Build execution

### Build Configuration
- **Config folder** - Holds various build configuration files (individual per-target JSON files in `config/build_profiles_*.json`)
- **Build system** - Must strictly follow the build system defined in [docs/BUILD_SYSTEM_OVERVIEW.md](docs/BUILD_SYSTEM_OVERVIEW.md)

### GPU Acceleration
- **CUDA** - NVIDIA GPU support (optional)
- **OpenCL** - Alternative GPU support (AMD/Intel)

### Platforms Supported
- **Windows 10/11** - MSVC/MinGW with Qt6
- **macOS 10.15+** - Xcode/Clang with Qt6
- **Linux** - GCC/Clang with Qt6

### Testing Guidelines
- **Test location** - Create tests only in the `tests/` folder and not elsewhere
- **No分散 test files** - All testing code must remain in the designated tests directory

## Development Constraints & Guidelines

### ⚠️ Important Constraints
- **C++17 Standard**: All code must use C++17 features only
- **Qt6 Dependency**: All GUI and cross-platform functionality must use Qt6
- **Security First**: All file operations must use safe deletion (trash/recycle bin only)
- **Memory Safety**: All GPU acceleration must have CPU fallback
- **Performance**: UI must remain responsive during file operations

### 🚫 Development Don'ts
- ❌ **Never delete files permanently** - Always move to system trash
- ❌ **Don't hardcode paths** - Use QStandardPaths and configuration files
- ❌ **Don't ignore compiler warnings** - All warnings must be resolved
- ❌ **Don't commit sensitive information** - API keys, passwords, personal data
- ❌ **Don't break existing functionality** - Maintain backward compatibility
- ❌ **Don't use non-portable code** - Must work on all supported platforms
- ❌ **Don't commit build artifacts** - Keep /build and /dist ignored
- ❌ **Don't create unnecessary summary documents** unless authorized
- ❌ **Don't guess or assume** when details are unclear - Ask for information instead

### ✅ Development Do's
- ✅ **Use Qt6 APIs** for all cross-platform functionality
- ✅ **Implement proper error handling** - Never crash on invalid data
- ✅ **Use RAII and smart pointers** for memory management
- ✅ **Write unit tests** for all new functionality
- ✅ **Follow existing code style** - Use clang-format
- ✅ **Document public APIs** with doxygen comments
- ✅ **Use proper logging** - Use provided logger facility
- ✅ **Test on multiple platforms** - Verify cross-platform compatibility
- ✅ **Update documentation** when making changes
- ✅ **Strictly follow the build system** as defined in [docs/BUILD_SYSTEM_OVERVIEW.md](docs/BUILD_SYSTEM_OVERVIEW.md)
- ✅ **Create tests only in the tests folder** and nowhere else
- ✅ **If you don't find specific details**, ask for information rather than assuming

## Build System Overview

### Modern Profile-Based Build System
CloneClean uses a sophisticated profile-based build system with automatic platform detection:

- **[Build System Documentation](docs/BUILD_SYSTEM_OVERVIEW.md)** - Complete guide
- **`scripts/build.py`** - Main build orchestrator
- **Per-target configuration files** in `config/build_profiles_*.json`

### Quick Build Commands
```bash
# Interactive build (recommended)
python scripts/build.py

# List all available targets
python scripts/build.py --list-targets

# Specific target build
python scripts/build.py --target <target-id> --build-type Release

# Non-interactive (for CI/CD)
python scripts/build.py --target linux-ninja-cpu --build-type Release --non-interactive
```

### Supported Build Targets
- **Linux CPU**: `linux-ninja-cpu` (produces DEB/RPM/TGZ packages)
- **Linux GPU**: `linux-ninja-gpu` (CUDA acceleration enabled)
- **Windows MSVC CPU**: `windows-msvc-cpu` (NSIS installer)
- **Windows MSVC GPU**: `windows-msvc-cuda` (CUDA acceleration)
- **Windows MinGW CPU**: `windows-mingw-cpu` (MinGW build)
- **macOS x64**: `macos-ninja-x86_64` (DMG installer)
- **macOS ARM64**: `macos-ninja-arm64` (DMG installer)

## Development Workflow

### 1. Task Planning & Tracking
- Refer to **[IMPLEMENTATION_TASKS.md](docs/IMPLEMENTATION_TASKS.md)** for current priorities
- Check **[PendingForNov27.md](PendingForNov27.md)** for recent pending tasks
- Update status in appropriate documentation files

### 2. Implementation Process
1. **Fork & Branch**: Create feature branch from development
2. **Code**: Follow existing style and patterns
3. **Test**: Run unit tests with `make check`
4. **Format**: Run `make format` or `ninja format`
5. **Document**: Update relevant documentation
6. **PR**: Submit pull request for review

### 3. Testing Strategy
- **Unit Tests**: Core functionality validation
- **Integration Tests**: Component interaction
- **Performance Tests**: Scalability validation  
- **UI Tests**: Automated interface testing
- **Cross-Platform Tests**: Platform-specific validation

### 4. Status Updates
- Update **[IMPLEMENTATION_TASKS.md](docs/IMPLEMENTATION_TASKS.md)** with progress
- Mark tasks as **✅ COMPLETE**, **🔄 IN PROGRESS**, or **⏸️ PENDING**
- Update phase status in documentation
- Keep **[Dec23.md](Dec23.md)** and other status documents current

## Key Resources

### Getting Help
- **[Development Setup](docs/DEVELOPMENT_SETUP.md)** - Environment setup guide
- **[Troubleshooting](docs/BUILD_SYSTEM_OVERVIEW.md#troubleshooting)** - Build issues
- **[Manual Testing Guide](docs/MANUAL_TESTING_GUIDE.md)** - Quality assurance

### Task Tracking
- **[TasksTracking/README.md](TasksTracking/README.md)** - New task tracking system (primary)
- **[Current Sprint](TasksTracking/current-sprint.md)** - Active development tasks
- **[Core Features](TasksTracking/core-features.md)** - Core functionality tasks
- **[Advanced Detection](TasksTracking/advanced-detection.md)** - Algorithm tasks
- **[File Type Enhancements](TasksTracking/file-type-enhancements.md)** - File handling tasks
- **[UI/UX Features](TasksTracking/ui-ux-features.md)** - Interface tasks
- **[Platform Integration](TasksTracking/platform-integration.md)** - Cross-platform tasks
- **[Performance](TasksTracking/performance.md)** - Optimization tasks
- **[Safety & Reliability](TasksTracking/safety-reliability.md)** - Safety tasks
- **[Testing & Quality](TasksTracking/testing-quality.md)** - Quality assurance tasks

### Quality Assurance
- **[Manual Testing Guide](docs/MANUAL_TESTING_GUIDE.md)** - Comprehensive test procedures
- **[Security Testing](tests/SECURITY_SAFETY_TESTING_README.md)** - Safety validation
- **[Performance Monitoring](tests/PERFORMANCE_MONITORING_README.md)** - Performance tracking
- **[Test Status Report](TasksTracking/test-status.md)** - Current testing status and execution results

### Historical Documentation
- **[docs/archive/](docs/archive/)** - Archived documents and historical information

## Current Development Status

### Phase 3: Cross-Platform Port & Branding (Current)
- **Status**: 85% Complete
- **Focus**: CloneClean rebranding and cross-platform validation
- **Completion**: Awaiting Windows/macOS testing

### Next Phase: Premium Features (Planning)
- Freemium model implementation
- Advanced features and polish
- Beta testing program

## Quick Start Commands

### Environment Setup
```bash
# Verify prerequisites
cmake --version
qmake --version  # or qmake6
ninja --version  # or make

# Install dependencies (Ubuntu/Debian example)
sudo apt update && sudo apt install build-essential cmake ninja-build qt6-base-dev qt6-tools-dev
```

### Build & Run
```bash
# Using unified build system (recommended)
python scripts/build.py

# Or manual build (for development)
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cloneclean
./build/cloneclean
```

### Running the Application
```bash
# To run the application after building, use the provided launch script
./launch.sh

# Note: Only use launch.sh to run the application - do not create other launch scripts
```

### Development Tasks
```bash
# Run all tests (the primary way to run all tests in the project)
make check
# or (if using ninja)
ninja check
# or (using CTest directly)
ctest --output-on-failure

# Format code
make format

# Static analysis
make cppcheck

# Generate documentation
make docs
```

---

**Last Updated**: December 23, 2025  
**Next Review**: As development status or architecture changes  
**Maintained By**: CloneClean Development Team