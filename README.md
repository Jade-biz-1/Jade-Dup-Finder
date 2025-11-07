<div align="center">
  <img src="docs/logo.png" alt="DupFinder Logo" width="200"/>
</div>

# DupFinder - Cross-Platform Duplicate File Finder

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](https://github.com/yourusername/dupfinder)
[![Qt](https://img.shields.io/badge/Qt-6.0%2B-green)](https://www.qt.io/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)](https://isocpp.org/)

DupFinder is a modern, cross-platform desktop application designed to help users identify and manage duplicate files on their systems. With an intuitive interface, sophisticated duplicate detection algorithms, and comprehensive safety features, DupFinder helps you reclaim disk space while protecting your data.

## 🚧 **Project Status: Work in Progress**

**DupFinder is actively under development!** While the core functionality is working and the application is usable, we're continuously improving and expanding features. This is an open-source project and **we welcome volunteers and contributors** of all skill levels.

### 🤝 **Join Our Development Team!**

We're looking for passionate developers, testers, designers, and documentation writers to help make DupFinder even better. Whether you're a seasoned developer or just starting out, there are ways to contribute:

- **Developers**: Help implement new features, fix bugs, or improve performance
- **Testers**: Help us identify issues and improve reliability across platforms  
- **UI/UX Designers**: Enhance the user experience and interface design
- **Technical Writers**: Improve documentation and user guides
- **Translators**: Help make DupFinder available in more languages

**Ready to contribute?** Get in touch with the project maintainer or check out our [Contributing Guidelines](#-contributing) below.

### 📧 **Contact the Author**

Interested in contributing or have questions about the project? 

- **GitHub Issues**: [Report bugs or request features](https://github.com/Jade-biz-1/Jade-Dup-Finder/issues)
- **GitHub Discussions**: [Join community discussions](https://github.com/Jade-biz-1/Jade-Dup-Finder/discussions)
- **Direct Contact**: Open an issue with the "question" label to get in touch with the maintainer

We believe in building great software together! 🚀

## 🎯 Key Features

### ⚡ **NEW: High-Performance Optimizations (November 2025)**
- **Handles Large File Sets**: Efficiently processes 378,000+ files without hanging or becoming unresponsive
- **Dramatic Speed Improvements**: File scanning reduced from 30+ minutes to 2-5 minutes
- **Smart Resource Management**: Single-instance hash calculator eliminates massive overhead
- **Optimized Batch Processing**: 100x improvement in duplicate detection throughput (5 → 500 files/batch)
- **Reduced UI Overhead**: 100x fewer cross-thread signals for better responsiveness
- **Command-Line Testing**: Non-UI test tool for performance validation and troubleshooting

### Core Functionality
- **Multi-Level Detection**: Quick scan, deep hash-based analysis, and specialized media detection
- **Smart Presets**: Predefined scan profiles for Downloads, Photos, Documents, and Full System scans
- **Comprehensive Safety**: Files moved to trash (never permanently deleted), undo capability, and session logging
- **Advanced Results Dashboard**: Professional 3-panel interface with smart selection and file operations
- **Visual Interface**: Modern Qt6-based UI with thumbnail previews and smart recommendations

### ✨ Advanced Features ✅ **IMPLEMENTED**
- **Three-Panel Results Interface**: Header, hierarchical results tree, and comprehensive actions panel
- **Smart Selection System**: Automatic recommendations for files to keep vs. delete based on file attributes
- **Bulk Operations**: Select and manage multiple files with detailed confirmation dialogs
- **Real-time Statistics**: Live updates of selection counts, space savings, and operation progress
- **File Operations Integration**: Copy paths, open locations, preview files, and system file management
- **Safety Confirmations**: Detailed impact summaries before any destructive operations

### Cross-Platform Support
- **Windows 10/11**: Native integration with Windows Explorer and Recycle Bin
- **macOS 10.15+**: Finder integration and system trash API support
- **Linux**: Desktop integration with system trash support (Ubuntu 20.04+, Fedora 35+)

### Business Model
- **Freemium**: Full features with scanning limitations (10,000 files or 100GB) for free users
- **Premium**: Unlimited scanning capacity with all advanced features

## 🚀 Quick Start

### Prerequisites
- Qt 6.0 or later
- CMake 3.20 or later  
- Modern C++17 compatible compiler:
  - Windows: Visual Studio 2019+ or MinGW-w64
  - macOS: Xcode 12+ (Clang)
  - Linux: GCC 9+ or Clang 10+

### Building from Source

#### Unified Build Orchestrator (Recommended)

DupFinder uses `scripts/build.py`, an intelligent build orchestrator that selects the
correct toolchain (CPU/GPU, OS, architecture) and copies finished installers
into the standardized `dist/` layout.

**Quick Start:**

1. **Configure your build environment:**
   - **Option A (Recommended):** Edit individual profile files for platforms you'll use:
     - Windows: `config/build_profiles_windows-msvc-cpu.json` or `config/build_profiles_windows-msvc-cuda.json`
     - Linux: `config/build_profiles_linux-cpu.json` or `config/build_profiles_linux-gpu.json`
     - macOS: `config/build_profiles_macos-x86_64.json` or `config/build_profiles_macos-arm64.json`
   - **Option B (Legacy):** Copy `config/build_profiles.example.json` to `config/build_profiles.json`
   - See [LOCAL_SETTINGS.md](LOCAL_SETTINGS.md) for reference configurations and examples

2. **List available build targets:**
   ```bash
   python scripts/build.py --list-targets
   ```

3. **Run interactive build:**
   ```bash
   python scripts/build.py
   ```
   The script detects your OS and GPU, shows available targets, and builds after confirmation.

4. **Find your artifacts:**
   - Windows: `dist/Win64/{Debug,Release}/`
   - Linux: `dist/Linux/{Debug,Release}/` (includes DEB, RPM, and TGZ packages)
   - macOS: `dist/MacOS/{X64,ARM}/{Debug,Release}/`

**For CI/CD and automation:**
```bash
python scripts/build.py --target <target-id> --build-type Release --non-interactive
```

**For comprehensive build system documentation, see [docs/BUILD_SYSTEM_OVERVIEW.md](docs/BUILD_SYSTEM_OVERVIEW.md)**

#### Clone the Repository
```bash
git clone https://github.com/yourusername/dupfinder.git
cd dupfinder
```

#### Manual Build (Alternative to build.py)

If you prefer manual CMake configuration instead of using the build orchestrator:

**Linux/Ubuntu Build:**
```bash
# Install dependencies
sudo apt update
sudo apt install build-essential cmake ninja-build qt6-base-dev qt6-tools-dev

# Configure with CMake
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --parallel

# Run
./build/dupfinder
```

**Windows Build:**
```cmd
# Using Visual Studio Developer Command Prompt
# Set Qt6_DIR environment variable first:
# set Qt6_DIR=C:\Qt6\6.9.3\msvc2022_64

cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

# Run
build\Release\dupfinder.exe
```

**macOS Build:**
```bash
# Install dependencies (using Homebrew)
brew install cmake ninja qt6

# Configure with CMake
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --parallel

# Run
open ./build/dupfinder.app
```

**Note:** Manual builds don't automatically copy artifacts to `dist/`. Use `scripts/build.py` for automated packaging and distribution.

## 📁 Project Structure

```
dupfinder/
├── src/                   # Source code
│   ├── core/              # Core duplicate detection algorithms
│   ├── gui/               # Qt6-based user interface
│   ├── gpu/               # GPU acceleration (CUDA/OpenCL)
│   ├── platform/          # Platform-specific implementations
│   │   ├── windows/       # Windows-specific code
│   │   ├── macos/         # macOS-specific code
│   │   └── linux/         # Linux-specific code
│   └── main.cpp           # Application entry point
├── include/               # Header files
├── tests/                 # Unit and integration tests
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── performance/       # Performance benchmarks
├── config/                # Build configuration
│   ├── build_profiles_*.json  # Per-target build profiles
│   └── build_profiles.example.json  # Example configuration
├── docs/                  # Documentation (see docs/README.md)
│   ├── BUILD_SYSTEM_OVERVIEW.md   # Complete build guide
│   ├── IMPLEMENTATION_TASKS.md    # Active task list
│   ├── PRD.md                     # Product requirements
│   ├── ARCHITECTURE_DESIGN.md     # System architecture
│   └── archive/           # Historical documentation
├── resources/             # Application resources
│   ├── icons/             # Application icons
│   └── translations/      # Internationalization files
├── scripts/               # Build and deployment scripts
│   └── build.py           # Unified build orchestrator
├── cmake/                 # CMake modules
├── build/                 # Build output (generated)
│   ├── windows/           # Windows builds
│   ├── linux/             # Linux builds
│   └── macos/             # macOS builds
└── dist/                  # Distribution packages (generated)
    ├── Win64/             # Windows installers
    ├── Linux/             # Linux packages (DEB/RPM/TGZ)
    └── MacOS/             # macOS disk images
```

## 📚 Documentation

**Complete documentation index available at [docs/README.md](docs/README.md)**

### Quick Links

#### For Developers
- **[BUILD_SYSTEM_OVERVIEW.md](docs/BUILD_SYSTEM_OVERVIEW.md)** - Comprehensive build system guide (start here!)
- **[LOCAL_SETTINGS.md](LOCAL_SETTINGS.md)** - Reference configurations for all platforms
- **[DEVELOPMENT_SETUP.md](docs/DEVELOPMENT_SETUP.md)** - Development environment setup
- **[ARCHITECTURE_DESIGN.md](docs/ARCHITECTURE_DESIGN.md)** - System architecture and design
- **[IMPLEMENTATION_TASKS.md](docs/IMPLEMENTATION_TASKS.md)** - Current development tasks

#### For Contributors
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md)** - Development processes
- **[IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)** - Project roadmap

#### Product Documentation
- **[PRD.md](docs/PRD.md)** - Product Requirements Document
- **[UI_DESIGN_SPECIFICATION.md](docs/UI_DESIGN_SPECIFICATION.md)** - User interface design
- **[MANUAL_TESTING_GUIDE.md](docs/MANUAL_TESTING_GUIDE.md)** - Testing procedures

### Documentation Organization

- **Active Documentation**: Current guides and references in `docs/`
- **Archived Documentation**: Historical documents in `docs/archive/`
- **Component Documentation**: Archived API docs in `docs/archive/`

## 🔧 Development

### Code Style
We use `clang-format` for consistent code formatting. Run the formatter with:
```bash
make format  # or ninja format
```

### Testing

✅ **Current Status**: Core test suite is stable and working!

```bash
# Build and run tests
cd build
make unit_tests
./tests/unit_tests

# Run all available tests
make check

# Core unit tests: 43/43 passing ✅
# Extended test framework available for advanced testing
```

We have a comprehensive testing infrastructure with:
- **Unit Tests**: Core functionality validation (100% stable)
- **Integration Tests**: Component interaction testing
- **Performance Tests**: Scalability and optimization validation  
- **Security Tests**: Safety and vulnerability assessment
- **UI Tests**: Automated interface testing
- **CI/CD Pipeline**: Automated testing on all platforms

**Want to help with testing?** This is a great area for new contributors! See our [testing documentation](docs/testing/) for guides and examples.

### Platform-Specific Development

#### Windows Development
- Use Visual Studio 2019+ with Qt VS Tools extension
- Ensure Qt6 is in your PATH or set Qt6_DIR
- For MSVC builds, install Visual C++ Redistributable

#### macOS Development  
- Use Xcode or Qt Creator
- Install Qt6 via Homebrew or Qt Installer
- Code signing required for distribution

#### Linux Development
- Install Qt6 development packages
- GCC 9+ or Clang 10+ recommended
- Install platform-specific packages for your distribution

## 📦 Distribution

### Creating Packages

**Using build.py (Recommended):**
```bash
python scripts/build.py --target <target-id> --build-type Release
```

Automatically creates platform-specific packages and copies them to `dist/`:

- **Windows**: NSIS installer (`.exe`)
  - `dist/Win64/Release/dupfinder-<version>-win64-<variant>.exe`
- **Linux**: Multiple package formats
  - `dist/Linux/Release/dupfinder-<version>-linux-x86_64-<variant>.deb` (Debian/Ubuntu)
  - `dist/Linux/Release/dupfinder-<version>-linux-x86_64-<variant>.rpm` (RedHat/Fedora)
  - `dist/Linux/Release/dupfinder-<version>-linux-x86_64-<variant>.tgz` (Universal)
- **macOS**: Disk image (`.dmg`)
  - `dist/MacOS/{X64,ARM}/Release/dupfinder-<version>-macos-<arch>.dmg`

**Manual packaging:**
```bash
cd build
cmake --build . --target package
```

### Build Variants

- **Windows**: `msvc-cpu`, `msvc-cuda` (GPU), `mingw-cpu`
- **Linux**: `cpu`, `gpu` (CUDA)
- **macOS**: `x86_64` (Intel), `arm64` (Apple Silicon)

### Continuous Integration
GitHub Actions can automatically build and test on all platforms. Configure with `scripts/build.py --non-interactive` for CI/CD pipelines.

## 🛡️ Safety Features

DupFinder prioritizes data safety:
- **No Permanent Deletion**: Files are always moved to system trash/recycle bin
- **Undo Operations**: Recent operations can be undone within the session
- **System File Protection**: Automatic exclusion of critical system files
- **Confirmation Dialogs**: Clear summaries before any file operations
- **Session Logging**: Complete log of all operations for recovery if needed

## 📊 Use Cases

### Primary Scenarios
1. **Disk Space Cleanup**: Free up storage space by removing duplicate files
2. **Download Folder Management**: Clean up accumulated duplicate downloads
3. **Photo Library Organization**: Consolidate duplicate photos from multiple devices

### Target Users
- **General Home Users**: Non-technical users wanting simple disk cleanup
- **Storage-Conscious Users**: People running low on disk space
- **Digital Asset Managers**: Users with large media collections

## 🏗️ Architecture

### Core Components
- **Duplicate Detection Engine**: Multi-algorithm approach (size+name, hash-based, media-specific)
- **File System Interface**: Cross-platform file operations with Qt6
- **Safety Manager**: Handles safe deletion and recovery operations
- **UI Controllers**: Modern Qt6 Widgets-based interface

### Threading Model
- Background scanning with progress updates
- Separate threads for I/O operations and hash computation
- Responsive UI during all operations

## 📈 Development Roadmap

### Phase 1: Foundation (Months 1-2) ✅ **COMPLETED + ADVANCED FEATURES**
- [x] ✅ Project structure and build system
- [x] ✅ Core duplicate detection algorithms (basic implementation)
- [x] ✅ **ENHANCED:** Advanced Qt6 interface with professional 3-panel Results Dashboard
- [x] ✅ Linux implementation with platform-specific integrations
- [x] ✅ **BONUS:** Advanced Results Window with smart selection and file operations
- [x] ✅ **BONUS:** Comprehensive file management with safety confirmations
- [x] ✅ **BONUS:** Real-time statistics and progress tracking
- [x] ✅ **BONUS:** Professional UI with filtering, sorting, and bulk operations

### Phase 2: Feature Complete (Months 3-4) ✅ **COMPLETED WITH ENHANCEMENTS**
- [x] ✅ Multi-level detection algorithms (implemented with advanced features)
- [x] ✅ **COMPLETED AHEAD OF SCHEDULE:** Comprehensive dashboard interface
- [x] ✅ **COMPLETED AHEAD OF SCHEDULE:** Safety and recovery features (full implementation)
- [x] ✅ **BONUS:** Performance optimizations with advanced threading and caching
- [x] ✅ **BONUS:** Comprehensive test suite with 200+ test files and CI/CD automation
- [x] ✅ **BONUS:** Enterprise-grade testing infrastructure and quality controls

### Phase 3: Cross-Platform (Months 5-7) ⚡ **IN PROGRESS**
- [x] ✅ macOS port and native integration completed
- [x] ✅ macOS platform files implemented (trash_manager.mm, platform_file_ops, system_integration)
- [x] ✅ macOS DMG installer configured and tested
- [x] ✅ **PERFORMANCE OPTIMIZATIONS:** Critical fixes for large file sets (378K+ files)
  - Fixed HashCalculator recreation bug (was creating 378K instances!)
  - Optimized batch processing (5 → 500 files per batch)
  - Enhanced file scanner performance (30+ min → 2-5 min)
  - Removed artificial delays and reduced cross-thread communication overhead
- [x] ✅ Command-line test tool for non-UI performance testing
- [ ] Windows port and native integration (platform files present, needs build testing)
- [ ] Platform-specific installers for Windows
- [ ] Cross-platform testing on all platforms

### Phase 4: Premium Features (Months 8-10)
- [ ] Freemium model implementation
- [ ] Payment processing integration
- [ ] Advanced features and polish
- [ ] Beta testing program

### Phase 5: Launch (Months 11-12)
- [ ] Public release
- [ ] Marketing and distribution
- [ ] User support systems
- [ ] Post-launch monitoring

## 🤝 Contributing

**We actively welcome contributions from developers of all experience levels!** DupFinder is a community-driven project and we believe great software is built together.

### 🌟 **Ways to Contribute**

#### For Developers
- **New Features**: Implement items from our roadmap or propose new functionality
- **Bug Fixes**: Help resolve issues and improve stability
- **Performance**: Optimize algorithms and improve efficiency
- **Cross-Platform**: Help with Windows and macOS platform support
- **Code Quality**: Refactoring, documentation, and best practices

#### For Non-Developers  
- **Testing**: Manual testing on different platforms and configurations
- **Documentation**: Improve user guides, API docs, and tutorials
- **UI/UX**: Design improvements and user experience enhancements
- **Translation**: Help make DupFinder available in more languages
- **Community**: Help other users, answer questions, provide feedback

### 🚀 **Getting Started as a Contributor**

#### Quick Start for New Contributors
1. **Browse Issues**: Look for issues labeled `good first issue` or `help wanted`
2. **Join Discussions**: Participate in [GitHub Discussions](https://github.com/Jade-biz-1/Jade-Dup-Finder/discussions)
3. **Ask Questions**: Don't hesitate to ask for guidance - we're here to help!
4. **Start Small**: Begin with documentation, testing, or small bug fixes

#### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`make check`)
5. Format your code (`make format`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### 📋 **Current Contribution Opportunities**

#### High Priority
- **Windows Platform Support**: Help implement Windows-specific features
- **macOS Platform Support**: Native macOS integration and bundle creation
- **Performance Optimization**: Large file handling and memory efficiency
- **Advanced UI Features**: Enhanced file preview and management tools

#### Good for Beginners
- **Documentation**: Improve README, add code comments, create tutorials
- **Testing**: Write additional test cases, manual testing on different systems
- **Bug Reports**: Identify and report issues with detailed reproduction steps
- **Code Cleanup**: Refactoring, formatting, and code organization

### 💬 **Get in Touch**

- **Questions**: Open an issue with the "question" label
- **Ideas**: Start a discussion in [GitHub Discussions](https://github.com/Jade-biz-1/Jade-Dup-Finder/discussions)
- **Collaboration**: Reach out if you want to work on larger features together

**We're excited to work with you!** Every contribution, no matter how small, helps make DupFinder better for everyone. 🎉

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Community

### Getting Help
- **Documentation**: [docs/](docs/) - Comprehensive guides and API documentation
- **Issues**: [GitHub Issues](https://github.com/Jade-biz-1/Jade-Dup-Finder/issues) - Bug reports and feature requests
- **Discussions**: [GitHub Discussions](https://github.com/Jade-biz-1/Jade-Dup-Finder/discussions) - Community Q&A and ideas

### Contributing & Volunteering
- **New Contributors Welcome**: We provide mentorship and guidance for first-time contributors
- **All Skill Levels**: From beginners to experts, there's a place for everyone
- **Flexible Commitment**: Contribute as much or as little as your schedule allows
- **Learning Opportunity**: Great project to learn Qt6, C++17, and modern software development practices

**Ready to join our community?** We'd love to have you aboard! 🚀

## ⭐ Acknowledgments

- Qt6 framework for cross-platform GUI capabilities
- Modern C++ community for best practices
- Open source contributors and testers

---

**DupFinder** - Helping you reclaim your disk space, one duplicate at a time! 🚀