# DupFinder Documentation Index

**Last Updated:** November 4, 2025

Welcome to the DupFinder documentation! This index helps you find the right documentation for your needs.

---

## 🚀 Quick Start

### New to DupFinder?

1. **[../README.md](../README.md)** - Start here! Project overview and quick start
2. **[BUILD_SYSTEM_OVERVIEW.md](BUILD_SYSTEM_OVERVIEW.md)** - Complete build system guide
3. **[LOCAL_SETTINGS.md](../LOCAL_SETTINGS.md)** - Reference configurations for your platform
4. **[DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md)** - Set up your development environment

### Want to Contribute?

1. **[../CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines
2. **[IMPLEMENTATION_TASKS.md](IMPLEMENTATION_TASKS.md)** - Current active tasks
3. **[DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md)** - Development processes

---

## 📋 Documentation Categories

### 🔧 Build & Development

| Document | Purpose | Audience |
|----------|---------|----------|
| **[BUILD_SYSTEM_OVERVIEW.md](BUILD_SYSTEM_OVERVIEW.md)** | Comprehensive build system guide covering all platforms, toolchains, and packaging | Developers, CI/CD engineers |
| **[LOCAL_SETTINGS.md](../LOCAL_SETTINGS.md)** | Reference configurations with example paths for Qt, CUDA, Visual Studio | All developers |
| **[DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md)** | Environment setup for Windows, Linux, and macOS | New contributors |
| **[DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md)** | Development processes, git workflow, and best practices | Contributors |

### 📐 Architecture & Design

| Document | Purpose | Audience |
|----------|---------|----------|
| **[ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md)** | System architecture, component design, and data flow | Developers, architects |
| **[ARCHITECTURAL_DECISIONS.md](ARCHITECTURAL_DECISIONS.md)** | Key architectural decisions and rationale | Technical leads |
| **[UI_DESIGN_SPECIFICATION.md](UI_DESIGN_SPECIFICATION.md)** | User interface design, layouts, and interaction patterns | UI developers, designers |
| **[PRD.md](PRD.md)** | Product Requirements Document - features and requirements | Product managers, developers |

### 📝 Planning & Tasks

| Document | Purpose | Audience |
|----------|---------|----------|
| **[IMPLEMENTATION_TASKS.md](IMPLEMENTATION_TASKS.md)** | **Current active task list** - what's being worked on now | All team members |
| **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** | Project roadmap and phase planning | Project managers |
| **[REMAINING_TASKS.md](REMAINING_TASKS.md)** | Upcoming tasks and future work | Planning team |

### 🧪 Testing & Quality

| Document | Purpose | Audience |
|----------|---------|----------|
| **[MANUAL_TESTING_GUIDE.md](MANUAL_TESTING_GUIDE.md)** | Manual testing procedures and test cases | QA testers |

### 🔍 Analysis & Maintenance

| Document | Purpose | Audience |
|----------|---------|----------|
| **[CLEANUP_ANALYSIS.md](CLEANUP_ANALYSIS.md)** | Code cleanup and technical debt analysis | Maintainers |
| **[UNDERSTANDING_GROK.md](UNDERSTANDING_GROK.md)** | Project understanding and onboarding guide | New developers |

---

## 📚 Archived Documentation

Historical and superseded documentation is organized in subdirectories:

### archive/

**Completion Summaries** (October 2025):
- `completion-summaries-2025-10/` - Phase completion reports and progress summaries

**Development Fixes** (October 2025):
- `development-fixes-2025-10/` - Bug fixes, issue resolutions, and fix summaries

**Component Documentation** (Archived):
- `ADVANCED_FEATURES.md` - Advanced features implementation details
- `API_FILESCANNER.md` - FileScanner API reference (superseded)
- `API_RESULTSWINDOW.md` - ResultsWindow API reference (superseded)

**Historical Documents**:
- Code reviews, fix summaries, task tracking from previous phases

> **Note:** Archived documents are kept for historical reference but may not reflect the current state of the project. Always refer to active documentation in the main `docs/` directory for current information.

---

## 🎯 Documentation by Role

### For New Developers

**Getting Started (in order):**
1. [../README.md](../README.md) - Project overview
2. [DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md) - Set up your environment
3. [BUILD_SYSTEM_OVERVIEW.md](BUILD_SYSTEM_OVERVIEW.md) - Build the project
4. [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md) - Understand the architecture
5. [IMPLEMENTATION_TASKS.md](IMPLEMENTATION_TASKS.md) - Find tasks to work on

### For Contributors

**Essential Reading:**
- [../CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute
- [IMPLEMENTATION_TASKS.md](IMPLEMENTATION_TASKS.md) - Active tasks
- [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md) - Development process
- [MANUAL_TESTING_GUIDE.md](MANUAL_TESTING_GUIDE.md) - Testing procedures

### For Build Engineers

**Build System Documentation:**
- [BUILD_SYSTEM_OVERVIEW.md](BUILD_SYSTEM_OVERVIEW.md) - Complete build guide (50+ pages)
- [LOCAL_SETTINGS.md](../LOCAL_SETTINGS.md) - Configuration examples
- Section 1: Architecture overview and dist folder structure
- Section 2: Machine preparation and toolchain setup
- Section 3: Using build.py orchestrator
- Section 4: Platform support matrix
- Section 7: Packaging and distribution

### For Product Managers

**Product Documentation:**
- [PRD.md](PRD.md) - Product requirements and features
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Project roadmap
- [IMPLEMENTATION_TASKS.md](IMPLEMENTATION_TASKS.md) - Current progress

### For QA / Testers

**Testing Resources:**
- [MANUAL_TESTING_GUIDE.md](MANUAL_TESTING_GUIDE.md) - Test procedures
- Performance benchmarks framework in `tests/performance_benchmark.cpp`

---

## 📊 Document Status Legend

| Status | Meaning |
|--------|---------|
| ✅ **Current** | Up-to-date, actively maintained |
| 🔄 **In Progress** | Being updated or revised |
| 📦 **Archived** | Historical, kept for reference |
| ⏸️ **Pending** | Planned but not yet created |

---

## 🔗 External References

### Build Configuration Files

Located in `../config/`:
- `build_profiles_windows-msvc-cpu.json` - Windows MSVC CPU build
- `build_profiles_windows-msvc-cuda.json` - Windows MSVC GPU build
- `build_profiles_windows-mingw-cpu.json` - Windows MinGW CPU build
- `build_profiles_linux-cpu.json` - Linux CPU build
- `build_profiles_linux-gpu.json` - Linux GPU build
- `build_profiles_macos-x86_64.json` - macOS Intel build
- `build_profiles_macos-arm64.json` - macOS Apple Silicon build

See [LOCAL_SETTINGS.md](../LOCAL_SETTINGS.md) for configuration examples.

### Source Code Documentation

Key source directories:
- `../src/core/` - Core duplicate detection engine
- `../src/gui/` - Qt6 user interface
- `../src/gpu/` - GPU acceleration (CUDA/OpenCL)
- `../src/platform/` - Platform-specific implementations
- `../tests/` - Unit, integration, and performance tests

---

## 📝 Documentation Maintenance

### Adding New Documentation

1. Create the document in the appropriate location:
   - Active documentation → `docs/`
   - Archived documentation → `docs/archive/`
2. Add an entry to this index
3. Update related documents with cross-references
4. Mention in commit message and pull request

### Archiving Documentation

When a document becomes outdated:
1. Move to `docs/archive/` or appropriate subdirectory
2. Update this index to mark as archived
3. Add archival note at top of document
4. Update any documents that reference it

### Documentation Standards

- **Format**: Markdown (.md)
- **Naming**: Use UPPERCASE_SNAKE_CASE for main docs
- **Headers**: Use ATX-style headers (`#`, `##`, etc.)
- **Links**: Use relative paths for internal links
- **Updates**: Include "Last Updated" date at top
- **Status**: Mark document status (Current/Archived/WIP)

---

## 🆘 Need Help?

### Documentation Issues

- **Missing documentation?** Open an issue with label `documentation`
- **Outdated information?** Open an issue or submit a PR
- **Unclear instructions?** Ask in GitHub Discussions

### Getting Started

If you're new and don't know where to start:
1. Read [../README.md](../README.md)
2. Follow [DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md)
3. Review [IMPLEMENTATION_TASKS.md](IMPLEMENTATION_TASKS.md) for beginner-friendly tasks
4. Ask questions in GitHub Discussions or Issues

---

## 📈 Documentation Statistics

**Total Active Documents:** 13
**Archived Documents:** 50+
**Total Size:** ~500 KB
**Last Major Update:** November 2025

---

**Maintained by:** DupFinder Development Team
**License:** MIT
**Repository:** [github.com/Jade-biz-1/Jade-Dup-Finder](https://github.com/Jade-biz-1/Jade-Dup-Finder)
