# Current Sprint Backlog

## Sprint Overview
**Sprint Period:** December 2025 - January 2026  
**Focus:** Phase 3 Completion & Platform Testing  
**Current Project Phase:** Phase 3: Cross-Platform Port & Branding (85% Complete)

## In Progress Tasks

### T27: GPU Acceleration Support
**Priority:** P2 (Medium)
**Status:** 🔄 IN PROGRESS
**Est. Completion:** December 2025
**Assignee:** Development Team
**Started:** November 3, 2025

#### Subtasks:
- [x] **T27.1:** GPU Environment Setup ✅ COMPLETE
- [ ] **T27.2:** CUDA Hash Calculator Implementation
  - [ ] Implement CUDA SHA-256 hash calculation kernel
  - [ ] Create CUDA memory management system
  - [ ] Add CUDA error handling and fallback mechanisms
  - [ ] Integrate CUDA calculator with HashCalculator interface

- [ ] **T27.3:** OpenCL Fallback Implementation
  - [ ] Implement OpenCL SHA-256 hash calculation kernel
  - [ ] Create OpenCL memory management system
  - [ ] Add OpenCL error handling and device detection
  - [ ] Integrate OpenCL calculator as CUDA fallback

- [ ] **T27.4:** GPU Integration and Testing
  - [ ] Update HashCalculator to use GPU acceleration
  - [ ] Add GPU performance benchmarking
  - [ ] Implement automatic CPU fallback
  - [ ] Add comprehensive GPU unit tests

- [ ] **T27.5:** UI and User Experience
  - [ ] Add GPU settings to preferences dialog
  - [ ] Display GPU status and device information
  - [ ] Add --gpu-info command line option
  - [ ] Provide clear GPU acceleration feedback

#### Acceptance Criteria:
- [ ] GPU acceleration available when libraries are installed
- [ ] Performance benchmarks show GPU speedup (3-10x for large files)
- [ ] Automatic fallback to CPU when GPU unavailable
- [ ] No performance regression on CPU-only systems
- [ ] Clear user feedback about GPU status and capabilities

#### Notes:
GPU acceleration in progress. CUDA environment successfully set up, implementation ongoing. Will provide significant performance improvements for large file processing.

---

### Cross-Platform Testing and Packaging
**Priority:** P1 (High)
**Status:** 🔄 IN PROGRESS
**Est. Completion:** December 2025
**Assignee:** Development Team

#### Subtasks:
- [ ] **Windows Platform Testing**
  - [ ] Test NSIS installer generation and installation
  - [ ] Verify all UI elements render correctly on Windows
  - [ ] Test file operations and hash generation on NTFS
  - [ ] Verify theme support and settings persistence

- [ ] **macOS Platform Testing**
  - [ ] Test DMG installer generation and installation
  - [ ] Verify all UI elements render correctly on macOS
  - [ ] Test file operations on APFS
  - [ ] Verify theme support and settings persistence
  - [ ] Test integration with macOS file system dialogs

#### Acceptance Criteria:
- [ ] Windows build produces working installer
- [ ] macOS build produces working DMG
- [ ] All UI elements render correctly on all platforms
- [ ] File operations work consistently across platforms
- [ ] Settings persist correctly on all platforms

#### Notes:
Cross-platform testing in progress. Build system infrastructure complete, now focusing on platform-specific testing and validation.

## Pending Tasks

### T24: UI/UX Enhancements for New Features
**Priority:** P2 (Medium)
**Status:** ⏸️ PENDING
**Est. Start:** January 2026
**Assignee:** Development Team

#### Subtasks:
- [ ] **T24.1:** Algorithm Selection UI
  - [ ] Add detection mode dropdown to scan dialog
  - [ ] Create algorithm configuration panel
  - [ ] Add similarity threshold sliders
  - [ ] Implement algorithm help/tooltips

- [ ] **T24.2:** File Type Configuration UI
  - [ ] Add file type inclusion/exclusion controls
  - [ ] Create archive scanning options
  - [ ] Add document content detection settings
  - [ ] Implement file type help system

#### Acceptance Criteria:
- [ ] Intuitive algorithm selection interface
- [ ] Clear explanations of each detection mode
- [ ] Easy configuration of similarity thresholds
- [ ] File type options are discoverable and usable

#### Notes:
UI enhancements for new features postponed to focus on critical implementation tasks. Will be addressed when resources allow.

---

### T28: Test Suite Architecture Improvements
**Priority:** P2 (Medium)
**Status:** ⏸️ PENDING
**Est. Start:** January 2026
**Assignee:** Development Team

#### Subtasks:
- [ ] **T28.1:** Refactor Multiple Main Functions
  - [ ] Separate tests with multiple main() into individual executables
  - [ ] Fix GUI test dependencies or make tests core-only

- [ ] **T28.2:** Update Test Architecture
  - [ ] Improve test maintainability
  - [ ] Add better test organization
  - [ ] Update build system for tests

#### Acceptance Criteria:
- [ ] All tests run without conflicts
- [ ] Test architecture supports both GUI and core tests
- [ ] Improved test execution and reporting

#### Notes:
Test architecture improvements deferred to allow focus on feature development. Will be addressed in future sprint.

---

### Remaining Branding Items
**Priority:** P2 (Medium)
**Status:** ✅ COMPLETE
**Completed:** December 23, 2025
**Assignee:** Development Team

#### Subtasks:
- [x] **Update GitHub URLs in About Dialog**
  - Location: `src/gui/about_dialog.cpp`
  - Current: Already pointing to correct Jade-biz-1/Jade-Dup-Finder repository
  - Status: No update needed - repository URL is correct

- [x] **Database Filename Migration** (Optional)
  - Current: Settings stored automatically by Qt using organization name
  - Migration: Handled automatically by Qt (CloneClean organization)
  - Status: Complete - no manual migration needed

- [x] **Code Comment Updates**
  - Found and updated: `tests/performance_benchmark.h`, `tests/CMakeLists.txt`, `tests/example_load_stress_testing.cpp`
  - Status: All DupFinder references removed from source code

#### Acceptance Criteria:
- [ ] GitHub URLs in About dialog point to actual CloneClean repository
- [ ] Database migration strategy decided
- [ ] All branding comments updated

#### Notes:
Minor branding cleanup items pending. Low priority but should be completed for consistency.

## Completed This Sprint

### T31: Modern Build System Implementation
**Priority:** P1 (High)
**Status:** ✅ COMPLETE
**Completed:** November 12, 2025
**Assignee:** Development Team

#### Summary:
Complete modern build system with profile-based orchestration implemented. Supports Windows, Linux, and macOS with multi-format packaging.

---

### T32: CloneClean Branding & UI Fixes
**Priority:** P1 (High)
**Status:** ✅ COMPLETE
**Completed:** November 26, 2025
**Assignee:** Development Team

#### Summary:
Complete rebranding from DupFinder to CloneClean with critical UI performance fixes implemented.

## Sprint Metrics

### Current Sprint Progress
- **In Progress:** 2 tasks
- **Pending:** 3 tasks
- **Completed This Sprint:** 2 tasks
- **Overall Sprint Goal Progress:** 70%

### Sprint Goals
1. ✅ Complete modern build system infrastructure
2. ✅ Complete application rebranding
3. 🔄 Complete cross-platform testing
4. 🔄 Complete GPU acceleration implementation
5. ⏸️ Complete minor UI enhancements

### Dependencies
- **T27 (GPU Acceleration)**: Requires CUDA toolkit installation and compatible GPU
- **Cross-Platform Testing**: Requires access to Windows and macOS systems
- **Branding Updates**: Requires decision on CloneClean GitHub repository location

## Next Sprint Priorities (January 2026)
1. Complete current in-progress tasks
2. Address pending UI enhancements
3. Begin Phase 4 planning
4. Continue test suite improvements

## Sprint Health
- **Velocity:** High (multiple tasks completed ahead of schedule)
- **Blockers:** Platform access for Windows/macOS testing
- **Risks:** GPU acceleration complexity
- **Quality:** All completed tasks meet acceptance criteria