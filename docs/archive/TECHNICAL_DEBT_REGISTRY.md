# Technical Debt Registry

## Overview
This document tracks technical debt items that need to be addressed in future development cycles. Items are categorized by priority and impact.

## High Priority Items

### GPU Acceleration Support
**Status**: Deferred  
**Impact**: Performance Enhancement  
**Effort**: 2-3 weeks  
**Details**: See `GPU_BUILD_ISSUES_RECORD.md`

**Description**: GPU acceleration code exists but is disabled due to missing dependencies. When GPU features are planned, build system needs updates and library dependencies must be resolved.

**Action Required**:
- Install GPU development libraries (OpenCL, CUDA, Vulkan)
- Update CMake configuration for GPU detection
- Enable GPU-specific code compilation
- Test GPU vs CPU performance benchmarks

---

## Medium Priority Items

### Test Suite Architecture
**Status**: Partially Fixed  
**Impact**: Development Workflow  
**Effort**: 1 week  

**Description**: Some performance tests have architectural issues with multiple main() functions and missing GUI dependencies.

**Action Required**:
- Refactor tests with multiple main() functions into separate executables
- Fix GUI test dependencies or make tests core-only
- Update test architecture for better maintainability

### Build System Warnings
**Status**: Active  
**Impact**: Code Quality  
**Effort**: 2-3 days  

**Description**: Multiple compiler warnings about type conversions and sign comparisons in hash_calculator.cpp and other core files.

**Action Required**:
- Fix conversion warnings (qint64 to double, qsizetype to int)
- Address sign comparison warnings
- Update code to use proper type casting

---

## Low Priority Items

### Code Documentation
**Status**: Ongoing  
**Impact**: Maintainability  
**Effort**: Ongoing  

**Description**: Some code sections need better documentation and comments.

**Action Required**:
- Add comprehensive API documentation
- Update inline comments for complex algorithms
- Create developer documentation

---

## Resolved Items

### ✅ Qt6::Widgets Dependency Issues
**Resolved**: October 30, 2025  
**Description**: Test executables were missing Qt6::Widgets dependencies  
**Solution**: Added Qt6::Widgets to all test target_link_libraries

### ✅ HashOptions API Compatibility
**Resolved**: October 30, 2025  
**Description**: Test code using outdated HashOptions member names  
**Solution**: Updated all test files to use current API

### ✅ Light Theme Contrast Issues
**Resolved**: October 30, 2025  
**Description**: Poor text visibility in light theme selections  
**Solution**: Implemented comprehensive theme-aware styling with proper contrast

---

## Tracking Information

**Last Updated**: October 30, 2025  
**Next Review**: When GPU development is planned  
**Responsible**: Development Team  

## Notes
- GPU issues are the highest priority technical debt when GPU features are planned
- Test architecture issues don't block main application functionality
- Build warnings should be addressed in next maintenance cycle