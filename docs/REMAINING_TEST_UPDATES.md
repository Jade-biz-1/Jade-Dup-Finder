# Remaining Test Updates Required

**Date:** October 25, 2025  
**Status:** In Progress - Core signal tests working, older tests need API updates

## Summary

During the test suite signal implementation fixes, we successfully got the core signal-related tests building and passing (27/28 tests, 96.4% success rate). However, several older tests need API updates to match current implementations.

## Completed Today ✅

1. **Test Framework Fixes:**
   - Fixed `test_base.h` macros to handle both QString and const char* messages
   - Made `shouldRunTest()` method public for test executable access
   - Added Qt6::Test include directories to ui_automation library

2. **Successfully Built and Tested:**
   - `test_scan_progress_tracking` - 8/8 PASSED ✅
   - `test_filescanner_hashcalculator` - 9/10 PASSED ✅
   - `test_scan_to_delete_workflow` - 10/10 PASSED ✅

3. **API Updates Completed:**
   - `test_hc002b_batch_processing.cpp` - Added QDirIterator include ✅
   - `test_error_scenarios.cpp` - Updated SafetyManager API usage ✅

## Remaining Work

### 1. test_hc002c_io_optimization.cpp

**Issue:** Uses obsolete HashCalculator::HashOptions members

**Required Changes:**
- Line 178: `memoryMappingEnabled` → Remove (feature removed)
- Line 179: `memoryMappingThreshold` → `memoryMapThreshold`
- Line 180: `readAheadEnabled` → Remove or use `readAheadSize > 0`
- Line 181: `asyncIOEnabled` → Remove (feature removed)
- Lines 197-200: `readAheadBufferSize` → `readAheadSize`
- Lines 216-219: `asyncIOThreshold` → Remove (feature removed)
- Lines 235-242: `bufferPoolEnabled`, `directIOEnabled` → Remove (features removed)

**Recommendation:** Review HashCalculator::HashOptions current structure and update test to match

### 2. Main Application - ScanErrorDialog

**Issue:** MainWindow references ScanErrorDialog but it's not being built

**Missing Methods:**
- `ScanErrorDialog::ScanErrorDialog(QWidget*)`
- `ScanErrorDialog::setErrors(QList<FileScanner::ScanErrorInfo> const&)`
- `ScanProgressDialog::viewErrorsRequested()` signal

**Files Affected:**
- `src/gui/main_window.cpp` - Lines 136, 139, 148, 854

**Recommendation:** Either:
1. Complete ScanErrorDialog implementation, OR
2. Comment out ScanErrorDialog references until it's implemented

### 3. Integration Tests Needing Dependencies

**Tests:**
- `test_error_scenarios` - Needs additional GUI sources
- `test_integration_workflow` - Needs dependency review
- `test_end_to_end_workflow` - Needs dependency review
- `test_restore_functionality` - Needs dependency review
- `test_file_scanner_coverage` - Needs dependency review
- `test_file_scanner_performance` - Needs dependency review

**Common Issue:** Missing GUI source files or headers in CMakeLists.txt

**Solution Pattern:** Follow the model from `test_scan_to_delete_workflow`:
```cmake
add_executable(test_name
    test_file.cpp
    ${TEST_COMMON_SOURCES}
    ${TEST_GUI_SOURCES}  # Add this
    # Additional theme/dialog sources if needed
    ../src/core/selection_history_manager.cpp
    ../src/core/scan_history_manager.cpp
    ../src/core/theme_manager.cpp
    # ... other dependencies
    ${TEST_COMMON_HEADERS}
    # Additional headers for MOC
    ../include/file_operation_queue.h
)

target_link_libraries(test_name
    Qt6::Core
    Qt6::Widgets  # Add this if GUI components used
    Qt6::Test
    Qt6::Concurrent
)
```

### 4. Unit Tests Linking

**Issue:** unit_tests and integration_tests still have undefined references

**Required:** Review and add missing source files to their CMakeLists.txt entries

## Quick Fixes for Tomorrow

### Priority 1: Fix test_hc002c_io_optimization.cpp (15 min)

Check current HashCalculator::HashOptions structure:
```bash
grep -A 50 "struct HashOptions" include/hash_calculator.h
```

Update test to use correct member names.

### Priority 2: Handle ScanErrorDialog (30 min)

Option A - Quick: Comment out references in main_window.cpp
```cpp
// Temporarily disabled - ScanErrorDialog not yet implemented
// if (!m_scanErrorDialog) {
//     m_scanErrorDialog = new ScanErrorDialog(this);
// }
```

Option B - Complete: Implement ScanErrorDialog
- Create `src/gui/scan_error_dialog.cpp`
- Create `include/scan_error_dialog.h`
- Add to CMakeLists.txt

### Priority 3: Fix Integration Test Dependencies (1-2 hours)

For each failing test:
1. Check what headers it includes
2. Add corresponding source files to CMakeLists.txt
3. Add required Qt modules (Widgets, Network, etc.)
4. Add theme system sources if ThemeManager is used
5. Rebuild and verify

## Test Organization Recommendation

Consider organizing tests into categories in CMakeLists.txt:

```cmake
# Core Unit Tests (Minimal dependencies)
add_executable(core_unit_tests ...)

# GUI Integration Tests (Full GUI stack)
add_executable(gui_integration_tests ...)

# Performance Tests (Benchmarking)
add_executable(performance_tests ...)
```

This would make dependencies clearer and builds faster for specific test categories.

## Success Metrics

**Current:**
- 3/30+ test executables building
- 27/28 built tests passing (96.4%)
- Core signal implementations verified ✅

**Goal:**
- All test executables building
- >95% test pass rate
- All signal implementations verified
- Updated API compatibility across all tests

## Estimated Effort

- **test_hc002c_io_optimization.cpp:** 15 minutes
- **ScanErrorDialog fix:** 30 minutes
- **Integration test dependencies:** 1-2 hours
- **Unit test linking:** 30-60 minutes
- **Verification and testing:** 30 minutes

**Total:** 3-4 hours to complete all remaining test updates

## Notes for Next Session

1. Start with quick fixes (test_hc002c_io_optimization)
2. Decide on ScanErrorDialog approach (comment out vs. implement)
3. Systematically add dependencies to integration tests
4. Run full test suite after all tests build
5. Document any new API patterns discovered
