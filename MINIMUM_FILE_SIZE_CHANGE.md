# Minimum File Size Default Changed

## Date: 2025-01-12

## Change Summary

Changed the default minimum file size from **1 MB to 0 MB** (all files included).

---

## Files Modified

### 1. `include/file_scanner.h`
**Line 84:**
```cpp
// Before:
qint64 minimumFileSize = 1024;     // Skip files smaller than this (1KB default)

// After:
qint64 minimumFileSize = 0;        // Include all files by default (0 bytes)
```

### 2. `src/gui/scan_dialog.cpp`

**Constructor (Line 136):**
```cpp
// Before:
m_currentConfig.minimumFileSize = 1024 * 1024; // 1MB

// After:
m_currentConfig.minimumFileSize = 0; // 0 MB - include all files
```

**UI SpinBox Setup (Line 298-301):**
```cpp
// Before:
m_minimumSize->setRange(1, 1024);
m_minimumSize->setValue(1);

// After:
m_minimumSize->setRange(0, 1024);
m_minimumSize->setValue(0);
```

**Photos Preset (Line 876):**
```cpp
// Before:
m_minimumSize->setValue(1); // 1MB+

// After:
m_minimumSize->setValue(0); // Include all files
```

**Documents Preset (Line 897):**
```cpp
// Before:
m_minimumSize->setValue(1); // 1MB+

// After:
m_minimumSize->setValue(0); // Include all files
```

**Custom Preset (Line 945):**
```cpp
// Before:
m_minimumSize->setValue(1); // 1MB

// After:
m_minimumSize->setValue(0); // Include all files
```

**Reset to Defaults (Line 1288):**
```cpp
// Before:
defaultConfig.minimumFileSize = 1024 * 1024; // 1MB

// After:
defaultConfig.minimumFileSize = 0; // 0 MB - include all files
```

---

## Impact

### User Experience
- **Before:** Users had to manually change the minimum file size to 0 to scan all files
- **After:** All files are included by default, users can increase the minimum if desired

### Use Cases Now Supported by Default
1. ✅ Finding duplicate small configuration files
2. ✅ Finding duplicate text files (often < 1MB)
3. ✅ Finding duplicate scripts and code files
4. ✅ Finding duplicate small images (icons, thumbnails)
5. ✅ Finding duplicate small documents

### Performance Considerations
- Scanning will now include more files by default
- Users can still set a minimum size to improve performance on large directories
- The UI clearly shows the minimum size setting (0 MB by default)

---

## Testing

### Verification Steps
1. ✅ Open the application
2. ✅ Click "New Scan"
3. ✅ Check the "Min Size" field - should show "0 MB"
4. ✅ All preset buttons should set minimum size to 0 MB
5. ✅ User can still manually increase the minimum size if desired

### Test Results
- ✅ Application builds successfully
- ✅ Default value is 0 MB in UI
- ✅ All presets use 0 MB
- ✅ User can adjust from 0 to 1024 MB

---

## Backward Compatibility

### Saved Presets
- Existing saved presets will retain their configured minimum file size
- New presets will default to 0 MB

### Configuration Files
- No breaking changes to configuration format
- Existing configurations will continue to work

---

## Documentation Updates

### User-Facing Changes
- The default behavior now includes all files
- Users who want to filter by size can set a minimum in the UI
- All preset configurations now include all files by default

### Developer Notes
- `FileScanner::ScanOptions::minimumFileSize` defaults to 0
- Test files explicitly set `minimumFileSize = 1` for small test files (unchanged)
- This is intentional - tests are explicit about their requirements

---

## Rationale

### Why This Change?
1. **More Intuitive:** Users expect "scan for duplicates" to mean "all duplicates"
2. **Better Default:** Most duplicate files are small (configs, scripts, text files)
3. **User Control:** Users can still filter by size if they want
4. **Consistency:** Other duplicate finders typically include all files by default

### Previous Limitation
The 1 MB default was too restrictive and would miss:
- Configuration files (typically < 100 KB)
- Text documents (typically < 500 KB)
- Code files (typically < 100 KB)
- Small images (icons, thumbnails)
- Scripts and batch files

---

## Summary

✅ **Change Complete**
- Default minimum file size: **0 MB** (was 1 MB)
- All files now included by default
- Users can still set a minimum size if desired
- All presets updated to use 0 MB
- Application rebuilt and tested

**Result:** Users will now find ALL duplicate files by default, not just files larger than 1 MB.

---

**Modified By:** Kiro AI Assistant  
**Date:** 2025-01-12  
**Status:** ✅ COMPLETE AND TESTED
