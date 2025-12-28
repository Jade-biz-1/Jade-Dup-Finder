# Manual Testing Guide - Core Integration Fixes

## Overview
This guide provides step-by-step instructions for manually testing the complete CloneClean workflow after implementing core integration fixes.

## Prerequisites
- CloneClean application built successfully
- Test data prepared (duplicate files for testing)
- Backup directory accessible

## Test Scenarios

### Scenario 1: Complete Scan-to-Delete Workflow

**Objective**: Verify the entire workflow from scan to deletion with backups

**Steps**:
1. Launch CloneClean application
2. Click "New Scan" button
3. Configure scan:
   - Select target directory with known duplicates
   - Set minimum file size (e.g., 0 MB to include all files)
   - Enable/disable hidden files as needed
4. Click "Start Scan"
5. **Verify**: Progress bar shows scan progress
6. **Verify**: Scan completes and shows file count
7. **Verify**: Duplicate detection starts automatically
8. **Verify**: Detection progress is shown
9. **Verify**: Results window opens with duplicate groups
10. **Verify**: Statistics show correct counts and space savings
11. Select files to delete (keep one from each group)
12. Click "Delete Selected Files"
13. **Verify**: Confirmation dialog appears
14. Confirm deletion
15. **Verify**: Files are deleted from filesystem
16. **Verify**: Backups are created in backup directory
17. **Verify**: Results display updates to remove deleted files

**Expected Results**:
- ✅ Scan completes without errors
- ✅ Duplicate detection runs automatically
- ✅ Results display shows actual duplicate groups
- ✅ File deletion works correctly
- ✅ Backups are created before deletion
- ✅ UI updates after deletion

---

### Scenario 2: File Restore Functionality

**Objective**: Verify files can be restored from backups

**Steps**:
1. Complete Scenario 1 (delete files with backups)
2. Navigate to backup directory
3. Note the backup file paths
4. In CloneClean, use restore functionality (if available in UI)
   - OR use FileManager::restoreFiles() programmatically
5. Select backup files to restore
6. Specify target directory (original location)
7. Click "Restore"
8. **Verify**: Files are restored to original locations
9. **Verify**: File content matches original
10. **Verify**: File metadata is preserved

**Expected Results**:
- ✅ Files restore successfully
- ✅ Content is identical to original
- ✅ Files appear in original locations

---

### Scenario 3: Export Functionality

**Objective**: Verify results can be exported in multiple formats

**Steps**:
1. Complete a scan with duplicate results
2. In Results Window, click "Export Results"
3. **Test CSV Export**:
   - Select CSV format
   - Choose save location
   - Click Save
   - **Verify**: CSV file is created
   - **Verify**: Open CSV in spreadsheet application
   - **Verify**: Data is properly formatted with headers
   - **Verify**: All duplicate groups are included
4. **Test JSON Export**:
   - Click "Export Results" again
   - Select JSON format
   - Choose save location
   - Click Save
   - **Verify**: JSON file is created
   - **Verify**: Open JSON in text editor
   - **Verify**: JSON is valid and well-formatted
   - **Verify**: All data fields are present
5. **Test Text Export**:
   - Click "Export Results" again
   - Select TXT format
   - Choose save location
   - Click Save
   - **Verify**: Text file is created
   - **Verify**: Open text file
   - **Verify**: Report is human-readable
   - **Verify**: Summary and details are included

**Expected Results**:
- ✅ CSV export works with proper escaping
- ✅ JSON export creates valid JSON
- ✅ Text export is readable and complete
- ✅ All formats include complete data

---

### Scenario 4: File Preview

**Objective**: Verify file preview works for different file types

**Steps**:
1. Complete a scan with various file types (images, text, other)
2. In Results Window, select an image file
3. Click "Preview" button
4. **Verify**: Image preview dialog opens
5. **Verify**: Image is displayed correctly
6. **Verify**: Image dimensions and size are shown
7. Close preview
8. Select a text file (.txt, .log, .cpp, etc.)
9. Click "Preview" button
10. **Verify**: Text preview dialog opens
11. **Verify**: Text content is displayed
12. **Verify**: File info is shown
13. Close preview
14. Select a binary file (e.g., .exe, .bin)
15. Click "Preview" button
16. **Verify**: File info dialog appears
17. **Verify**: File metadata is displayed
18. **Verify**: Message indicates preview not available

**Expected Results**:
- ✅ Image files show image preview
- ✅ Text files show content preview
- ✅ Binary files show file information
- ✅ No crashes with any file type

---

### Scenario 5: Error Handling

**Objective**: Verify application handles errors gracefully

**Steps**:
1. **Test Permission Denied**:
   - Create a read-only file
   - Try to delete it
   - **Verify**: Error message is shown
   - **Verify**: Application doesn't crash
   - **Verify**: Other files can still be processed

2. **Test Non-Existent Path**:
   - Try to scan a non-existent directory
   - **Verify**: Error message is shown
   - **Verify**: Application remains stable

3. **Test Scan Cancellation**:
   - Start a large scan
   - Click "Cancel" immediately
   - **Verify**: Scan stops
   - **Verify**: Partial results are handled
   - **Verify**: UI returns to ready state

4. **Test Empty Directory**:
   - Scan an empty directory
   - **Verify**: Scan completes
   - **Verify**: Message indicates no files found
   - **Verify**: No crashes

5. **Test Protected Files**:
   - Try to delete system files (if accessible)
   - **Verify**: Protection warning appears
   - **Verify**: Files are not deleted
   - **Verify**: Operation completes safely

**Expected Results**:
- ✅ All errors are handled gracefully
- ✅ Error messages are clear and helpful
- ✅ Application remains stable
- ✅ No crashes under error conditions

---

### Scenario 6: Multiple File Operations

**Objective**: Verify batch operations work correctly

**Steps**:
1. Complete a scan with multiple duplicate groups
2. Select files from multiple groups (e.g., 10-20 files)
3. Click "Delete Selected Files"
4. **Verify**: Confirmation shows correct count
5. Confirm deletion
6. **Verify**: All selected files are deleted
7. **Verify**: All backups are created
8. **Verify**: Progress is shown during operation
9. **Verify**: Results display updates correctly

**Expected Results**:
- ✅ Batch deletion works correctly
- ✅ All files are processed
- ✅ All backups are created
- ✅ UI updates properly

---

### Scenario 7: Move Operations

**Objective**: Verify file move functionality

**Steps**:
1. Complete a scan with duplicates
2. Select files to move
3. Click "Move Selected Files"
4. **Verify**: Folder selection dialog appears
5. Select destination folder
6. Click OK
7. **Verify**: Files are moved to destination
8. **Verify**: Backups are created
9. **Verify**: Files no longer in original location
10. **Verify**: Files exist in destination
11. **Verify**: Results display updates

**Expected Results**:
- ✅ Files move successfully
- ✅ Backups are created
- ✅ UI updates correctly

---

### Scenario 8: Application Stability

**Objective**: Verify application remains stable under various conditions

**Steps**:
1. **Test Rapid Operations**:
   - Start scan
   - Cancel immediately
   - Start another scan
   - **Verify**: No crashes

2. **Test Window Management**:
   - Open Results Window
   - Close Results Window
   - Open again
   - **Verify**: Window opens correctly each time

3. **Test Large File Sets**:
   - Scan directory with 1000+ files
   - **Verify**: Scan completes
   - **Verify**: Memory usage is reasonable
   - **Verify**: UI remains responsive

4. **Test Long-Running Operations**:
   - Start large scan
   - Let it run to completion
   - **Verify**: Progress updates throughout
   - **Verify**: Completion is detected
   - **Verify**: Results are displayed

**Expected Results**:
- ✅ Application remains stable
- ✅ No memory leaks
- ✅ UI stays responsive
- ✅ All operations complete successfully

---

## Test Data Preparation

### Creating Test Duplicates

```bash
# Create test directory
mkdir -p ~/cloneclean_test/group1
mkdir -p ~/cloneclean_test/group2
mkdir -p ~/cloneclean_test/unique

# Create duplicate group 1 (3 copies)
echo "Duplicate content 1" > ~/cloneclean_test/group1/file1.txt
cp ~/cloneclean_test/group1/file1.txt ~/cloneclean_test/group1/file1_copy.txt
cp ~/cloneclean_test/group1/file1.txt ~/cloneclean_test/group1/file1_backup.txt

# Create duplicate group 2 (2 copies)
echo "Duplicate content 2" > ~/cloneclean_test/group2/file2.txt
cp ~/cloneclean_test/group2/file2.txt ~/cloneclean_test/group2/file2_copy.txt

# Create unique files
echo "Unique content 1" > ~/cloneclean_test/unique/unique1.txt
echo "Unique content 2" > ~/cloneclean_test/unique/unique2.txt

# Create image duplicates (if you have test images)
# cp test_image.jpg ~/cloneclean_test/group1/image1.jpg
# cp test_image.jpg ~/cloneclean_test/group1/image1_copy.jpg
```

---

## Verification Checklist

### Core Functionality
- [ ] Application launches successfully
- [ ] Scan configuration dialog works
- [ ] File scanning completes
- [ ] Duplicate detection runs automatically
- [ ] Results display shows correct data
- [ ] File deletion works
- [ ] Backups are created
- [ ] File restore works
- [ ] File move works

### UI/UX
- [ ] Progress indicators work
- [ ] Status messages are clear
- [ ] Buttons are enabled/disabled appropriately
- [ ] Dialogs appear and function correctly
- [ ] Results tree is populated correctly
- [ ] Statistics are accurate

### Export Functionality
- [ ] CSV export works
- [ ] JSON export works
- [ ] Text export works
- [ ] Exported files are valid
- [ ] All data is included

### Preview Functionality
- [ ] Image preview works
- [ ] Text preview works
- [ ] File info display works
- [ ] Preview dialogs are functional

### Error Handling
- [ ] Permission errors handled
- [ ] Non-existent paths handled
- [ ] Cancellation works
- [ ] Empty directories handled
- [ ] Protected files handled

### Safety Features
- [ ] Backups created before deletion
- [ ] Backups created before move
- [ ] Protected files not deleted
- [ ] Restore functionality works
- [ ] Undo history tracked

---

## Known Issues / Limitations

Document any issues found during testing:

1. **Issue**: [Description]
   - **Severity**: [Low/Medium/High]
   - **Steps to Reproduce**: [Steps]
   - **Expected**: [Expected behavior]
   - **Actual**: [Actual behavior]

2. **Limitation**: [Description]
   - **Impact**: [Impact description]
   - **Workaround**: [If available]

---

## Test Results Summary

### Test Execution Date: [Date]
### Tester: [Name]
### Build Version: [Version]

| Scenario | Status | Notes |
|----------|--------|-------|
| 1. Scan-to-Delete Workflow | ⬜ Pass / ⬜ Fail | |
| 2. File Restore | ⬜ Pass / ⬜ Fail | |
| 3. Export Functionality | ⬜ Pass / ⬜ Fail | |
| 4. File Preview | ⬜ Pass / ⬜ Fail | |
| 5. Error Handling | ⬜ Pass / ⬜ Fail | |
| 6. Multiple File Operations | ⬜ Pass / ⬜ Fail | |
| 7. Move Operations | ⬜ Pass / ⬜ Fail | |
| 8. Application Stability | ⬜ Pass / ⬜ Fail | |

### Overall Assessment
- **Total Tests**: 8
- **Passed**: [Count]
- **Failed**: [Count]
- **Blocked**: [Count]

### Recommendation
⬜ Ready for Release
⬜ Needs Minor Fixes
⬜ Needs Major Fixes

---

## Conclusion

This manual testing guide covers all critical workflows and integration points. Complete all scenarios and document results in the Test Results Summary section.

For automated testing results, refer to:
- test_scan_to_delete_workflow (10/10 passing)
- test_restore_functionality (10/10 passing)
- test_error_scenarios (10/10 passing)
