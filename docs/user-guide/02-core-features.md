# Volume 2: Core Features

**DupFinder User Guide - Volume 2**  
**Last Updated:** October 17, 2025

---

## Table of Contents

1. [Scanning for Duplicates](#scanning-for-duplicates)
2. [Scan Configuration](#scan-configuration)
3. [Understanding Results](#understanding-results)
4. [File Selection](#file-selection)
5. [File Operations](#file-operations)
6. [Scan History](#scan-history)

---

## Scanning for Duplicates

### Quick Actions (Recommended)

The fastest way to find duplicates is using pre-configured Quick Actions:

#### 🚀 Quick Scan
**Best for:** General cleanup, first-time users
- **Scans:** Home, Downloads, Documents folders
- **File size:** 1 MB minimum (ignores small files)
- **Duration:** 2-5 minutes typically
- **Finds:** Most common duplicates in user folders

#### 📂 Downloads Cleanup
**Best for:** Cleaning up downloaded files
- **Scans:** Downloads folder only
- **File size:** All files (0 MB minimum)
- **Duration:** 30 seconds - 2 minutes
- **Finds:** Re-downloaded files, browser duplicates

#### 📸 Photo Cleanup
**Best for:** Managing photo collections
- **Scans:** Pictures folder and subfolders
- **File types:** Images only (JPG, PNG, GIF, etc.)
- **Features:** Image preview, metadata comparison
- **Finds:** Duplicate photos from multiple devices

#### 📄 Documents
**Best for:** Document organization
- **Scans:** Documents folder
- **File types:** Documents (PDF, DOC, TXT, etc.)
- **Features:** Content preview
- **Finds:** Multiple versions of documents

#### 🖥️ Full System
**Best for:** Comprehensive cleanup
- **Scans:** Entire home directory
- **File size:** 1 MB minimum
- **Duration:** 10-60 minutes depending on data
- **Finds:** All duplicates across your system

#### ⚙️ Custom Scan
**Best for:** Specific requirements
- **Opens:** Scan configuration dialog
- **Control:** Full control over all scan parameters
- **Flexibility:** Scan any combination of folders

### Custom Scan Configuration

Click "New Scan" or "Custom Scan" to access advanced options:

#### Scan Locations
```
📁 Select Folders to Scan
┌─────────────────────────────────────────┐
│ ☑️ Home (/home/username)                │
│ ☑️ Downloads (/home/username/Downloads) │
│ ☑️ Documents (/home/username/Documents) │
│ ☐ Pictures (/home/username/Pictures)   │
│ ☐ Music (/home/username/Music)         │
│ ☐ Videos (/home/username/Videos)       │
│                                         │
│ [+ Add Folder...] [- Remove]           │
└─────────────────────────────────────────┘

📋 Quick Presets:
[Downloads] [Photos] [Documents] [Media] [Custom] [Full System]
```

**Tips:**
- **Select specific folders** for faster, focused scans
- **Use presets** to quickly configure common scenarios
- **Add custom folders** using the "Add Folder" button

#### Scan Options
```
⚙️ Scan Options
┌─────────────────────────────────────────┐
│ Detection: Smart (Recommended) ▼        │
│ Min Size: 1 MB                          │
│ Max Depth: Unlimited ▼                  │
│                                         │
│ Include:                                │
│ ☐ Hidden files                          │
│ ☑️ Follow symlinks                      │
│                                         │
│ File Types:                             │
│ ☑️ All ☐ Images ☐ Documents ☐ Videos    │
│                                         │
│ Exclude Patterns:                       │
│ *.tmp, *.log, Thumbs.db                 │
│ [+ Add Pattern] [- Remove]              │
└─────────────────────────────────────────┘
```

**Key Options:**
- **Detection Mode:** Smart (recommended), Quick, Deep, or Media-specific
- **Minimum Size:** Ignore files smaller than this size
- **File Types:** Focus on specific file types
- **Exclude Patterns:** Skip files matching patterns (temp files, logs, etc.)

#### Scan Preview
```
📊 Scan Preview
┌─────────────────────────────────────────┐
│ Estimated: ~2,500 files, ~15 GB        │
│ Folders: 45 folders will be scanned    │
│ Duration: Estimated 3-5 minutes        │
│                                         │
│ ✅ Within limits (Free plan: 100 GB)   │
└─────────────────────────────────────────┘
```

The preview helps you understand:
- **Scope:** How many files will be scanned
- **Time:** Estimated scan duration
- **Limits:** Whether scan fits within plan limits

---

## Understanding Results

### Results Window Layout

After scanning, the Results Window opens with three main areas:

#### Left Panel: Duplicate Groups Tree
Shows all duplicate groups in a hierarchical view:

```
🔍 Search: [                    ] 🔍
📊 Filters: Size [All ▼] Type [All ▼] Sort [Size ▼]

☑️ Select All  📋 Recommended  🔄 Clear  🧠 Smart Select

📁 Group 1: Photos (3 files, 15.2 MB, 10.1 MB wasted)
├─ ☐ 📸 vacation_2023.jpg (5.1 MB) ✅ Keep
├─ ☑️ 📸 vacation_2023 (1).jpg (5.1 MB) 
└─ ☑️ 📸 IMG_20231015_001.jpg (5.1 MB)

📁 Group 2: Documents (2 files, 2.4 MB, 1.2 MB wasted)
├─ ☐ 📄 report.pdf (1.2 MB) ✅ Keep
└─ ☑️ 📄 report_old.pdf (1.2 MB)
```

**Understanding the Tree:**
- **Groups:** Each group contains identical files
- **Checkboxes:** ☑️ = selected for deletion, ☐ = will be kept
- **Icons:** File type indicators (📸 images, 📄 documents, etc.)
- **Recommendations:** ✅ indicates DupFinder's recommended file to keep
- **Statistics:** Shows total size and potential savings per group

#### Center Panel: File Details
Shows detailed information about the selected file:

**File Info Tab:**
```
📄 File Information
┌─────────────────────────────────────────┐
│ [File Preview - Image/Text shown here]  │
│                                         │
│ Name: vacation_2023.jpg                 │
│ Size: 5.1 MB (5,242,880 bytes)        │
│ Path: /home/user/Pictures/Vacation     │
│ Modified: October 15, 2023 2:30 PM    │
│ Type: JPEG Image                       │
│ Hash: a1b2c3d4e5f6...                  │
└─────────────────────────────────────────┘
```

**Group Info Tab:**
```
📁 Group Information
┌─────────────────────────────────────────┐
│ Files in Group: 3                       │
│ Total Size: 15.2 MB                     │
│ Wasted Space: 10.1 MB (66%)            │
│ Recommended: vacation_2023.jpg          │
│                                         │
│ All Files in Group:                     │
│ ┌─────────────────────────────────────┐ │
│ │ vacation_2023.jpg      5.1 MB  ✅   │ │
│ │ vacation_2023 (1).jpg  5.1 MB      │ │
│ │ IMG_20231015_001.jpg   5.1 MB      │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

**Relationships Tab:**
```
🔗 Duplicate Relationships
┌─────────────────────────────────────────┐
│ [Fit] [+] [-] [Reset] [Layout]         │
│                                         │
│     ●────────●                          │
│    /          \                         │
│   ●            ●                        │
│ file1.jpg   file2.jpg                  │
│                                         │
│ Visual graph showing which files are    │
│ duplicates of each other               │
└─────────────────────────────────────────┘
```

#### Right Panel: Actions
Provides tools for managing selected files:

**File Actions:**
- **🗑️ Delete File:** Move selected file to trash
- **📁 Move File:** Move to different location
- **👁️ Preview File:** View file contents
- **📂 Open Location:** Show file in file manager
- **📋 Copy Path:** Copy file path to clipboard

**Bulk Actions:**
- **🗑️ Delete Selected:** Remove all selected files
- **📁 Move Selected:** Move all selected files
- **📤 Export Results:** Save results to file

**Selection Summary:**
- Shows count of selected files
- Displays total size of selection
- Shows potential space savings

---

## File Selection

### Automatic Recommendations

DupFinder automatically recommends which files to keep based on:

1. **File Location Priority:**
   - Organized folders (Documents, Pictures) > Temporary locations (Downloads, Desktop)
   - Deeper folder structures > Root directories
   - Named folders > Generic folders

2. **File Name Quality:**
   - Original names > Names with "copy", "duplicate", "(1)", etc.
   - Descriptive names > Generic names (IMG_001, etc.)
   - Clean names > Names with special characters

3. **File Metadata:**
   - Newer modification dates (usually more recent)
   - Better file paths (organized vs. scattered)
   - Larger file sizes (in case of quality differences)

### Manual Selection

You can override recommendations:

#### Individual Selection
- **Click checkboxes** next to files to select/deselect
- **☑️ Selected:** File will be deleted
- **☐ Unselected:** File will be kept

#### Bulk Selection Tools
- **☑️ Select All:** Select all files for deletion (keeps one per group)
- **📋 Recommended:** Use DupFinder's smart recommendations
- **🔄 Clear:** Deselect all files
- **🧠 Smart Select:** Advanced selection with criteria

### Smart Selection

The Smart Selection feature provides intelligent file selection:

#### Selection Modes
1. **Oldest Files:** Select files with earliest modification dates
2. **Newest Files:** Select files with latest modification dates
3. **Largest Files:** Select files with largest file sizes
4. **Smallest Files:** Select files with smallest file sizes
5. **By Path Pattern:** Select files matching path patterns
6. **By File Type:** Select files of specific types
7. **By Location:** Select files in specific locations
8. **By Multiple Criteria:** Combine multiple selection rules

#### Using Smart Selection
1. **Click "🧠 Smart Select"** button
2. **Choose selection mode** from dropdown
3. **Set criteria** (date ranges, size limits, patterns)
4. **Preview selection** to see which files will be selected
5. **Apply selection** to update checkboxes in results

#### Example: Clean Up Old Downloads
```
Smart Selection Dialog
┌─────────────────────────────────────────┐
│ Mode: Oldest Files ▼                    │
│                                         │
│ ☑️ Date Range: Last 30 days             │
│ ☑️ Size Range: 1 MB to 100 MB           │
│ ☐ File Types: (all types)              │
│ ☑️ Location: */Downloads/*              │
│                                         │
│ Logic: ● AND (all criteria)             │
│        ○ OR (any criteria)              │
│                                         │
│ Limits: Max 50 files, 25% of total     │
│                                         │
│ Preview: 23 files will be selected      │
│ [Update Preview] [Select Files]         │
└─────────────────────────────────────────┘
```

---

## File Operations

### Individual File Operations

Right-click any file or use the Actions panel:

#### 🗑️ Delete File
- **Action:** Moves file to system trash
- **Safety:** File can be restored from trash
- **Confirmation:** Shows file details before deletion
- **Backup:** Automatic backup created before operation

#### 📁 Move File
- **Action:** Moves file to different location
- **Dialog:** File browser to choose destination
- **Validation:** Checks destination exists and is writable
- **Backup:** Original location recorded for undo

#### 👁️ Preview File
- **Images:** Shows image with zoom and pan
- **Documents:** Shows text content (PDF, TXT, etc.)
- **Other files:** Shows file information and metadata
- **Size limit:** Large files show summary instead of full content

#### 📂 Open Location
- **Action:** Opens file manager at file location
- **Highlight:** File is selected in file manager
- **Cross-platform:** Uses system default file manager

#### 📋 Copy Path
- **Action:** Copies full file path to clipboard
- **Format:** Uses system path format
- **Usage:** Paste into other applications or scripts

### Bulk Operations

Select multiple files and use bulk operations:

#### 🗑️ Delete Selected
1. **Select files** using checkboxes
2. **Click "Delete Selected"**
3. **Review confirmation dialog:**
   ```
   Delete Confirmation
   ┌─────────────────────────────────────────┐
   │ You are about to delete 5 files:       │
   │                                         │
   │ • photo_copy.jpg (5.1 MB)              │
   │ • document_old.pdf (1.2 MB)            │
   │ • video_duplicate.mp4 (25.3 MB)        │
   │ • ... and 2 more files                 │
   │                                         │
   │ Total size: 31.6 MB                    │
   │ Files will be moved to trash           │
   │                                         │
   │ [Cancel] [Delete Files]                 │
   └─────────────────────────────────────────┘
   ```
4. **Confirm operation** after reviewing
5. **Monitor progress** in progress dialog
6. **View results** showing success/failure for each file

#### 📁 Move Selected
1. **Select files** to move
2. **Click "Move Selected"**
3. **Choose destination** folder
4. **Confirm operation**
5. **Monitor progress** with detailed status

#### 📤 Export Results
Save scan results for documentation or analysis:
- **CSV:** Spreadsheet format for analysis
- **JSON:** Structured data for programming
- **TXT:** Plain text for documentation
- **HTML:** Rich format with thumbnails and styling

### Operation Progress

All file operations show detailed progress:

```
File Operation Progress
┌─────────────────────────────────────────┐
│ Deleting Files...                       │
│                                         │
│ Progress: ████████░░ 80% (4 of 5 files) │
│ Current: photo_duplicate.jpg            │
│ Speed: 2.1 MB/s                         │
│ Time Remaining: 15 seconds              │
│                                         │
│ Status: Moving to trash...              │
│                                         │
│ [Cancel Operation]                      │
└─────────────────────────────────────────┘
```

**Features:**
- **Real-time progress:** Shows current file being processed
- **Speed monitoring:** Displays operation speed
- **Time estimates:** Estimated time remaining
- **Cancellation:** Can cancel operation at any time
- **Error handling:** Shows errors without stopping operation

---

## Scan Configuration

### Detection Modes

Choose how DupFinder identifies duplicates:

#### Smart (Recommended)
- **Method:** Adaptive algorithm based on file types
- **Speed:** Balanced speed and accuracy
- **Accuracy:** High accuracy for most file types
- **Best for:** General use, mixed file types

#### Quick
- **Method:** Size + filename comparison
- **Speed:** Very fast
- **Accuracy:** Good for obvious duplicates
- **Best for:** Large datasets, initial cleanup

#### Deep
- **Method:** Full content hash comparison
- **Speed:** Slower but thorough
- **Accuracy:** Highest accuracy, finds all duplicates
- **Best for:** Critical cleanup, ensuring no false positives

#### Media
- **Method:** Content + metadata comparison
- **Speed:** Moderate
- **Accuracy:** Specialized for media files
- **Best for:** Photo and video collections

#### Perceptual (Images)
- **Method:** Visual similarity using perceptual hashing
- **Speed:** Moderate (includes image processing)
- **Accuracy:** Finds visually similar images (resized, compressed, edited)
- **Best for:** Photo libraries, finding duplicate images with variations

### File Filters

Control which files are included in the scan:

#### Size Filters
- **Minimum Size:** Skip files smaller than specified size
  - 0 MB: Include all files (thorough but slower)
  - 1 MB: Skip tiny files (recommended for most scans)
  - 10 MB: Focus on large files only (quick cleanup)

- **Maximum Size:** Skip files larger than specified size
  - Unlimited: Include all files (default)
  - 1 GB: Skip very large files (faster scanning)

#### File Type Filters
- **All:** Include all file types (default)
- **Images:** JPG, PNG, GIF, BMP, TIFF, WebP, SVG
- **Documents:** PDF, DOC, DOCX, TXT, RTF, ODT
- **Videos:** MP4, AVI, MKV, MOV, WMV, FLV
- **Audio:** MP3, WAV, FLAC, AAC, OGG, WMA
- **Archives:** ZIP, RAR, 7Z, TAR, GZ, BZ2

#### Advanced Options
- **Include Hidden Files:** Scan files starting with "." (Linux/macOS)
- **Follow Symlinks:** Follow symbolic links to other directories
- **Scan Archives:** Look inside ZIP files and other archives
- **Maximum Depth:** Limit how deep into subdirectories to scan

### Exclude Patterns

Skip files matching specific patterns:

#### Common Patterns (Pre-configured)
- `*.tmp` - Temporary files
- `*.log` - Log files
- `Thumbs.db` - Windows thumbnail cache
- `.DS_Store` - macOS folder metadata
- `*.cache` - Cache files
- `*.bak` - Backup files

#### Custom Patterns
Add your own patterns:
- `*/node_modules/*` - Skip Node.js dependencies
- `*/.git/*` - Skip Git repositories
- `*.iso` - Skip disk images
- `*_backup.*` - Skip files with "_backup" in name

#### Pattern Syntax
- `*` - Matches any characters
- `?` - Matches single character
- `*/folder/*` - Matches files in any "folder" directory
- `*.ext` - Matches files with specific extension

---

## Scan History

### Viewing Past Scans

DupFinder automatically saves all scan results:

#### Recent Scans Widget
The main dashboard shows your 5 most recent scans:
```
📊 Recent Scans
┌─────────────────────────────────────────┐
│ Oct 17, 2:30 PM - Downloads Cleanup     │
│ 📊 3 groups, 15.2 MB savings            │
│                                         │
│ Oct 16, 10:15 AM - Photo Cleanup        │
│ 📊 8 groups, 127.5 MB savings           │
│                                         │
│ Oct 15, 4:45 PM - Quick Scan            │
│ 📊 12 groups, 2.1 GB savings            │
│                                         │
│ [View All History]                      │
└─────────────────────────────────────────┘
```

**Click any scan** to view its results in the Results Window.

#### Full History Dialog
Click "View All History" for comprehensive scan management:

```
Scan History
┌─────────────────────────────────────────────────────────────────┐
│ Search: [                    ] 🔍  [Export to CSV] [Delete Old] │
├─────────────────────────────────────────────────────────────────┤
│ Date       │ Type      │ Location    │ Groups │ Savings │ Action │
│ Oct 17 2PM │ Downloads │ ~/Downloads │   3    │ 15.2 MB │ [View] │
│ Oct 16 10AM│ Photos    │ ~/Pictures  │   8    │ 127.5MB │ [View] │
│ Oct 15 4PM │ Quick     │ Multiple    │  12    │ 2.1 GB  │ [View] │
│ Oct 14 9AM │ Documents │ ~/Documents │   2    │ 5.7 MB  │ [View] │
│ Oct 13 3PM │ Custom    │ ~/Projects  │   5    │ 45.2 MB │ [View] │
└─────────────────────────────────────────────────────────────────┘
```

**Features:**
- **Search:** Find specific scans by location or type
- **Sort:** Sort by date, type, savings, etc.
- **Export:** Export history to CSV for analysis
- **Delete:** Remove old scan records
- **View:** Open any scan's results

### Managing History

#### Automatic Cleanup
- **Retention:** Scans are kept for 90 days by default
- **Storage:** Each scan record is ~1-10 KB
- **Cleanup:** Old scans are automatically removed

#### Manual Management
- **Delete individual scans** from the history dialog
- **Export important scans** before deletion
- **Clear all history** using the settings dialog

---

## Next Steps

Now that you understand the core features:

1. **Try different scan types** to see what works best for your needs
2. **Experiment with filters** to focus on specific file types
3. **Learn advanced features** in Volume 3
4. **Set up safety preferences** covered in Volume 4
5. **Master keyboard shortcuts** detailed in Volume 5

### Practice Recommendations

1. **Start with Downloads folder** - usually has obvious duplicates
2. **Use recommendations initially** - they're usually safe
3. **Preview files when uncertain** - better safe than sorry
4. **Check trash after operations** - verify files were moved correctly
5. **Export results** for important cleanup sessions

---

**Ready for more?** Continue to **Volume 3: Advanced Features** to learn about smart selection, visualization, and advanced export options.

---

*Volume 2 Complete - Continue to Volume 3 for advanced features and power user tools.*