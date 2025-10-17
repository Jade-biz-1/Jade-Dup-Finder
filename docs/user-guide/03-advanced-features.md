# Volume 3: Advanced Features

**DupFinder User Guide - Volume 3**  
**Last Updated:** October 17, 2025

---

## Table of Contents

1. [Smart Selection Algorithms](#smart-selection-algorithms)
2. [Advanced Filtering and Grouping](#advanced-filtering-and-grouping)
3. [Relationship Visualization](#relationship-visualization)
4. [Export and Reporting](#export-and-reporting)
5. [Scan Presets and Automation](#scan-presets-and-automation)
6. [Advanced File Operations](#advanced-file-operations)

---

## Smart Selection Algorithms

### Overview

Smart Selection uses intelligent algorithms to automatically select files based on various criteria, saving you time when dealing with large numbers of duplicates.

### Accessing Smart Selection

1. **From Results Window:** Click "🧠 Smart Select" button
2. **From Menu:** Results → Smart Selection
3. **Keyboard:** Ctrl+Shift+S

### Selection Modes

#### 1. Oldest Files
**Use Case:** Remove outdated versions, keep recent files
- Selects files with earliest modification dates
- Useful for cleaning up old downloads or backups
- Preserves most recent versions automatically

#### 2. Newest Files  
**Use Case:** Remove recent duplicates, keep original files
- Selects files with latest modification dates
- Good for removing recently created copies
- Preserves original files in their locations

#### 3. Largest Files
**Use Case:** Remove larger versions when smaller ones suffice
- Selects files with largest file sizes
- Useful when file quality differences exist
- Can help save more disk space

#### 4. Smallest Files
**Use Case:** Remove compressed or lower quality versions
- Selects files with smallest file sizes
- Good for keeping highest quality versions
- Preserves better quality files

#### 5. By Path Pattern
**Use Case:** Select files in specific locations
- Uses wildcard patterns to match file paths
- Examples: `*/Downloads/*`, `*/temp/*`, `*_backup.*`
- Flexible pattern matching with * and ? wildcards

#### 6. By File Type
**Use Case:** Focus on specific file types
- Select files by extension (jpg, pdf, mp4, etc.)
- Can specify multiple types: "jpg,png,gif"
- Case-insensitive matching

#### 7. By Location
**Use Case:** Clean up specific directories
- Target files in particular folders
- Multiple location patterns supported
- Useful for cleaning Downloads, Desktop, etc.

#### 8. By Multiple Criteria
**Use Case:** Complex selection rules
- Combine multiple criteria with AND/OR logic
- Most powerful and flexible option
- Can create sophisticated selection rules

### Smart Selection Dialog

```
Smart Selection
┌─────────────────────────────────────────────────────────────┐
│ Selection Mode: By Multiple Criteria ▼                     │
│                                                             │
│ ☑️ Date Range: From Oct 1, 2023 to Oct 17, 2025           │
│ ☑️ Size Range: From 1 MB to 100 MB                         │
│ ☑️ File Types: jpg,png,pdf,mp4                             │
│ ☑️ Location Patterns: */Downloads/*,*/Desktop/*            │
│                                                             │
│ Combine Criteria: ● AND (all must match)                   │
│                   ○ OR (any can match)                     │
│                                                             │
│ Limits: Max Files: 100  Percentage: 50% ████████████░░░░   │
│                                                             │
│ ┌─ Selection Preview ─────────────────────────────────────┐ │
│ │ Estimated Selection: 23 files (45.7 MB)                │ │
│ │                                                         │ │
│ │ • photo_copy.jpg (5.1 MB) - Downloads                  │ │
│ │ • document_old.pdf (1.2 MB) - Desktop                  │ │
│ │ • video_duplicate.mp4 (25.3 MB) - Downloads            │ │
│ │ • ... and 20 more files                                │ │
│ │                                                         │ │
│ │ [Update Preview]                                        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Presets: [Quick Cleanup ▼] [Save Current] [Delete]         │
│                                                             │
│ [Cancel] [Reset to Defaults] [Apply Selection]             │
└─────────────────────────────────────────────────────────────┘
```

### Selection Presets

#### Built-in Presets
- **Quick Cleanup:** Downloads folder, files older than 30 days
- **Photo Cleanup:** Image files, keep newest in each group
- **Document Cleanup:** Office files, keep files in Documents folder
- **Media Cleanup:** Video/audio files, keep largest versions
- **Backup Cleanup:** Files with "backup", "copy", "old" in names

#### Creating Custom Presets
1. Configure your selection criteria
2. Click "Save Current" button
3. Enter preset name and description
4. Preset is saved for future use

#### Managing Presets
- **Load:** Select from dropdown to apply saved criteria
- **Edit:** Modify existing preset and save again
- **Delete:** Remove presets you no longer need
- **Export:** Save presets to file for sharing

---

## Advanced Filtering and Grouping

### Advanced Filters

Beyond basic search, DupFinder offers sophisticated filtering:

#### Filter Panel
```
🔍 Advanced Filters
┌─────────────────────────────────────────────────────────────┐
│ Search: [vacation photos        ] 🔍                       │
│                                                             │
│ Size: [1 MB ▼] to [100 MB ▼]    Date: [Last Month ▼]      │
│ Type: [Images ▼]                 Path: [*/Pictures/* ]     │
│                                                             │
│ Advanced:                                                   │
│ ☑️ Show only groups with 3+ files                          │
│ ☑️ Hide groups with <10 MB savings                         │
│ ☐ Show only recommended selections                         │
│                                                             │
│ [Clear Filters] [Save as Preset] [Apply]                   │
└─────────────────────────────────────────────────────────────┘
```

#### Filter Types

**Text Search:**
- Search in file names and paths
- Supports partial matching
- Case-insensitive by default
- Regular expressions supported (advanced)

**Size Filters:**
- Minimum and maximum file sizes
- Units: bytes, KB, MB, GB
- Filter individual files or entire groups

**Date Filters:**
- Modification date ranges
- Quick presets: Today, This Week, This Month, This Year
- Custom date ranges supported

**Type Filters:**
- Filter by file extensions
- Predefined categories: Images, Documents, Videos, Audio
- Custom extension lists

**Path Filters:**
- Filter by file location
- Wildcard patterns supported
- Include/exclude specific directories

**Group Filters:**
- Minimum files per group
- Minimum savings per group
- Show only groups with recommendations

### Grouping Options

Organize results in different ways:

#### Grouping Modes

**By Hash (Default):**
- Groups files with identical content
- Most accurate grouping method
- Shows true duplicates

**By Size:**
- Groups files with same file size
- Faster than hash-based grouping
- May include false positives

**By Type:**
- Groups files by extension
- Useful for type-specific cleanup
- Shows file type distribution

**By Date:**
- Groups files by modification date
- Useful for time-based cleanup
- Shows when duplicates were created

**By Location:**
- Groups files by directory
- Shows where duplicates are located
- Useful for folder-specific cleanup

#### Grouping Dialog
```
Grouping Options
┌─────────────────────────────────────────────────────────────┐
│ Primary Grouping: By Hash (Content) ▼                      │
│ Secondary Sort: By Size (Largest First) ▼                  │
│                                                             │
│ Display Options:                                            │
│ ☑️ Show group statistics                                    │
│ ☑️ Show file counts in group headers                       │
│ ☑️ Show potential savings                                   │
│ ☐ Collapse single-file groups                              │
│                                                             │
│ Group Limits:                                               │
│ Min files per group: 2                                     │
│ Max groups to show: 1000                                   │
│                                                             │
│ [Apply] [Reset to Defaults] [Cancel]                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Relationship Visualization

### Overview

The Relationship Visualization shows visual connections between duplicate files, making it easier to understand complex duplicate relationships.

### Accessing Visualization

1. **Results Window:** Click "🔗 Relationships" tab in details panel
2. **Menu:** View → Show Relationships
3. **Keyboard:** Ctrl+R

### Visualization Features

#### Interactive Graph
```
🔗 Duplicate Relationships
┌─────────────────────────────────────────────────────────────┐
│ [Fit] [+] [-] [Reset] [Layout ▼] [Settings]                │
│                                                             │
│     Group 1          Group 2                               │
│       ●                ●─────●                             │
│      /|\\              /       \\                            │
│     ● ● ●            ●         ●                           │
│   file1 file2      file3     file4                        │
│   file3            file5                                   │
│                                                             │
│ Colors: 🔴 Group 1  🔵 Group 2  🟢 Group 3                │
│                                                             │
│ Selected: photo_vacation.jpg                               │
│ Group: 3 files, 15.2 MB total                             │
└─────────────────────────────────────────────────────────────┘
```

#### Visual Elements

**Nodes (Circles):**
- Each circle represents a file
- Size indicates file size (larger = bigger file)
- Color indicates duplicate group
- Selected files have thick borders

**Connections (Lines):**
- Lines connect duplicate files
- Solid lines = direct duplicates
- Dashed lines = similar files
- Line thickness = relationship strength

**Colors:**
- Each duplicate group has a unique color
- Recommended files have green highlights
- Selected files have red highlights

#### Layout Algorithms

**Circular Layout:**
- Arranges groups in circles
- Good for small to medium datasets
- Clear group separation

**Force-Directed Layout:**
- Uses physics simulation
- Naturally separates groups
- Good for complex relationships

**Hierarchical Layout:**
- Arranges groups in rows/columns
- Good for large datasets
- Easy to follow structure

#### Interaction

**Mouse Controls:**
- **Click:** Select file and show details
- **Double-click:** Open file preview
- **Drag:** Move nodes around
- **Wheel:** Zoom in/out
- **Right-click:** Context menu with file operations

**Keyboard Controls:**
- **Arrow keys:** Navigate between files
- **Space:** Toggle file selection
- **Enter:** Preview selected file
- **Delete:** Delete selected file

### Visualization Settings

```
Visualization Settings
┌─────────────────────────────────────────────────────────────┐
│ Display:                                                    │
│ ☑️ Show file names                                          │
│ ☑️ Show file sizes                                          │
│ ☐ Show full paths                                          │
│ ☑️ Show group colors                                        │
│                                                             │
│ Node Size: Small ●●●●●●●● Large                            │
│ Animation: None ●●●●●●●● Smooth                            │
│                                                             │
│ Layout: Force-Directed ▼                                   │
│ Update: Real-time ● Manual ○                               │
│                                                             │
│ [Apply] [Reset] [Cancel]                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Export and Reporting

### Export Formats

DupFinder supports multiple export formats for different use cases:

#### CSV (Spreadsheet)
**Best for:** Data analysis, spreadsheet applications
- Comma-separated values
- Compatible with Excel, LibreOffice, Google Sheets
- Includes all file metadata
- Easy to sort and filter

#### JSON (Structured Data)
**Best for:** Programming, data processing
- Machine-readable format
- Preserves data structure
- Includes nested group information
- API integration friendly

#### TXT (Plain Text)
**Best for:** Documentation, simple reports
- Human-readable format
- No special software required
- Good for email or documentation
- Compact file size

#### HTML (Rich Report)
**Best for:** Professional reports, presentations
- Rich formatting with CSS styling
- Embedded thumbnails for images
- Interactive elements
- Print-friendly layout

### HTML Export Features

#### Professional Styling
```html
<!DOCTYPE html>
<html>
<head>
    <title>Duplicate Files Report</title>
    <style>
        /* Professional CSS styling */
        body { font-family: 'Segoe UI', sans-serif; }
        .group { border: 1px solid #ddd; margin: 20px 0; }
        .file-item { display: flex; align-items: center; }
        .thumbnail { width: 64px; height: 64px; }
        /* Responsive design for mobile */
    </style>
</head>
<body>
    <h1>🔍 Duplicate Files Report</h1>
    
    <div class="summary">
        <div class="stat">15 Groups</div>
        <div class="stat">127 Files</div>
        <div class="stat">2.1 GB Savings</div>
    </div>
    
    <!-- File groups with thumbnails -->
</body>
</html>
```

#### Thumbnail Integration
- **Automatic generation:** Creates thumbnails for images and videos
- **Embedded images:** Thumbnails saved alongside HTML file
- **Responsive design:** Works on desktop and mobile
- **Fallback icons:** Shows file type icons for non-image files

#### Interactive Elements
- **Hover effects:** File details on mouse hover
- **Expandable groups:** Click to expand/collapse groups
- **Sorting options:** Sort by name, size, date
- **Print optimization:** Clean layout for printing

### Export Dialog

```
Export Results
┌─────────────────────────────────────────────────────────────┐
│ Format: HTML with Thumbnails ▼                             │
│ File: [duplicate_report_2025-10-17.html    ] [Browse...]   │
│                                                             │
│ Options:                                                    │
│ ☑️ Include file thumbnails (HTML only)                     │
│ ☑️ Include group statistics                                 │
│ ☑️ Include scan information                                 │
│ ☑️ Include selection status                                 │
│                                                             │
│ Content:                                                    │
│ ● All groups (15 groups, 127 files)                       │
│ ○ Selected groups only (3 groups, 23 files)               │
│ ○ Filtered results only (8 groups, 45 files)              │
│                                                             │
│ Thumbnail Size: Medium ●●●●●●●● Large                      │
│                                                             │
│ [Preview] [Export] [Cancel]                                │
└─────────────────────────────────────────────────────────────┘
```

### Report Templates

#### Executive Summary
- High-level statistics
- Potential savings summary
- Recommendations overview
- Action items list

#### Detailed Analysis
- Complete file listings
- Group-by-group breakdown
- File metadata included
- Selection recommendations

#### Technical Report
- Hash values included
- Full file paths
- Detailed timestamps
- Error logs (if any)

---

## Scan Presets and Automation

### Preset Management

#### Built-in Presets
- **Quick Scan:** Common user folders, 1MB minimum
- **Downloads Cleanup:** Downloads folder, all files
- **Photo Cleanup:** Pictures folder, images only
- **Document Cleanup:** Documents folder, office files
- **Full System:** Home directory, comprehensive scan
- **Media Cleanup:** Videos and music, large files only

#### Creating Custom Presets

1. **Configure scan settings** in the New Scan dialog
2. **Click "Save as Preset"** button
3. **Enter preset details:**
   ```
   Save Scan Preset
   ┌─────────────────────────────────────────────────────────┐
   │ Name: [Project Files Cleanup                    ]       │
   │                                                         │
   │ Description:                                            │
   │ ┌─────────────────────────────────────────────────────┐ │
   │ │ Scans project directories for duplicate source      │ │
   │ │ files, build artifacts, and documentation.          │ │
   │ │ Focuses on files larger than 5MB.                  │ │
   │ └─────────────────────────────────────────────────────┘ │
   │                                                         │
   │ Category: Work ▼                                       │
   │                                                         │
   │ ☑️ Make this my default preset                          │
   │                                                         │
   │ [Save] [Cancel]                                        │
   └─────────────────────────────────────────────────────────┘
   ```

#### Preset Categories
- **Personal:** Home cleanup, photo organization
- **Work:** Project files, document management
- **System:** System cleanup, maintenance
- **Media:** Photo, video, music organization
- **Custom:** User-defined categories

### Preset Manager

```
Preset Manager
┌─────────────────────────────────────────────────────────────┐
│ Category: All ▼          Search: [                    ] 🔍  │
├─────────────────────────────────────────────────────────────┤
│ Name                │ Category │ Last Used │ Actions        │
│ Quick Scan          │ Built-in │ Today     │ [Edit] [Run]   │
│ Downloads Cleanup   │ Built-in │ Yesterday │ [Edit] [Run]   │
│ Project Files       │ Work     │ Oct 15    │ [Edit] [Delete]│
│ Photo Organization  │ Personal │ Oct 10    │ [Edit] [Delete]│
│ System Maintenance  │ System   │ Oct 5     │ [Edit] [Delete]│
├─────────────────────────────────────────────────────────────┤
│ [New Preset] [Import] [Export] [Reset to Defaults]         │
└─────────────────────────────────────────────────────────────┘
```

#### Preset Operations
- **Edit:** Modify preset settings
- **Run:** Execute preset immediately
- **Delete:** Remove custom presets
- **Import/Export:** Share presets with others
- **Duplicate:** Create copy for modification

---

## Advanced File Operations

### Operation Queue

DupFinder uses an operation queue system for managing file operations:

#### Queue Management
```
Operation Queue
┌─────────────────────────────────────────────────────────────┐
│ Status: Processing (2 of 5 operations)                     │
│                                                             │
│ Current: Deleting photo_duplicates.jpg                     │
│ Progress: ████████████░░░░ 75% (15 of 20 files)           │
│ Speed: 2.1 MB/s                                           │
│ ETA: 45 seconds                                            │
│                                                             │
│ Queue:                                                      │
│ ✅ Delete Downloads duplicates (completed)                 │
│ 🔄 Delete Photo duplicates (in progress)                   │
│ ⏳ Move Document duplicates (queued)                       │
│ ⏳ Export scan results (queued)                            │
│ ⏳ Backup operation logs (queued)                          │
│                                                             │
│ [Pause] [Cancel Current] [View History]                   │
└─────────────────────────────────────────────────────────────┘
```

#### Queue Features
- **Sequential processing:** Operations run one at a time
- **Progress tracking:** Detailed progress for each operation
- **Pause/Resume:** Can pause and resume operations
- **Cancellation:** Cancel individual operations
- **Error handling:** Continues with next operation on errors

### Batch Operations

#### Smart Batch Processing
- **Automatic grouping:** Groups similar operations
- **Optimization:** Optimizes operation order
- **Error recovery:** Retries failed operations
- **Progress reporting:** Shows overall batch progress

#### Batch Configuration
```
Batch Operation Settings
┌─────────────────────────────────────────────────────────────┐
│ Processing:                                                 │
│ ● Sequential (safer, slower)                               │
│ ○ Parallel (faster, more resource intensive)               │
│                                                             │
│ Error Handling:                                             │
│ ● Continue on errors                                        │
│ ○ Stop on first error                                      │
│ ○ Ask for each error                                       │
│                                                             │
│ Confirmation:                                               │
│ ☑️ Confirm before each operation                            │
│ ☑️ Show detailed progress                                   │
│ ☑️ Create operation log                                     │
│                                                             │
│ [Apply] [Reset] [Cancel]                                   │
└─────────────────────────────────────────────────────────────┘
```

### Operation History

#### History Tracking
All operations are logged for review and analysis:

```
Operation History
┌─────────────────────────────────────────────────────────────┐
│ Filter: [Last 7 days ▼] [All types ▼] Search: [        ] │
├─────────────────────────────────────────────────────────────┤
│ Time     │ Operation │ Files │ Size   │ Status │ Details    │
│ 2:30 PM  │ Delete    │ 15    │ 127 MB │ ✅ Done │ [View Log] │
│ 2:25 PM  │ Move      │ 8     │ 45 MB  │ ✅ Done │ [View Log] │
│ 10:15 AM │ Delete    │ 23    │ 2.1 GB │ ✅ Done │ [View Log] │
│ 9:45 AM  │ Export    │ -     │ 15 KB  │ ✅ Done │ [View Log] │
│ Yesterday│ Delete    │ 5     │ 67 MB  │ ⚠️ Error│ [View Log] │
├─────────────────────────────────────────────────────────────┤
│ Total: 51 files processed, 2.3 GB affected                 │
│ [Export History] [Clear Old] [Retry Failed]                │
└─────────────────────────────────────────────────────────────┘
```

#### History Features
- **Detailed logs:** Complete record of all operations
- **Error tracking:** Shows failed operations with reasons
- **Statistics:** Summary of operations and data processed
- **Retry capability:** Can retry failed operations
- **Export:** Export history for analysis

---

## Next Steps

With advanced features mastered:

1. **Configure safety settings** in Volume 4
2. **Learn keyboard shortcuts** in Volume 5 for efficiency
3. **Check troubleshooting guide** in Volume 6 for common issues
4. **Experiment with different workflows** to find what works best
5. **Share presets** with team members or family

### Advanced User Tips

1. **Create workflow-specific presets** for different cleanup scenarios
2. **Use smart selection** to automate repetitive selection tasks
3. **Export results** before major cleanup operations
4. **Monitor operation queue** during large batch operations
5. **Review operation history** to track cleanup progress over time

---

**Ready for safety features?** Continue to **Volume 4: Safety & Settings** to learn about backup systems and protection features.

---

*Volume 3 Complete - Continue to Volume 4 for safety features and settings management.*