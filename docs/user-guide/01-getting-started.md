# Volume 1: Getting Started

**DupFinder User Guide - Volume 1**  
**Last Updated:** October 17, 2025

---

## Table of Contents

1. [Installation](#installation)
2. [First Launch](#first-launch)
3. [Interface Overview](#interface-overview)
4. [Basic Concepts](#basic-concepts)
5. [Your First Scan](#your-first-scan)
6. [Understanding Results](#understanding-results)

---

## Installation

### System Requirements

**Minimum Requirements:**
- **OS:** Linux (Ubuntu 20.04+ or equivalent)
- **RAM:** 4 GB
- **Storage:** 100 MB for application + space for backups
- **Display:** 1024x768 resolution

**Recommended:**
- **OS:** Linux (Ubuntu 22.04+ or equivalent)
- **RAM:** 8 GB or more
- **Storage:** 1 GB free space for optimal performance
- **Display:** 1920x1080 or higher

### Installation Steps

1. **Download** DupFinder from the official website
2. **Extract** the archive to your preferred location
3. **Make executable:** `chmod +x dupfinder`
4. **Run:** `./dupfinder` or double-click the executable

### First-Time Setup

DupFinder requires no special configuration and works out of the box. On first launch, it will:
- Create configuration directories in `~/.config/dupfinder/`
- Initialize default settings
- Set up safety features automatically

---

## First Launch

### Welcome Screen

When you launch DupFinder for the first time, you'll see the main dashboard:

```
┌─ DupFinder - Main Dashboard ───────────────────────────────────┐
│ 📁 New Scan    ⚙️ Settings    ❓ Help    🔄 Restore           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 🚀 Quick Actions                                               │
│ ┌─────────────────┐ ┌─────────────────┐                       │
│ │ 🚀 Quick Scan    │ │ 📂 Downloads     │                       │
│ │ Common locations │ │ Cleanup folder   │                       │
│ └─────────────────┘ └─────────────────┘                       │
│ ┌─────────────────┐ ┌─────────────────┐                       │
│ │ 📸 Photo Cleanup │ │ 📄 Documents     │                       │
│ │ Find duplicate   │ │ Scan documents   │                       │
│ │ images          │ │ folder          │                       │
│ └─────────────────┘ └─────────────────┘                       │
│ ┌─────────────────┐ ┌─────────────────┐                       │
│ │ 🖥️ Full System   │ │ ⚙️ Custom Scan   │                       │
│ │ Comprehensive   │ │ Configure your   │                       │
│ │ scan            │ │ own scan        │                       │
│ └─────────────────┘ └─────────────────┘                       │
│                                                                 │
│ 📊 Recent Scans                                               │
│ No scans yet - start with Quick Scan above!                   │
│                                                                 │
│ 💾 System Overview                                             │
│ Disk Space: 512 GB total, 127 GB available (75% used)        │
│ Potential Savings: Run a scan to see duplicate file savings   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Elements

- **Header Bar:** Main navigation with New Scan, Settings, Help, and Restore
- **Quick Actions:** Pre-configured scan options for common scenarios
- **Recent Scans:** History of your previous scans (empty initially)
- **System Overview:** Current disk space information

---

## Interface Overview

### Main Window Components

#### 1. Header Bar
- **📁 New Scan:** Opens custom scan configuration dialog
- **⚙️ Settings:** Access application preferences and configuration
- **❓ Help:** Built-in help system and keyboard shortcuts
- **🔄 Restore:** Access backup and restore functionality

#### 2. Quick Actions Panel
Six pre-configured scan options:
- **🚀 Quick Scan:** Scans common locations (Downloads, Documents, Desktop)
- **📂 Downloads:** Focuses on Downloads folder cleanup
- **📸 Photo Cleanup:** Specialized for image duplicate detection
- **📄 Documents:** Scans document folders
- **🖥️ Full System:** Comprehensive system-wide scan
- **⚙️ Custom Scan:** Opens configuration dialog for custom settings

#### 3. Recent Scans
- Shows history of completed scans
- Click any scan to view its results
- Displays scan date, location, and findings summary

#### 4. System Overview
- Real-time disk space information
- Potential savings from previous scans
- System health indicators

### Results Window (Opens After Scanning)

The Results Window uses a professional 3-panel layout:

```
┌─ Results Window ────────────────────────────────────────────────┐
│ 🔍 Results: 3 groups found, 2.1 GB potential savings          │
├─────────────────────────────────────────────────────────────────┤
│ Tree View        │  Details Panel   │   Actions Panel         │
│ ┌─────────────────┐ ┌─────────────────┐ ┌───────────────────────┐ │
│ │🔍 [Search...]   │ │📄 File Info     │ │ File Actions          │ │
│ │📊 Size: All ▼   │ │                 │ │ ┌───────────────────┐ │ │
│ │🏷️ Type: All ▼   │ │ Name: photo.jpg │ │ │ 🗑️ Delete File    │ │ │
│ │                 │ │ Size: 2.1 MB    │ │ │ 📁 Move File      │ │ │
│ │☑️ Select All     │ │ Path: ~/Photos  │ │ │ 👁️ Preview File   │ │ │
│ │📋 Recommended   │ │ Modified: Oct 17│ │ │ 📂 Open Location  │ │ │
│ │🔄 Clear         │ │ Type: JPEG      │ │ └───────────────────┘ │ │
│ │                 │ │                 │ │                       │ │
│ │📁 Group 1       │ │📁 Group Info    │ │ Bulk Actions          │ │
│ │├─☐ photo1.jpg   │ │                 │ │ ┌───────────────────┐ │ │
│ │└─☑️ photo2.jpg   │ │ Files: 2        │ │ │ 🗑️ Delete Selected│ │ │
│ │  (Keep this)    │ │ Total: 4.2 MB   │ │ │ 📁 Move Selected  │ │ │
│ │                 │ │ Waste: 2.1 MB   │ │ │ 📤 Export Results │ │ │
│ │📁 Group 2       │ │                 │ │ └───────────────────┘ │ │
│ │├─☑️ doc1.pdf     │ │🔗 Relationships │ │                       │ │
│ │└─☐ doc2.pdf     │ │                 │ │ Selection Summary     │ │
│ │  (Keep this)    │ │ [Visual graph]  │ │ Selected: 2 files     │ │
│ └─────────────────┘ └─────────────────┘ │ Size: 2.1 MB          │ │
│                                         │ Savings: 2.1 MB       │ │
│                                         └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

#### Left Panel: Tree View
- **Search:** Find specific files or paths
- **Filters:** Filter by size, type, date
- **Selection Tools:** Select all, recommended, or clear selection
- **Duplicate Groups:** Hierarchical view of duplicate file groups
- **Checkboxes:** Select files for operations (☑️ = delete, ☐ = keep)

#### Center Panel: Details
- **File Info Tab:** Detailed information about selected file
- **Group Info Tab:** Statistics about the current duplicate group
- **Relationships Tab:** Visual graph showing file relationships

#### Right Panel: Actions
- **File Actions:** Operations for individual files
- **Bulk Actions:** Operations for multiple selected files
- **Selection Summary:** Current selection statistics

---

## Basic Concepts

### What Are Duplicates?

DupFinder identifies duplicates using multiple methods:

1. **Identical Files:** Exact byte-for-byte copies
2. **Hash Matching:** Files with identical content but possibly different names
3. **Size + Name:** Files with same size and similar names (quick detection)

### Duplicate Groups

Files are organized into **groups** where each group contains files that are duplicates of each other:

```
Group 1: vacation_photo.jpg
├─ vacation_photo.jpg (Original)
├─ vacation_photo (1).jpg (Copy)
└─ IMG_001_copy.jpg (Renamed copy)

Group 2: document.pdf
├─ document.pdf (Keep - in Documents folder)
└─ document.pdf (Delete - in Downloads folder)
```

### Selection Logic

DupFinder uses smart algorithms to recommend which files to keep:

- **✅ Keep (Recommended):** Usually the original or best-located file
- **☑️ Delete (Selected):** Duplicates that can be safely removed
- **☐ Keep (Unselected):** Files you choose to preserve

### Safety First

DupFinder prioritizes safety:
- **No permanent deletion:** All files go to system trash
- **Backup system:** Automatic backups before operations
- **Protected paths:** System files are automatically protected
- **Confirmation dialogs:** Review all operations before execution

---

## Your First Scan

### Recommended: Quick Scan

For your first experience, use the Quick Scan option:

1. **Click "🚀 Quick Scan"** from the main dashboard
2. **Review the preset configuration:**
   - Scans: Downloads, Documents, Desktop folders
   - Minimum size: 1 MB (ignores tiny files)
   - File types: All types included
   - Hidden files: Excluded (safer)

3. **Click "Start Scan"** to begin
4. **Watch the progress dialog:**
   - Shows files being scanned
   - Displays scan rate (files per second)
   - Provides estimated time remaining
   - Can be cancelled if needed

5. **Wait for completion** (usually 1-5 minutes for typical folders)

### What Happens During Scanning

1. **File Discovery:** DupFinder finds all files in selected locations
2. **Size Filtering:** Applies minimum size filter
3. **Hash Calculation:** Calculates unique fingerprints for files
4. **Duplicate Detection:** Groups files with identical content
5. **Smart Analysis:** Determines recommendations for each group

### Progress Indicators

The scan progress dialog shows:
- **Overall Progress:** Percentage complete with progress bar
- **Current Activity:** "Scanning folder: /home/user/Downloads"
- **Files Processed:** "1,247 of ~2,500 files"
- **Scan Rate:** "45 files/second"
- **Time Remaining:** "Estimated 2 minutes remaining"
- **Data Processed:** "1.2 GB scanned"

---

## Understanding Results

### No Duplicates Found

If no duplicates are found:
- **Good news:** Your selected folders are already clean!
- **Check settings:** Minimum file size might be too high
- **Expand scope:** Try scanning additional folders
- **Different file types:** Some duplicates might be in different formats

### Duplicates Found

When duplicates are found, the Results Window opens showing:

#### Group Summary
```
📁 Group 1: 3 files, 15.2 MB total, 10.1 MB wasted
├─ ☐ original_photo.jpg (5.1 MB) - Keep this ✅
├─ ☑️ photo_copy.jpg (5.1 MB) - Delete
└─ ☑️ IMG_001.jpg (5.1 MB) - Delete
```

#### Key Information
- **Group size:** How many duplicate files
- **Total size:** Combined size of all duplicates
- **Wasted space:** How much space you can save
- **Recommendations:** Which files to keep (☐) vs delete (☑️)

### Making Decisions

For each group, you can:

1. **Accept recommendations:** DupFinder's smart suggestions (recommended for beginners)
2. **Review manually:** Check each file using preview and details
3. **Modify selection:** Change which files to keep or delete
4. **Skip group:** Leave all files unchanged

### Preview Files

Before making decisions:
- **Click any file** to see details in the center panel
- **Use Preview button** to view file contents
- **Check file paths** to understand file locations
- **Compare dates** to identify the most recent version

---

## Next Steps

After completing your first scan:

1. **Review Volume 2** for detailed information about core features
2. **Explore the Results Window** to understand all available options
3. **Try different scan types** using other Quick Action buttons
4. **Check Volume 4** for safety features and settings
5. **Learn keyboard shortcuts** in Volume 5 for faster operation

### Quick Tips for New Users

- **Start small:** Begin with Downloads or a specific folder
- **Use recommendations:** DupFinder's suggestions are usually safe
- **Preview before deleting:** When in doubt, preview the file
- **Check trash:** Remember that deleted files go to trash, not permanent deletion
- **Take your time:** It's better to be careful than to make mistakes

---

**Ready to continue?** Move on to **Volume 2: Core Features** to learn about all the scanning and file management capabilities.

---

*Volume 1 Complete - Continue to Volume 2 for core features and detailed usage instructions.*