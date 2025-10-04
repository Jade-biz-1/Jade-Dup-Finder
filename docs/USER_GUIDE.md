# DupFinder User Guide

**Version:** 1.0  
**Last Updated:** 2025-10-04  
**Target Audience:** End users (all skill levels)  

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Common Use Cases](#common-use-cases)
5. [Safety Features](#safety-features)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### First Launch

When you launch DupFinder for the first time, you'll see the main dashboard with:

- **Quick Actions**: Pre-configured scan options for common scenarios
- **Scan History**: Previously completed scans (empty on first launch)
- **System Overview**: Current disk space information

### Interface Overview

DupFinder uses a clean, modern interface designed for ease of use:

```
┌─ Main Dashboard ────────────────────────────────────────────────┐
│ 📁 New Scan    ⚙️ Settings    ❓ Help    👤 Free Plan          │
├─────────────────────────────────────────────────────────────────┤
│ Quick Actions:                                                  │
│ [🚀 Start Quick Scan]  [📂 Downloads Cleanup]                 │
│ [📸 Photo Cleanup]     [📄 Documents]                         │
│                                                                 │
│ Recent Scans: (Your scan history will appear here)             │
│                                                                 │
│ System Overview:                                                │
│ 💾 Disk Space: 512 GB    🟢 Available: 127 GB (25%)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Basic Usage

### Quick Scan (Recommended for Beginners)

1. **Click "Start Quick Scan"** from the main dashboard
2. **Select directories** you want to scan (Downloads and Documents are pre-selected)
3. **Click "Start Scan"**
4. **Wait for results** - DupFinder will show a progress dialog
5. **Review duplicates** in the Results Window

### Custom Scan Setup

For more control over the scanning process:

1. **Click "New Scan"** from the main menu
2. **Configure scan options:**
   - **Locations**: Choose specific folders to scan
   - **File Types**: Filter by file types (All, Images, Documents, Videos, Audio)
   - **Size Limits**: Set minimum file size (default: 1MB)
   - **Advanced Options**: Include hidden files, follow symlinks, etc.

3. **Preview scan scope** - DupFinder estimates the files to be scanned
4. **Start scan** when ready

### Understanding Results

The Results Window shows duplicates in a professional 3-panel interface:

```
┌─ Results Window ────────────────────────────────────────────────┐
│ 🔍 Duplicate Files Results    2 groups, 3.1 GB potential savings  │
├─────────────────────────────────────────────────────────────────┤
│ Filter & Selection │  File Details   │   Actions              │
│ ┌─────────────────┐ │ ┌─────────────────┐ │ ┌───────────────┐      │
│ │[Search box]     │ │ │File: photo.jpg │ │ │🗑️ Delete File │      │
│ │Size: [All ▼]    │ │ │Size: 2.1 MB    │ │ │📁 Move File   │      │
│ │                 │ │ │Path: ~/Photos  │ │ │👁️ Ignore      │      │
│ │☑️ Select All     │ │ │Modified: Oct 2 │ │ │👀 Preview     │      │
│ │[Recommended]    │ │ │                 │ │ │📂 Open Folder │      │
│ │                 │ │ │                 │ │ │               │      │
│ │📁 Group: photos │ │ │                 │ │ │Bulk Actions:  │      │
│ │├─☐ photo1.jpg   │ │ │                 │ │ │🗑️ Delete All │      │
│ │└─☑️ photo2.jpg   │ │ │                 │ │ │Selected: 1    │      │
│ │  (Recommended)  │ │ │                 │ │ │Savings: 2.1MB │      │
│ └─────────────────┘ │ └─────────────────┘ │ └───────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Elements:

- **Groups**: Duplicates are organized into groups of identical files
- **Checkboxes**: Select files for deletion (✅) or keeping (☐)
- **Recommendations**: DupFinder suggests which files to keep (marked "Recommended")
- **Details Panel**: Shows information about selected files
- **Actions Panel**: Provides individual and bulk file operations

---

## Advanced Features

### Smart Selection

DupFinder's smart selection algorithm helps you decide which files to keep:

- **Newest files**: Generally kept (likely the most recent version)
- **Files in organized folders**: Preferred over files in temporary locations
- **Original names**: Files without "copy", "duplicate", or numbers in names
- **Better locations**: Files in structured folders vs. Desktop/Downloads

### Filtering Options

Use filters to focus on specific types of duplicates:

- **Search**: Find files by name or path
- **Size Filter**: Show only files above certain sizes (>1MB, >10MB, etc.)
- **Type Filter**: Focus on Images, Documents, Videos, etc.
- **Date Range**: Find duplicates modified within specific timeframes

### Bulk Operations

Manage multiple files efficiently:

1. **Select multiple files** using checkboxes
2. **Choose bulk action**:
   - **Delete Selected**: Move to trash (safest option)
   - **Move Selected**: Move to a different folder
   - **Ignore Selected**: Exclude from future scans

3. **Review confirmation** dialog showing exactly what will happen
4. **Confirm operation** after reviewing the impact

### File Operations

For individual files, right-click or use the Actions panel:

- **🔍 Preview**: View file contents without opening
- **📂 Open Location**: Show file in system file manager
- **📋 Copy Path**: Copy file path to clipboard
- **🗑️ Delete**: Move to trash (recoverable)
- **📁 Move**: Move to different location
- **👁️ Ignore**: Exclude from results

---

## Common Use Cases

### 1. Downloads Folder Cleanup

**Problem**: Your Downloads folder is cluttered with duplicate files.

**Solution**:
1. Click **"Downloads Cleanup"** from Quick Actions
2. DupFinder scans your Downloads folder automatically
3. Review duplicates - often these are re-downloaded files
4. Use **"Select Recommended"** to quickly choose files to remove
5. Click **"Delete Selected"** to clean up

**Safety**: Files go to trash, not permanently deleted.

### 2. Photo Library Organization

**Problem**: Duplicate photos from multiple devices and cloud syncs.

**Solution**:
1. Click **"Photo Cleanup"** or select your Photos directory
2. DupFinder uses image comparison to find identical photos
3. **Check the Details panel** to see file creation dates and sources
4. **Keep the highest quality** or most recent versions
5. Use **bulk operations** for efficient cleanup

**Tips**: 
- DupFinder shows image previews to help you choose
- Photos in organized albums are usually preferred over those in "Camera Roll"

### 3. Document Deduplication

**Problem**: Multiple versions of documents scattered across folders.

**Solution**:
1. Select **Documents** folder(s) for scanning
2. Filter by **"Documents"** file type
3. **Review each group carefully** - document duplicates often have important differences
4. Use **Preview** feature to compare document contents
5. Keep the most recent or complete version

**Warning**: Be extra careful with documents as they may have important revisions.

### 4. Full System Scan

**Problem**: Need comprehensive duplicate cleanup across entire system.

**Solution**:
1. Click **"Full System Scan"** (Premium feature)
2. **Choose specific drives** or scan entire system
3. **Use size filters** to focus on large files first (biggest impact)
4. **Work through results systematically** by file type or size
5. **Multiple cleanup sessions** are recommended for large systems

**Note**: Full system scans can take considerable time and find many duplicates.

---

## Safety Features

DupFinder is designed with safety as the top priority:

### 1. Trash, Never Delete

- **All removals go to system trash** (Recycle Bin on Windows, Trash on macOS/Linux)
- **Files can be restored** using your system's trash/recycle bin
- **No permanent deletion** option to prevent accidents

### 2. Confirmation Dialogs

Before any file operation, DupFinder shows:
- **Exact files affected** with full paths
- **Total size** of files being operated on
- **Operation summary** in plain English
- **Cancel option** always available

### 3. System File Protection

DupFinder automatically excludes:
- **Operating system files**
- **Application binaries**
- **System configuration files**
- **Critical system directories**

### 4. Session History

- **Complete log** of all operations during your session
- **Undo capability** for recent operations (where possible)
- **Recovery information** if you need to restore files

### 5. Preview Before Action

- **File preview** available for images, documents, and text files
- **Detailed file information** including size, dates, and path
- **Group comparison** to understand relationships between duplicates

---

## Troubleshooting

### Common Issues

#### "No duplicates found"
**Possible causes:**
- Minimum file size is too high (check scan settings)
- Files are actually different (minor differences make files unique)
- Scanning incomplete due to permissions

**Solutions:**
- Lower minimum file size to 1KB for thorough scanning
- Check scan settings and included directories
- Run as administrator if permission errors occur

#### "Scan is very slow"
**Possible causes:**
- Scanning very large directories
- Including network drives or cloud storage
- System resource limitations

**Solutions:**
- Use size filters to scan large files first
- Exclude network locations for faster local scanning
- Close other resource-intensive applications

#### "Can't delete files"
**Possible causes:**
- Files are in use by another program
- Permission restrictions
- Files are on read-only media

**Solutions:**
- Close programs that might be using the files
- Run DupFinder as administrator (if needed)
- Check if files are on CD/DVD or read-only drives

#### "Results window shows strange characters"
**Possible causes:**
- File names with special characters or different encodings
- Font display issues

**Solutions:**
- This is usually display-only; functionality is not affected
- Check that your system has proper font support
- Files will still be processed correctly

### Getting Help

If you need additional assistance:

1. **Check the FAQ** in the Help menu
2. **Review error messages** carefully - they often contain solution hints
3. **Check system requirements** to ensure compatibility
4. **Contact support** through the Help menu if issues persist

### Performance Tips

- **Start with Quick Scan** to get familiar with the interface
- **Use size filters** to focus on files that will save the most space
- **Work in sessions** - don't try to clean everything at once
- **Review before bulk operations** - small mistakes are easier to fix than large ones
- **Keep DupFinder updated** for the latest features and bug fixes

---

## Conclusion

DupFinder is designed to make duplicate file management safe and efficient. Remember:

- **Safety first**: Files go to trash, never permanent deletion
- **Start small**: Use Quick Scan or focus on specific folders first  
- **Review carefully**: Especially for documents and important files
- **Use recommendations**: DupFinder's smart selection saves time
- **Take your time**: Better to be thorough than to make mistakes

Happy cleaning! 🧹✨