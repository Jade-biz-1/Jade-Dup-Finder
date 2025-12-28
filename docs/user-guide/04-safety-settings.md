# Volume 4: Safety & Settings

**CloneClean User Guide - Volume 4**  
**Last Updated:** October 17, 2025

---

## Table of Contents

1. [Safety Features Overview](#safety-features-overview)
2. [Backup and Restore System](#backup-and-restore-system)
3. [Protected Paths Management](#protected-paths-management)
4. [Application Settings](#application-settings)
5. [Safety Preferences](#safety-preferences)
6. [Recovery Procedures](#recovery-procedures)

---

## Safety Features Overview

CloneClean prioritizes data safety with multiple layers of protection to prevent accidental data loss.

### Core Safety Principles

1. **No Permanent Deletion:** Files are moved to system trash, never permanently deleted
2. **Automatic Backups:** Critical operations create automatic backups
3. **Protected Paths:** System and important directories are automatically protected
4. **Confirmation Dialogs:** All destructive operations require confirmation
5. **Operation Logging:** Complete audit trail of all operations
6. **Undo Capability:** Most operations can be undone

### Safety Status Indicator

The main window shows current safety status:

```
🛡️ Safety Status: PROTECTED
┌─────────────────────────────────────────────────────────────┐
│ ✅ Backup system active                                     │
│ ✅ Protected paths configured                               │
│ ✅ Trash integration enabled                                │
│ ✅ Operation logging enabled                                │
│ ⚠️ 15 GB backup space remaining                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Backup and Restore System

### Automatic Backup Creation

CloneClean automatically creates backups before major operations:

#### When Backups Are Created
- Before deleting files
- Before moving files to different locations
- Before bulk operations (10+ files)
- Before operations on protected file types
- When explicitly requested by user

#### Backup Storage
```
Backup Location: ~/.local/share/cloneclean/backups/

Structure:
├── 2025-10-17_143022_delete_operation/
│   ├── manifest.json
│   ├── files/
│   │   ├── photo1.jpg
│   │   ├── document.pdf
│   │   └── video.mp4
│   └── metadata/
│       ├── original_paths.txt
│       └── operation_log.txt
├── 2025-10-17_102315_move_operation/
└── 2025-10-16_165430_bulk_delete/
```

### Backup Management

#### Backup Settings
```
Backup Settings
┌─────────────────────────────────────────────────────────────┐
│ Automatic Backups: ● Enabled ○ Disabled                   │
│                                                             │
│ Backup Location: [~/.local/share/cloneclean/backups/]       │
│ [Browse...] [Reset to Default]                             │
│                                                             │
│ Storage Limits:                                             │
│ Max backup size: 10 GB                                     │
│ Max backup age: 30 days                                    │
│ Max backup count: 50 backups                               │
│                                                             │
│ Backup Triggers:                                            │
│ ☑️ Before file deletion                                     │
│ ☑️ Before file moves                                        │
│ ☑️ Before bulk operations (10+ files)                      │
│ ☑️ Before operations on important files                     │
│                                                             │
│ Cleanup:                                                    │
│ ● Automatic cleanup when limits exceeded                   │
│ ○ Ask before cleanup                                        │
│ ○ Never cleanup automatically                               │
│                                                             │
│ [Apply] [Test Backup] [View Backups]                       │
└─────────────────────────────────────────────────────────────┘
```

### Restore Operations

#### Accessing Restore
1. **Main Window:** Click "🔄 Restore" button
2. **Menu:** File → Restore from Backup
3. **Keyboard:** Ctrl+Shift+R
4. **Emergency:** Available even if main database is corrupted

#### Restore Dialog
```
Restore from Backup
┌─────────────────────────────────────────────────────────────┐
│ Available Backups:                                          │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Oct 17, 2:30 PM - Delete Operation                     │ │
│ │ 📊 15 files, 127 MB                                    │ │
│ │ 📁 /home/user/Pictures/vacation/                       │ │
│ │ Status: ✅ Complete backup available                   │ │
│ │                                                         │ │
│ │ Oct 17, 10:23 AM - Move Operation                      │ │
│ │ 📊 8 files, 45 MB                                      │ │
│ │ 📁 /home/user/Documents/                               │ │
│ │ Status: ✅ Complete backup available                   │ │
│ │                                                         │ │
│ │ Oct 16, 4:54 PM - Bulk Delete                          │ │
│ │ 📊 23 files, 2.1 GB                                    │ │
│ │ 📁 Multiple locations                                   │ │
│ │ Status: ⚠️ Partial backup (some files too large)      │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Selected Backup Details:                                    │
│ Date: October 17, 2025 2:30 PM                            │
│ Operation: Delete Operation                                 │
│ Files: 15 files (127 MB)                                   │
│ Location: /home/user/Pictures/vacation/                    │
│                                                             │
│ Restore Options:                                            │
│ ● Restore all files to original locations                  │
│ ○ Restore selected files only                              │
│ ○ Restore to different location                            │
│                                                             │
│ ☑️ Verify file integrity after restore                     │
│ ☑️ Create restore log                                       │
│                                                             │
│ [Preview Files] [Restore] [Delete Backup] [Cancel]        │
└─────────────────────────────────────────────────────────────┘
```

#### Restore Process
1. **Select backup** from the list
2. **Choose restore options** (all files, selected files, or different location)
3. **Preview files** to verify backup contents
4. **Click Restore** to begin restoration
5. **Monitor progress** in the restore progress dialog
6. **Verify results** using the restore log

---

## Protected Paths Management

### Automatic Protection

CloneClean automatically protects critical system paths:

#### System Protected Paths
- `/bin/`, `/sbin/`, `/usr/bin/` - System binaries
- `/etc/` - System configuration
- `/var/log/` - System logs
- `/proc/`, `/sys/` - Virtual filesystems
- `~/.config/` - User configuration (optional)
- `~/.ssh/` - SSH keys and configuration

#### Application Protected Paths
- CloneClean configuration directory
- Active backup directories
- Currently open files
- Files in use by other applications

### Custom Protection Rules

#### Protection Manager
```
Protected Paths Manager
┌─────────────────────────────────────────────────────────────┐
│ [Add Path] [Add Pattern] [Import] [Export]                  │
├─────────────────────────────────────────────────────────────┤
│ Type    │ Path/Pattern              │ Status │ Actions       │
│ System  │ /bin/                     │ ✅ On  │ [View]        │
│ System  │ /etc/                     │ ✅ On  │ [View]        │
│ System  │ ~/.ssh/                   │ ✅ On  │ [View]        │
│ Custom  │ ~/important_projects/     │ ✅ On  │ [Edit][Delete]│
│ Custom  │ *.tax                     │ ✅ On  │ [Edit][Delete]│
│ Custom  │ ~/family_photos/          │ ✅ On  │ [Edit][Delete]│
│ Pattern │ *_backup.*                │ ⚠️ Off │ [Enable]      │
├─────────────────────────────────────────────────────────────┤
│ Protection Status: 6 active rules, 1 disabled              │
│ Last scan: 127 files protected, 0 violations               │
└─────────────────────────────────────────────────────────────┘
```

#### Adding Protection Rules

**Add Specific Path:**
```
Add Protected Path
┌─────────────────────────────────────────────────────────────┐
│ Path: [/home/user/important_documents/          ] [Browse]  │
│                                                             │
│ Protection Type:                                            │
│ ● Complete protection (no operations allowed)              │
│ ○ Read-only (preview allowed, no modifications)            │
│ ○ Backup required (operations require backup)              │
│                                                             │
│ Apply to:                                                   │
│ ☑️ Files in this directory                                  │
│ ☑️ Subdirectories                                           │
│ ☐ Hidden files                                             │
│                                                             │
│ Description: [Important tax documents and receipts]        │
│                                                             │
│ [Test Path] [Add] [Cancel]                                 │
└─────────────────────────────────────────────────────────────┘
```

**Add Pattern Rule:**
```
Add Protection Pattern
┌─────────────────────────────────────────────────────────────┐
│ Pattern: [*_important.*                         ]           │
│                                                             │
│ Pattern Type:                                               │
│ ● Filename pattern (matches file names)                    │
│ ○ Path pattern (matches full paths)                        │
│                                                             │
│ Examples of files this pattern will protect:               │
│ • document_important.pdf                                   │
│ • photo_important.jpg                                      │
│ • backup_important.zip                                     │
│                                                             │
│ Protection Level: Complete Protection ▼                    │
│                                                             │
│ [Test Pattern] [Add] [Cancel]                              │
└─────────────────────────────────────────────────────────────┘
```

### Protection Violations

When protected files are encountered:

```
Protection Violation Detected
┌─────────────────────────────────────────────────────────────┐
│ ⚠️ Attempted operation on protected file                   │
│                                                             │
│ File: /home/user/important_documents/taxes_2024.pdf        │
│ Operation: Delete                                           │
│ Protection Rule: ~/important_documents/ (Complete)         │
│                                                             │
│ This file is protected by a custom rule. The requested     │
│ operation cannot be performed.                              │
│                                                             │
│ Options:                                                    │
│ ● Skip this file and continue                              │
│ ○ Temporarily disable protection for this operation        │
│ ○ Remove protection rule permanently                        │
│ ○ Cancel entire operation                                   │
│                                                             │
│ [Apply to All] [Continue] [Cancel]                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Application Settings

### General Settings

```
General Settings
┌─────────────────────────────────────────────────────────────┐
│ Startup:                                                    │
│ ☑️ Show dashboard on startup                                │
│ ☑️ Check for updates automatically                          │
│ ☐ Start minimized to system tray                          │
│ ☑️ Remember window size and position                        │
│                                                             │
│ Interface:                                                  │
│ Theme: System Default ▼                                    │
│ Language: English ▼                                        │
│ Font size: Medium ▼                                        │
│                                                             │
│ Behavior:                                                   │
│ ☑️ Confirm before closing with active operations           │
│ ☑️ Show tooltips and help text                             │
│ ☑️ Enable keyboard shortcuts                                │
│ ☐ Minimize to tray instead of closing                     │
│                                                             │
│ [Apply] [Reset to Defaults] [Cancel]                       │
└─────────────────────────────────────────────────────────────┘
```

### Scanning Settings

```
Scanning Settings
┌─────────────────────────────────────────────────────────────┐
│ Default Detection Mode: Smart ▼                            │
│ Default Minimum Size: 1 MB                                 │
│                                                             │
│ Performance:                                                │
│ Thread count: Auto (8 threads) ▼                          │
│ Memory limit: 2 GB                                         │
│ I/O priority: Normal ▼                                     │
│                                                             │
│ Hash Algorithm: SHA-256 ▼                                  │
│ Hash caching: ● Enabled ○ Disabled                        │
│ Cache size: 500 MB                                         │
│                                                             │
│ Progress Display:                                           │
│ ☑️ Show detailed progress                                   │
│ ☑️ Show files per second                                    │
│ ☑️ Show estimated time remaining                            │
│ ☑️ Show current file being processed                       │
│                                                             │
│ [Apply] [Test Performance] [Reset]                         │
└─────────────────────────────────────────────────────────────┘
```

### Display Settings

```
Display Settings
┌─────────────────────────────────────────────────────────────┐
│ Results View:                                               │
│ Default grouping: By Hash ▼                                │
│ Default sorting: By Size (Largest First) ▼                 │
│ Items per page: 100 ▼                                      │
│                                                             │
│ File Information:                                           │
│ ☑️ Show file sizes in human-readable format                │
│ ☑️ Show modification dates                                  │
│ ☑️ Show file paths                                          │
│ ☑️ Show file type icons                                     │
│                                                             │
│ Thumbnails:                                                 │
│ ☑️ Enable thumbnails for images                             │
│ ☑️ Enable thumbnails for videos                             │
│ Thumbnail size: Medium ▼                                   │
│ Cache size: 100 MB                                         │
│                                                             │
│ Colors and Highlighting:                                    │
│ ☑️ Highlight recommended files                              │
│ ☑️ Use colors for different file types                     │
│ ☑️ Show selection status with colors                       │
│                                                             │
│ [Apply] [Preview] [Reset]                                  │
└─────────────────────────────────────────────────────────────┘
```

### Advanced Settings

```
Advanced Settings
┌─────────────────────────────────────────────────────────────┐
│ Logging:                                                    │
│ Log level: Info ▼                                          │
│ Log file size: 10 MB                                       │
│ Keep logs for: 30 days                                     │
│ ☑️ Log file operations                                      │
│ ☑️ Log scan operations                                      │
│ ☑️ Log errors and warnings                                  │
│                                                             │
│ Database:                                                   │
│ Database location: [~/.local/share/cloneclean/] [Browse]    │
│ ☑️ Auto-vacuum database                                     │
│ ☑️ Compress database                                        │
│ Backup frequency: Weekly ▼                                 │
│                                                             │
│ Network:                                                    │
│ ☑️ Check for updates                                        │
│ Update channel: Stable ▼                                   │
│ Proxy settings: [Configure...]                             │
│                                                             │
│ Experimental:                                               │
│ ☐ Enable experimental features                             │
│ ☐ Detailed performance monitoring                          │
│ ☐ Advanced debugging                                       │
│                                                             │
│ [Apply] [Export Settings] [Import Settings] [Reset All]   │
└─────────────────────────────────────────────────────────────┘
```

---

## Safety Preferences

### Safety Levels

CloneClean offers different safety levels:

#### Maximum Safety (Default)
- All operations require confirmation
- Automatic backups for all operations
- Protected paths strictly enforced
- Detailed operation logging
- Conservative recommendations

#### Balanced Safety
- Confirmations for major operations only
- Automatic backups for delete operations
- Protected paths enforced with warnings
- Standard operation logging
- Balanced recommendations

#### Minimal Safety (Advanced Users)
- Confirmations for bulk operations only
- Backups on request only
- Protected paths as warnings only
- Basic operation logging
- Aggressive recommendations

### Confirmation Settings

```
Confirmation Settings
┌─────────────────────────────────────────────────────────────┐
│ Require confirmation for:                                   │
│ ☑️ File deletion (individual files)                         │
│ ☑️ Bulk file deletion (multiple files)                     │
│ ☑️ File moves                                               │
│ ☑️ Operations on large files (>100 MB)                     │
│ ☑️ Operations on many files (>50 files)                    │
│ ☑️ Operations on protected file types                       │
│                                                             │
│ Confirmation Style:                                         │
│ ● Detailed dialog with file list                           │
│ ○ Simple yes/no confirmation                               │
│ ○ Checkbox confirmation ("I understand")                   │
│                                                             │
│ Timeout:                                                    │
│ ☑️ Auto-cancel after 60 seconds of inactivity              │
│                                                             │
│ [Apply] [Test Confirmations] [Reset]                       │
└─────────────────────────────────────────────────────────────┘
```

### Emergency Settings

```
Emergency Settings
┌─────────────────────────────────────────────────────────────┐
│ Emergency Access:                                           │
│ ☑️ Enable emergency restore mode                            │
│ ☑️ Allow restore without main database                      │
│ ☑️ Create emergency backup before major operations          │
│                                                             │
│ Emergency Contacts:                                         │
│ Backup location: [External drive or network location]      │
│ Recovery email: [user@example.com                    ]     │
│                                                             │
│ Panic Mode:                                                 │
│ ☑️ Enable panic button (stops all operations)              │
│ ☑️ Auto-backup on panic                                     │
│ ☑️ Send emergency notification                              │
│                                                             │
│ Recovery Options:                                           │
│ ☑️ Keep operation logs for forensic analysis               │
│ ☑️ Create system restore point before operations           │
│ ☑️ Enable safe mode (read-only operations)                 │
│                                                             │
│ [Test Emergency Mode] [Apply] [Reset]                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Recovery Procedures

### Common Recovery Scenarios

#### Accidentally Deleted Files

1. **Check System Trash:**
   - Open file manager
   - Navigate to Trash/Recycle Bin
   - Restore files if found

2. **Use CloneClean Restore:**
   - Click "🔄 Restore" in main window
   - Find the relevant backup
   - Select files to restore
   - Choose restore location

3. **Check Operation History:**
   - View → Operation History
   - Find the delete operation
   - Use "Undo" if available

#### Corrupted Database

1. **Database Recovery:**
   ```
   Database Recovery
   ┌─────────────────────────────────────────────────────────┐
   │ ⚠️ Database corruption detected                         │
   │                                                         │
   │ CloneClean has detected issues with the main database.  │
   │ This may cause scan results to be incomplete or        │
   │ incorrect.                                              │
   │                                                         │
   │ Recovery Options:                                       │
   │ ● Automatic repair (recommended)                       │
   │ ○ Restore from backup                                  │
   │ ○ Rebuild database from scratch                        │
   │ ○ Continue with current database (not recommended)     │
   │                                                         │
   │ [Start Recovery] [View Details] [Cancel]               │
   └─────────────────────────────────────────────────────────┘
   ```

2. **Manual Recovery:**
   - Close CloneClean
   - Navigate to `~/.local/share/cloneclean/`
   - Rename `database.db` to `database.db.backup`
   - Restart CloneClean (creates new database)
   - Import settings and scan history if needed

#### Lost Configuration

1. **Settings Recovery:**
   - File → Import Settings
   - Browse to backup location
   - Select settings file
   - Restart application

2. **Reset to Defaults:**
   - Settings → Advanced → Reset All
   - Confirm reset operation
   - Reconfigure important settings

#### System Crash During Operation

1. **Automatic Recovery:**
   - CloneClean detects incomplete operations on startup
   - Offers to complete, rollback, or ignore
   - Shows recovery dialog with options

2. **Manual Recovery:**
   - Check operation logs in `~/.local/share/cloneclean/logs/`
   - Review backup directory for recent backups
   - Use restore function to recover files if needed

### Emergency Procedures

#### Panic Button
If something goes wrong during an operation:

1. **Press Escape key** or click "Emergency Stop"
2. **Confirm emergency stop** in dialog
3. **Review partial results** in emergency dialog
4. **Choose recovery action:**
   - Rollback partial changes
   - Complete operation manually
   - Restore from backup

#### Safe Mode
If CloneClean is behaving unexpectedly:

1. **Start in safe mode:** `cloneclean --safe-mode`
2. **Limited functionality:** Read-only operations only
3. **Diagnostic tools:** Access to logs and diagnostics
4. **Recovery options:** Backup and restore functions available

---

## Next Steps

With safety features configured:

1. **Learn keyboard shortcuts** in Volume 5 for efficient operation
2. **Review troubleshooting guide** in Volume 6 for common issues
3. **Test backup and restore** with non-critical files
4. **Configure protection rules** for your important directories
5. **Set up regular backup schedule** for peace of mind

### Safety Best Practices

1. **Test restore procedures** regularly with sample files
2. **Keep backups on separate storage** (external drive, cloud)
3. **Review protection rules** periodically
4. **Monitor backup space usage** to prevent storage issues
5. **Document your safety configuration** for future reference

---

**Ready for shortcuts?** Continue to **Volume 5: Keyboard Shortcuts & Tips** to learn efficient operation techniques.

---

*Volume 4 Complete - Continue to Volume 5 for keyboard shortcuts and power user tips.*