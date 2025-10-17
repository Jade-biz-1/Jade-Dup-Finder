# Volume 6: Troubleshooting & FAQ

**DupFinder User Guide - Volume 6**  
**Last Updated:** October 17, 2025

---

## Table of Contents

1. [Common Issues and Solutions](#common-issues-and-solutions)
2. [Performance Problems](#performance-problems)
3. [Scan Issues](#scan-issues)
4. [File Operation Problems](#file-operation-problems)
5. [Interface Issues](#interface-issues)
6. [Frequently Asked Questions](#frequently-asked-questions)
7. [Getting Help](#getting-help)

---

## Common Issues and Solutions

### Application Won't Start

#### Symptom: DupFinder crashes on startup
**Possible Causes:**
- Corrupted configuration files
- Missing system dependencies
- Insufficient permissions

**Solutions:**
1. **Reset configuration:**
   ```bash
   # Linux/macOS
   rm -rf ~/.config/DupFinder/
   rm -rf ~/.local/share/dupfinder/
   
   # Windows
   # Delete: %APPDATA%\DupFinder\
   # Delete: %LOCALAPPDATA%\dupfinder\
   ```

2. **Check system requirements:**
   - Qt 5.15+ or Qt 6.2+
   - 4GB RAM minimum
   - 100MB free disk space

3. **Run with debug output:**
   ```bash
   dupfinder --debug --log-level=debug
   ```

#### Symptom: "Cannot find Qt libraries" error
**Solution:**
```bash
# Linux - Install Qt development packages
sudo apt install qt5-default libqt5widgets5-dev

# macOS - Install via Homebrew
brew install qt5

# Windows - Ensure Qt DLLs are in PATH or application directory
```

### Database Issues

#### Symptom: "Database corruption detected"
**Solutions:**
1. **Automatic repair:**
   - DupFinder will offer automatic repair on startup
   - Choose "Automatic repair (recommended)"
   - Wait for repair to complete

2. **Manual database reset:**
   ```bash
   # Backup current database
   cp ~/.local/share/dupfinder/database.db ~/.local/share/dupfinder/database.db.backup
   
   # Remove corrupted database
   rm ~/.local/share/dupfinder/database.db
   
   # Restart DupFinder (creates new database)
   ```

3. **Restore from backup:**
   - Use File → Restore from Backup
   - Select most recent backup
   - Follow restore wizard

#### Symptom: Scan history disappeared
**Solution:**
```bash
# Check if history files exist
ls ~/.local/share/dupfinder/history/

# If files exist but not showing, reset database
rm ~/.local/share/dupfinder/database.db

# Restart application to rebuild database
```

---

## Performance Problems

### Slow Scanning

#### Symptom: Scans take much longer than expected
**Diagnosis:**
1. **Check system resources:**
   - Open system monitor
   - Look for high CPU, memory, or disk usage
   - Check if other applications are competing for resources

2. **Review scan configuration:**
   - Large number of files being scanned
   - Network drives included in scan
   - Very small minimum file size (< 1MB)

**Solutions:**
1. **Optimize scan settings:**
   ```
   Recommended Settings:
   - Thread Count: Number of CPU cores
   - Memory Limit: 2-4 GB
   - Minimum File Size: 1 MB or higher
   - Hash Algorithm: SHA-256
   ```

2. **Exclude unnecessary locations:**
   - System directories (/bin, /usr, /etc)
   - Version control directories (.git, .svn)
   - Build directories (build/, node_modules/)
   - Cache directories

3. **Use incremental scanning:**
   - Scan smaller directories first
   - Use size-based filtering
   - Process one file type at a time

#### Symptom: High memory usage during scan
**Solutions:**
1. **Reduce memory limit:**
   - Settings → Scanning → Memory Limit
   - Set to 50% of available RAM
   - Restart scan

2. **Increase virtual memory:**
   ```bash
   # Linux - Check swap space
   free -h
   
   # Add swap if needed
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **Scan in smaller chunks:**
   - Divide large directories into smaller scans
   - Use exclude patterns to limit scope
   - Process results between scans

### Slow File Operations

#### Symptom: Deleting/moving files is very slow
**Causes:**
- Large number of files selected
- Files on slow storage (network drives, USB)
- Antivirus software interference
- Insufficient disk space

**Solutions:**
1. **Batch operations:**
   - Process files in smaller batches (50-100 files)
   - Use operation queue to manage large operations
   - Monitor progress and pause if needed

2. **Storage optimization:**
   - Ensure sufficient free space (20% minimum)
   - Defragment hard drives (Windows)
   - Check disk health with SMART tools

3. **Antivirus exclusions:**
   - Add DupFinder to antivirus exclusions
   - Temporarily disable real-time scanning
   - Use antivirus "gaming mode" during operations

---

## Scan Issues

### No Duplicates Found

#### Symptom: Scan completes but finds no duplicates
**Possible Causes:**
- Minimum file size too high
- Scan locations don't contain duplicates
- Files are similar but not identical
- Exclude patterns too aggressive

**Solutions:**
1. **Adjust scan parameters:**
   - Lower minimum file size to 100KB or 1MB
   - Check detection mode (use "Exact" for true duplicates)
   - Review exclude patterns

2. **Verify scan locations:**
   - Ensure folders contain files
   - Check folder permissions
   - Look for hidden files if needed

3. **Test with known duplicates:**
   - Copy a file to create a duplicate
   - Run scan on that location
   - Verify detection works

#### Symptom: Scan finds too many false positives
**Solutions:**
1. **Use stricter detection:**
   - Switch to "Exact" detection mode
   - Increase minimum file size
   - Use content-based detection only

2. **Review file types:**
   - Some file types (like empty files) may appear as duplicates
   - Exclude file types that commonly have identical content
   - Focus on specific file types (images, documents, videos)

### Scan Errors

#### Symptom: "Permission denied" errors during scan
**Solutions:**
1. **Run with appropriate permissions:**
   ```bash
   # Linux/macOS - for system directories
   sudo dupfinder
   
   # Better: Change ownership of user directories
   sudo chown -R $USER:$USER ~/Documents ~/Pictures
   ```

2. **Skip protected directories:**
   - Add system directories to exclude patterns
   - Focus on user-accessible locations
   - Use preset scans that avoid system areas

#### Symptom: "File not found" errors
**Causes:**
- Files deleted during scan
- Network connectivity issues
- Symbolic links to non-existent files

**Solutions:**
1. **Enable error handling:**
   - Settings → Scanning → Continue on errors
   - Review error log after scan
   - Skip problematic files

2. **Network drive issues:**
   - Ensure stable network connection
   - Avoid scanning during network maintenance
   - Consider copying files locally first

---

## File Operation Problems

### Cannot Delete Files

#### Symptom: "Cannot delete file" error
**Common Causes:**
- File is in use by another application
- Insufficient permissions
- File is read-only or system-protected
- Antivirus software blocking deletion

**Solutions:**
1. **Close applications:**
   - Close all applications that might be using the files
   - Check system tray for background applications
   - Restart computer if necessary

2. **Check file properties:**
   ```bash
   # Linux - Check file permissions
   ls -la filename
   
   # Change permissions if needed
   chmod 644 filename
   
   # Windows - Check file attributes
   attrib filename
   
   # Remove read-only attribute
   attrib -r filename
   ```

3. **Use safe mode:**
   - Boot into safe mode (Windows)
   - Use single-user mode (macOS/Linux)
   - Delete files without interference

#### Symptom: Files deleted but still appear in results
**Solutions:**
1. **Refresh results:**
   - Press F5 or click Refresh
   - Close and reopen results window
   - Restart application if needed

2. **Clear cache:**
   - Settings → Advanced → Clear Cache
   - Restart application
   - Re-run scan if necessary

### Backup and Restore Issues

#### Symptom: "Backup failed" error
**Causes:**
- Insufficient disk space for backup
- Backup location not accessible
- Permissions issues

**Solutions:**
1. **Check backup location:**
   - Verify backup directory exists
   - Ensure sufficient free space
   - Test write permissions

2. **Change backup location:**
   - Settings → Safety → Backup Location
   - Choose location with more space
   - Use external drive if needed

#### Symptom: Cannot restore files
**Solutions:**
1. **Verify backup integrity:**
   - Check backup files exist
   - Verify backup manifest
   - Try restoring individual files

2. **Manual restore:**
   ```bash
   # Navigate to backup directory
   cd ~/.local/share/dupfinder/backups/
   
   # List available backups
   ls -la
   
   # Copy files manually
   cp backup_folder/files/* /original/location/
   ```

---

## Interface Issues

### Display Problems

#### Symptom: Interface appears corrupted or garbled
**Solutions:**
1. **Reset interface settings:**
   - Settings → General → Reset Interface
   - Restart application
   - Reconfigure preferences

2. **Update graphics drivers:**
   - Download latest drivers for your graphics card
   - Restart computer after installation
   - Test with different Qt themes

3. **Check display scaling:**
   - Adjust system display scaling
   - Use integer scaling factors (100%, 200%)
   - Restart application after changes

#### Symptom: Thumbnails not showing
**Solutions:**
1. **Enable thumbnail generation:**
   - Settings → Display → Enable Thumbnails
   - Increase thumbnail cache size
   - Clear thumbnail cache and regenerate

2. **Check file types:**
   - Verify file types support thumbnails
   - Install additional image codecs if needed
   - Test with common image formats (JPEG, PNG)

### Responsiveness Issues

#### Symptom: Interface freezes during operations
**Solutions:**
1. **Increase operation timeout:**
   - Settings → Advanced → Operation Timeout
   - Set to higher value (300 seconds)
   - Enable operation progress display

2. **Use background processing:**
   - Enable background operations
   - Reduce UI update frequency
   - Close other applications to free resources

---

## Frequently Asked Questions

### General Questions

**Q: Is DupFinder safe to use?**
A: Yes, DupFinder includes multiple safety features:
- Files are moved to trash, not permanently deleted
- Automatic backups before operations
- Protected paths prevent system file deletion
- Undo functionality for most operations

**Q: How accurate is duplicate detection?**
A: DupFinder uses cryptographic hashing (SHA-256) for exact duplicate detection, which is 99.999% accurate. Similar file detection uses fuzzy matching and may have false positives.

**Q: Can I recover deleted files?**
A: Yes, through multiple methods:
- Check system trash/recycle bin
- Use DupFinder's restore function
- Restore from automatic backups
- Use system file recovery tools

**Q: Does DupFinder work with network drives?**
A: Yes, but performance may be slower. For best results:
- Ensure stable network connection
- Consider copying files locally first
- Use wired connection instead of WiFi

### Technical Questions

**Q: What file systems are supported?**
A: DupFinder works with all major file systems:
- Windows: NTFS, FAT32, exFAT
- macOS: APFS, HFS+, FAT32
- Linux: ext4, ext3, XFS, Btrfs, FAT32

**Q: How much disk space do I need?**
A: Requirements depend on usage:
- Application: ~100 MB
- Database: 1-10 MB per 100,000 files
- Backups: Up to 10 GB (configurable)
- Thumbnails: 1-100 MB (configurable)

**Q: Can I run multiple scans simultaneously?**
A: No, DupFinder runs one scan at a time to prevent resource conflicts and ensure accuracy. You can queue multiple scans using presets.

**Q: What happens to symbolic links?**
A: DupFinder follows symbolic links by default but can be configured to:
- Skip symbolic links entirely
- Treat links as separate files
- Follow links but avoid infinite loops

### Performance Questions

**Q: Why is my first scan slower than subsequent ones?**
A: First scans are slower because:
- File system cache is empty
- Hash cache is being built
- Thumbnails are being generated
- Database is being populated

**Q: How can I speed up scans?**
A: Several optimization strategies:
- Use SSD storage instead of HDD
- Increase memory allocation
- Use more CPU threads
- Exclude unnecessary directories
- Set higher minimum file size

**Q: Should I scan my entire system?**
A: Generally not recommended because:
- System files rarely have duplicates
- Scan takes much longer
- Higher risk of false positives
- Focus on user data directories instead

---

## Getting Help

### Built-in Help Resources

1. **Help Button (F1):**
   - Quick start guide
   - Keyboard shortcuts
   - Basic troubleshooting

2. **Tooltips:**
   - Hover over buttons and controls
   - Context-sensitive help
   - Feature explanations

3. **Status Messages:**
   - Bottom status bar
   - Progress dialogs
   - Error notifications

### Log Files and Debugging

#### Accessing Log Files
```bash
# Linux/macOS
~/.local/share/dupfinder/logs/

# Windows
%LOCALAPPDATA%\dupfinder\logs\
```

#### Log File Types
- **application.log** - General application events
- **scan.log** - Scan operations and results
- **operations.log** - File operations (delete, move)
- **errors.log** - Error messages and stack traces

#### Enabling Debug Mode
```bash
# Command line debug mode
dupfinder --debug --log-level=debug

# Or set in Settings → Advanced → Logging
```

### Reporting Issues

#### Before Reporting
1. **Check this troubleshooting guide**
2. **Search existing issues** on project repository
3. **Try basic solutions** (restart, reset settings)
4. **Gather system information**

#### Information to Include
- **System details:** OS, version, architecture
- **DupFinder version:** Help → About
- **Steps to reproduce:** Detailed sequence
- **Expected vs actual behavior**
- **Log files:** Relevant portions
- **Screenshots:** If interface issue

#### Where to Report
- **GitHub Issues:** For bugs and feature requests
- **Community Forum:** For usage questions
- **Email Support:** For sensitive issues

### Community Resources

#### Documentation
- **User Guide:** Complete feature documentation
- **API Documentation:** For developers
- **Video Tutorials:** Step-by-step guides
- **FAQ Database:** Searchable knowledge base

#### Community Support
- **User Forum:** Community discussions
- **Discord/Slack:** Real-time chat support
- **Reddit Community:** Tips and tricks
- **Stack Overflow:** Technical questions

---

## Emergency Procedures

### Data Recovery Emergency

If you accidentally deleted important files:

1. **Stop immediately** - Don't write new data to disk
2. **Check trash/recycle bin** first
3. **Use DupFinder restore** if available
4. **Try system file recovery tools**
5. **Contact data recovery service** if critical

### Application Recovery

If DupFinder becomes unusable:

1. **Backup current data:**
   ```bash
   cp -r ~/.local/share/dupfinder/ ~/dupfinder_backup/
   ```

2. **Reset to defaults:**
   ```bash
   rm -rf ~/.config/DupFinder/
   ```

3. **Reinstall application** if necessary

4. **Restore data from backup** after reinstall

---

**Still need help?** Check the project documentation at [dupfinder.org](https://dupfinder.org) or contact support through the official channels.

---

*Volume 6 Complete - You now have comprehensive knowledge of DupFinder! Return to any volume as needed for reference.*