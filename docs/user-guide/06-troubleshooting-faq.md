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

---

## Perfems

### Slow Scanning

#### Symptom: Scans take much longer than expected
**Solutions:**
*
   - Increase minimum file size to 1MB+
   ctories
ds
   cation

2. **Check system resou

   - Ensure suffiRAM
pace
   - Monitor CPU usage

3. **Network drive considerati*
   - Avoid scanning netws
   - Use wired connection instead of WiFi
   - Consider copying files locally first

### High Memory Usage

#### Symptom: Application uses exces
**Solutions:**
**
   - Settings mit
   - Set to 50% of available RM
   - Rt scan

2. **Scan in smaller chunks:**
   - Divide large directories
   - Use exclude patterns
   - Process results between



## Scan Issues

### No Duplicates Found

ates
**Solutions:**
1. **Adjust scan parameters:**
   - Lower minimum file size
   - Check detection mode settings

   - Verify scan locations contain files

2. **Test with known duplicates:**
   - Create test duplicate files
   - Run scan on test location
   - Verify detecks

### Permission Errors

#### Symptom: "Permission den
**Solution
1. 
   ```bash
   # For system directories (use ly)
   sudo dupfinder
   ```

2. **Ss:**
ns
   - Focus on user-accessible s
   - Use preset scans

---

## File Operation Problems

### Cannot Delete Files

#### Symptom: "Cannot delete fil" error
**Solutions:**
1. **Close applications using fil
   - Close all applicatio
ay
   - Restart ceeded

2. **Check file permissions:**
   ```bash
   # Linux - Check and fix permissions

   chmod 644 filename
   
   # Windows - Remove read-only attrie
   attrib -r filename
  ```

### Backup and Restore Issues

#### Symptom: "Backup failed" error
ons:**
1. 
ory exists
   - Ensure suce
missions


   - Settings → Safety → Backup Location
   - Choose locationace
   - Use external drive if needed

---

e Issues

### Display Problems

#### Symptom: Interface appears corrupted
**Solutions:**
ngs:**
   - Settings → General → Resterface
   - Restart application
   - Update graphics drivers


   - Use integer scaling (100%, 20
   - Restart after changes

### Thumbnails Not Showing

#### Symptom: Image thumbnails don't appear
**Solutions:**
1. **Enable thumbnails:**
   - Settings → Display → Enable Thumls
   - Increase cache size
   - Clear and regenerate cache

2. **Check file types:**
   - Verify supported formats
   - Install additional codecs
   - Test with common formats (JPEG, PNG)

---

## Frequently Asked Questions

### General Questions

**Q: Is DupFinder safe to use?**
A: Yes, DupFinders:
- Fdeleted
- Automatic backups before operations
- Protected paths prevent system file deletion
- Undo

**Q: How accurate is duplicate det
A: Very accurate - uses SHA-256 cryptographic hliable.

**Q: Can I recover deleted files?**
:
- System trash/recycle bin
- DupFinderon
- Automatic backups
- System file recovery tools

es?**
A: Yes, but pe
- Ensure stable connection
- Consider copying files locally first
- Use wired instead of WiFi

s

**Q: What file systems are supported?**
A: All major file systems:
- Windows: NTFS, FAT32, exFAT
32
- L

**Q: How much disk space i?**

- Application: ~100 MB
es
- Backups: Up to 10 GB (configurable)
- Thumbnails: 1-10)

**Q: Can I run multiple scsly?**
A: No, one scan at a time to prevent co

estions

**Q: Why is my first scan 
A: First scans are slower because:
- File system cache is empty
- Hash cache being built
ed
- Database being populated

**Q: How can I speed up scans?**
A: Several strateg:
- UDD
- Increase memory allocation
- Use more CPU thread
- Eories
- Set higher minimum file size

---

## Getting Help

esources

1. **Help Button (F1):**
   - Quick start guide
   - Keyboard shortcuts
ng

2. **Tooltips:
   - Hover over controls
   - Context-sensitive help
   - Feature explanations

s:**
   - Bottom status bar
   - Progress dialogs
   - Error notifications

 Files

g Logs
```bash
# Linux/macS
~/.local/share/dupfinder/logs/

# Windows
\
```

#### Log Types
- **application.log** - General events
- **scan.log** - Scan operans

- **errors.log** - Error messa

#### Debug Mode
```bash
g
dupfinder --debug --log-level=debu
```

### Reporting Issues

#### Before Reporting
ide
2. Search existing issues
3. Try bas
4. Gather system information

###clude
- System details (OS, versi
- DupFind version
- Suce
- Expected vs actual behior
- Relevant log files
- Screlicable

###rces

- **Documentation:*uides

- **GitHub:** Bug reres
ssues

---

## Emergency Procedures

### Data Recovery Emergency

If important files were acciden

1. **Stop immediately** - Don't write neta
2. **Check trash/recycle bin** firt
ilable
4. **Try system recovery tools**
5. **Contact data recovery service

### Application Recovery

If DupFinder becomes unusable:

1. **Backup current data:**
   ```bash
   cp -r ~/.local/share/dupfinder/ckup/
   ```

2. **Reset to defaults:**
   ```bash
   rm -rf ~/.config/DupFinder/
   ```

3. **Reinstall if necessa*
**

---

**Need more help?** Check the project documeels.

---

*Volume 6 Complete - You now havereference.*or s needed folume ao any vturn tnder! Ref DupFiledge oknowrehensive  comp