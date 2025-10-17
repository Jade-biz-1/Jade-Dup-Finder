# Volume 5: Keyboard Shortcuts & Tips

**DupFinder User Guide - Volume 5**  
**Last Updated:** October 17, 2025

---

## Table of Contents

1. [Complete Keyboard Shortcuts Reference](#complete-keyboard-shortcuts-reference)
2. [Power User Tips and Tricks](#power-user-tips-and-tricks)
3. [Workflow Optimization](#workflow-optimization)
4. [Performance Recommendations](#performance-recommendations)
5. [Advanced Techniques](#advanced-techniques)
6. [Customization Options](#customization-options)

---

## Complete Keyboard Shortcuts Reference

### Global Shortcuts (Available Everywhere)

| Shortcut | Action | Description |
|----------|--------|-------------|
| **F1** | Help | Open built-in help system |
| **Ctrl+N** | New Scan | Open scan configuration dialog |
| **Ctrl+O** | Open Results | Open existing scan results |
| **Ctrl+S** | Save/Export | Export current results |
| **Ctrl+Q** | Quit | Exit DupFinder |
| **Ctrl+,** | Settings | Open settings dialog |
| **F11** | Full Screen | Toggle full screen mode |
| **Ctrl+Shift+R** | Restore | Open backup restore dialog |

### Main Dashboard Shortcuts

| Shortcut | Action | Description |
|----------|--------|-------------|
| **1-6** | Quick Actions | Activate Quick Action buttons 1-6 |
| **Ctrl+1** | Quick Scan | Start Quick Scan preset |
| **Ctrl+2** | Downloads | Start Downloads cleanup |
| **Ctrl+3** | Photos | Start Photo cleanup |
| **Ctrl+4** | Documents | Start Document cleanup |
| **Ctrl+5** | Full System | Start Full System scan |
| **Ctrl+6** | Custom Scan | Open custom scan dialog |
| **Ctrl+H** | History | Open scan history dialog |
| **Ctrl+R** | Refresh | Refresh dashboard information |

### Results Window Shortcuts

#### Navigation and Selection
| Shortcut | Action | Description |
|----------|--------|-------------|
| **↑/↓** | Navigate | Move up/down in results tree |
| **←/→** | Expand/Collapse | Expand or collapse groups |
| **Home** | First Item | Go to first item in results |
| **End** | Last Item | Go to last item in results |
| **Page Up/Down** | Page Navigation | Move one page up/down |
| **Space** | Toggle Selection | Toggle file selection checkbox |
| **Ctrl+A** | Select All | Select all files for deletion |
| **Ctrl+D** | Deselect All | Deselect all files |
| **Ctrl+I** | Invert Selection | Invert current selection |
| **Ctrl+R** | Recommended | Apply recommended selection |

#### File Operations
| Shortcut | Action | Description |
|----------|--------|-------------|
| **Delete** | Delete File | Delete selected file |
| **Shift+Delete** | Delete All Selected | Delete all selected files |
| **Ctrl+M** | Move File | Move selected file |
| **Ctrl+Shift+M** | Move Selected | Move all selected files |
| **Enter** | Preview File | Preview selected file |
| **Ctrl+Enter** | Open Location | Open file location in file manager |
| **Ctrl+C** | Copy Path | Copy file path to clipboard |
| **F2** | Rename | Rename selected file |

#### View and Display
| Shortcut | Action | Description |
|----------|--------|-------------|
| **Ctrl+F** | Find/Search | Focus search box |
| **Ctrl+G** | Grouping Options | Open grouping dialog |
| **Ctrl+T** | Toggle Thumbnails | Show/hide thumbnails |
| **Ctrl+1** | File Info Tab | Switch to File Info tab |
| **Ctrl+2** | Group Info Tab | Switch to Group Info tab |
| **Ctrl+3** | Relationships Tab | Switch to Relationships tab |
| **F5** | Refresh | Refresh results display |
| **Ctrl+Plus** | Zoom In | Increase thumbnail size |
| **Ctrl+Minus** | Zoom Out | Decrease thumbnail size |
| **Ctrl+0** | Reset Zoom | Reset thumbnail size |

---

## Power User Tips and Tricks

### Efficient Scanning Strategies

#### 1. Hierarchical Scanning Approach
Start with small, manageable scans and gradually expand:

1. **Downloads folder** (quick wins)
2. **Desktop cleanup** (visible impact)
3. **Documents folder** (organized approach)
4. **Pictures folder** (with thumbnails)
5. **Full system scan** (comprehensive)

**Benefits:**
- Build confidence with easy wins
- Learn the interface gradually
- Avoid overwhelming results
- Develop good habits

#### 2. Size-Based Strategy
Focus on largest files first for maximum impact:

- **Scan 1:** Files > 100 MB (biggest impact)
- **Scan 2:** Files 10-100 MB (good balance)
- **Scan 3:** Files 1-10 MB (thorough cleanup)
- **Scan 4:** Files < 1 MB (final polish)

#### 3. Type-Specific Campaigns
Focus on one file type at a time:

- **Week 1:** Video files (usually largest)
- **Week 2:** Photo collections
- **Week 3:** Document archives
- **Week 4:** Audio files
- **Week 5:** Software installers

### Advanced Selection Techniques

#### Pattern-Based Selection
Use Smart Selection with patterns:
- `*_copy.*` - Files with "copy" in name
- `*/Downloads/*` - All files in Downloads
- `*.tmp` - Temporary files
- `*backup*` - Backup files

#### Multi-Criteria Selection
Combine multiple criteria:
- Size > 10 MB AND Date < 30 days ago
- Type = Video AND Location = Downloads
- Name contains "duplicate" OR "copy"

### Workflow Optimization

#### The "Three-Pass Method"
1. **Pass 1:** Quick scan, delete obvious duplicates
2. **Pass 2:** Detailed review, handle edge cases
3. **Pass 3:** Final cleanup, organize remaining files

#### Batch Processing
- Group similar operations together
- Use operation queue for large batches
- Schedule intensive scans during off-hours
- Process one file type at a time

---

## Performance Recommendations

### System Optimization

#### Hardware Considerations
- **SSD vs HDD:** SSDs dramatically improve scan speed
- **RAM:** More RAM allows larger file caches
- **CPU:** Multi-core processors benefit from parallel scanning
- **Network:** Avoid scanning network drives during peak hours

#### DupFinder Settings
```
Optimal Performance Settings:
- Thread Count: CPU cores - 1
- Memory Limit: 50% of available RAM
- Hash Algorithm: SHA-256 (good balance)
- Cache Size: 25% of memory limit
- I/O Priority: Normal (unless system is idle)
```

### Scan Optimization

#### Exclude Patterns for Speed
Common exclusions to improve performance:
- `*/node_modules/*` - JavaScript dependencies
- `*/.git/*` - Git repositories
- `*/build/*` - Build artifacts
- `*/cache/*` - Cache directories
- `*/.vscode/*` - Editor files

#### Smart Folder Selection
- **Include:** User data folders (Documents, Pictures, Downloads)
- **Exclude:** System folders (/bin, /usr, /etc on Linux)
- **Consider:** External drives (slower but often have duplicates)

### Memory Management

#### Large File Handling
- Set appropriate minimum file size (1MB+)
- Use streaming for very large files
- Monitor memory usage during scans
- Consider splitting large scans into smaller chunks

---

## Advanced Techniques

### Regular Expression Patterns

#### Advanced Exclude Patterns
```
Pattern Examples:
- `.*\.(tmp|temp|cache)$` - Temporary files
- `.*_backup_\d{4}-\d{2}-\d{2}.*` - Dated backups
- `.*/\.(git|svn|hg)/.*` - Version control
- `.*\.(log|bak|old)$` - Log and backup files
```

#### Smart Selection Patterns
```
Selection Examples:
- `.*_duplicate.*` - Files with "duplicate" in name
- `.*/Downloads/.*\.(exe|msi|dmg)$` - Installers in Downloads
- `.*\.(jpg|jpeg|png|gif)$` - Image files only
- `.*_\d{4}-\d{2}-\d{2}_.*` - Files with date stamps
```

### Automation Strategies

#### Preset Workflows
Create presets for common scenarios:

**Daily Cleanup Preset:**
- Scan: Downloads, Desktop, Temp folders
- Min size: 1 MB
- Auto-select: Files older than 7 days

**Weekly Deep Clean:**
- Scan: Documents, Pictures, Videos
- Min size: 10 MB
- Smart select: Keep newest in each group

**Monthly Archive:**
- Scan: Entire user directory
- Min size: 50 MB
- Export results before cleanup

#### Scheduled Operations
While DupFinder doesn't have built-in scheduling, you can:
- Use system task scheduler
- Create batch scripts for common operations
- Set up automated exports for record-keeping

### Integration with System Tools

#### File Manager Integration
- Use "Open Location" to jump to file manager
- Copy paths for use in other tools
- Export file lists for external processing

#### Command Line Integration
Export results and process with command-line tools:
```bash
# Process exported CSV with command-line tools
grep "\.mp4$" duplicates.csv | sort -k3 -nr
```

---

## Customization Options

### Interface Customization

#### Theme and Appearance
- **Dark Mode:** Better for extended use
- **Light Mode:** Better for detailed file inspection
- **Font Size:** Adjust for screen size and vision
- **Icon Size:** Scale for high-DPI displays

#### Layout Preferences
- **Panel Arrangement:** Customize info panel position
- **Column Visibility:** Show/hide specific file information
- **Thumbnail Size:** Balance between detail and performance
- **Group Expansion:** Default expand/collapse behavior

### Behavioral Customization

#### Safety Settings
Customize safety levels based on experience:

**Beginner Settings:**
- Confirm all operations
- Automatic backups enabled
- Conservative recommendations
- Detailed progress information

**Expert Settings:**
- Minimal confirmations
- Selective backups
- Aggressive recommendations
- Streamlined interface

#### Default Preferences
Set defaults that match your workflow:
- Default scan locations
- Preferred file size limits
- Standard exclude patterns
- Export format preferences

### Keyboard Shortcut Customization

#### Custom Shortcuts
While not directly customizable in the UI, you can:
- Use system-level shortcut managers
- Create desktop shortcuts for common presets
- Use accessibility tools for custom key combinations

#### Workflow-Specific Shortcuts
Memorize shortcuts for your most common tasks:
- Photo cleanup: `Ctrl+3` → `Ctrl+T` → `Space` → `Delete`
- Quick export: `Ctrl+S` → `Enter`
- Undo selection: `Ctrl+Z`

---

## Next Steps

With shortcuts and tips mastered:

1. **Practice keyboard navigation** to build muscle memory
2. **Create custom presets** for your specific workflows
3. **Experiment with advanced patterns** for better automation
4. **Monitor performance** and adjust settings as needed
5. **Check troubleshooting guide** in Volume 6 for common issues

### Building Expertise

1. **Start with basics:** Master core shortcuts first
2. **Add complexity gradually:** Introduce advanced features over time
3. **Document your workflows:** Keep notes on what works best
4. **Share knowledge:** Help others learn efficient techniques
5. **Stay updated:** Check for new features and improvements

---

**Need help with issues?** Continue to **Volume 6: Troubleshooting & FAQ** for solutions to common problems.

---

*Volume 5 Complete - Continue to Volume 6 for troubleshooting and frequently asked questions.*