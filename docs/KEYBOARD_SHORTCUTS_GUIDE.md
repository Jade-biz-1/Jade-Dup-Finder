# Keyboard Shortcuts Guide

**Feature:** T19 - Add Keyboard Shortcuts  
**Status:** ✅ Complete  
**Date:** October 16, 2025

## Overview

DupFinder provides comprehensive keyboard shortcuts for efficient navigation and operation. This guide covers all available shortcuts across the main window and results window, enabling power users to work quickly without relying on mouse interactions.

## User Story Fulfilled

### US-11.4: Keyboard Shortcuts
*"As a user, I want to see keyboard shortcuts"*

**Implementation:**
- 20+ keyboard shortcuts across main and results windows
- Standard shortcuts following platform conventions
- Context-sensitive shortcuts for different windows
- Comprehensive help documentation with shortcut listings

## Main Window Shortcuts

### File Operations
| Shortcut | Action | Description |
|----------|--------|-------------|
| **Ctrl+N** | New Scan | Open scan configuration dialog |
| **Ctrl+O** | View History | Open scan history dialog |
| **Ctrl+S** | Export Results | Export results (when results window is open) |
| **Ctrl+Z** | Undo/Restore | Open restore dialog for file recovery |

### Application Control
| Shortcut | Action | Description |
|----------|--------|-------------|
| **Ctrl+,** | Settings | Open application settings |
| **Ctrl+Shift+S** | Safety Features | Open safety configuration dialog |
| **Ctrl+Q** | Quit | Exit application |
| **F1** | Help | Show help dialog with shortcuts |

### Navigation & Refresh
| Shortcut | Action | Description |
|----------|--------|-------------|
| **F5** | Refresh Stats | Update system statistics |
| **Ctrl+R** | Refresh Stats | Alternative refresh shortcut |
| **Escape** | Cancel/Close | Cancel operation or close active dialog |

### Quick Scan Presets
| Shortcut | Action | Description |
|----------|--------|-------------|
| **Ctrl+1** | Quick Scan | Scan common locations |
| **Ctrl+2** | Downloads Cleanup | Scan Downloads folder |
| **Ctrl+3** | Photo Cleanup | Scan Pictures folder |
| **Ctrl+4** | Documents Scan | Scan Documents folder |
| **Ctrl+5** | Full System Scan | Comprehensive system scan |
| **Ctrl+6** | Custom Scan | Open custom scan configuration |

## Results Window Shortcuts

### Selection Operations
| Shortcut | Action | Description |
|----------|--------|-------------|
| **Ctrl+A** | Select All | Select all duplicate files |
| **Ctrl+D** | Clear Selection | Deselect all files |
| **Ctrl+I** | Invert Selection | Invert current selection |
| **Ctrl+Z** | Undo Selection | Undo last selection change |
| **Ctrl+Y** | Redo Selection | Redo last undone selection |

### File Operations
| Shortcut | Action | Description |
|----------|--------|-------------|
| **Delete** | Delete Files | Delete selected duplicate files |
| **Ctrl+C** | Copy Path | Copy selected file path to clipboard |
| **Space** | Preview File | Preview selected file |
| **Enter** | Open Location | Open file location in system explorer |

### View & Navigation
| Shortcut | Action | Description |
|----------|--------|-------------|
| **Ctrl+F** | Find/Search | Focus search filter box |
| **Ctrl+S** | Export Results | Export current results to file |
| **Ctrl+R** | Refresh Results | Refresh results display |
| **F5** | Refresh Results | Alternative refresh shortcut |
| **Escape** | Clear Filters/Close | Clear search filters or close window |

## Advanced Shortcuts

### Future Enhancements (Planned)
| Shortcut | Action | Description |
|----------|--------|-------------|
| **Ctrl+Shift+F** | Advanced Filter | Open advanced filtering dialog |
| **Ctrl+G** | Grouping Options | Configure result grouping |
| **Ctrl+T** | Toggle Thumbnails | Show/hide file thumbnails |

## Platform-Specific Variations

### Windows/Linux
- **Ctrl+,** for Settings (standard on Linux)
- **Ctrl+Q** for Quit
- **F1** for Help

### macOS (Future Support)
- **Cmd+,** for Preferences
- **Cmd+Q** for Quit
- **Cmd+?** for Help

## Context-Sensitive Behavior

### Smart Escape Key
The Escape key behaves intelligently based on context:
1. **Main Window:** Cancels active scan or closes progress dialog
2. **Results Window:** Clears search filters first, then closes window
3. **Dialogs:** Closes the dialog (standard behavior)

### Conditional Shortcuts
Some shortcuts are only active when relevant:
- **Ctrl+S** in main window only works when results window is open
- **Delete** key only works when files are selected in results
- **Space** preview only works when a single file is selected

## Accessibility Features

### Keyboard Navigation
- **Tab** and **Shift+Tab** navigate between controls
- **Arrow keys** navigate within lists and trees
- **Enter** activates focused buttons
- **Space** toggles checkboxes and buttons

### Screen Reader Support
- All shortcuts have accessible names
- Tooltip descriptions include shortcut information
- Status messages announce shortcut actions

## Implementation Details

### Technical Architecture
- **QShortcut** objects for reliable cross-platform shortcuts
- **Context-aware activation** prevents conflicts between windows
- **Standard Qt key sequences** for platform consistency
- **Signal/slot connections** for maintainable code

### Memory Management
- Shortcuts are automatically cleaned up with parent widgets
- No memory leaks from shortcut objects
- Efficient event handling without performance impact

## Usage Tips

### Power User Workflow
1. **Ctrl+N** → Configure scan
2. **Ctrl+1-6** → Quick preset selection
3. **Escape** → Cancel if needed
4. **Ctrl+A** → Select all duplicates in results
5. **Delete** → Remove selected files
6. **Ctrl+Z** → Undo if mistake made

### Efficiency Shortcuts
- **Ctrl+F** to quickly find specific files
- **Space** to preview before deleting
- **Ctrl+C** to copy paths for external tools
- **F5** to refresh after external changes

### Safety Shortcuts
- **Ctrl+Z** for quick access to restore functionality
- **Ctrl+Shift+S** for safety configuration
- **Escape** for quick operation cancellation

## Troubleshooting

### Shortcuts Not Working
**Possible Causes:**
- Dialog has focus instead of main window
- Conflicting system shortcuts
- Application not in focus

**Solutions:**
- Click on main window to ensure focus
- Check system shortcut conflicts
- Restart application if needed

### Platform Differences
**Windows/Linux:**
- Use Ctrl key for shortcuts
- F1 opens help dialog
- Standard Windows conventions

**Future macOS Support:**
- Cmd key instead of Ctrl
- Different help key (Cmd+?)
- macOS-specific conventions

## Customization (Future Enhancement)

### Planned Features
- **Custom shortcut configuration** in settings
- **User-defined shortcuts** for frequent actions
- **Shortcut conflict detection** and resolution
- **Import/export** shortcut configurations

### Current Limitations
- Shortcuts are currently hardcoded
- No user customization available
- Platform-specific shortcuts not yet implemented

## Related Documentation

### See Also
- **Help Dialog:** In-app shortcut reference (F1)
- **Settings Dialog:** Application configuration
- **Safety Features:** File protection settings
- **Results Window:** Duplicate management interface

### Integration Points
- **Main Window:** Primary shortcut hub
- **Results Window:** File operation shortcuts
- **Dialogs:** Standard dialog shortcuts (Escape, Enter)
- **System Integration:** Platform-specific behaviors

---

**Last Updated:** October 16, 2025  
**Version:** 1.0  
**Related:** US-11.4, Main Window, Results Window, User Experience