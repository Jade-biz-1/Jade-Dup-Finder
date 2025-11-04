# Preset Manager Usage Guide

## Overview
The Preset Manager allows you to save, load, and manage scan configuration presets. This makes it easy to reuse common scan configurations without having to reconfigure settings each time.

## Opening the Preset Manager

From the Scan Configuration Dialog, click the **"📋 Manage Presets"** button in the bottom button bar.

## Built-in Presets

The system comes with four built-in presets that cannot be modified or deleted:

### 🔒 Downloads
- **Target**: Downloads folder
- **Detection Mode**: Smart
- **Purpose**: Quick cleanup of duplicate downloads

### 🔒 Photos
- **Target**: Pictures folder
- **Detection Mode**: Deep (hash-based)
- **Minimum Size**: 10 KB
- **Purpose**: Find duplicate images with high accuracy

### 🔒 Documents
- **Target**: Documents folder
- **Detection Mode**: Smart
- **Purpose**: Find duplicate documents

### 🔒 Media
- **Target**: Music and Videos folders
- **Detection Mode**: Media (with metadata)
- **Minimum Size**: 100 KB
- **Purpose**: Find duplicate media files using metadata

## Creating a New Preset

1. Configure your scan settings in the Scan Configuration Dialog
2. Click **"Save as Preset"** button
3. Enter a name for your preset
4. The preset is saved and can be loaded later

**Alternative Method**:
1. Open the Preset Manager
2. Click **"+ New"** button
3. Enter a name and optional description
4. The preset is created with default settings
5. You can then load it and modify the settings

## Loading a Preset

### From Preset Manager:
1. Open the Preset Manager
2. Select a preset from the list
3. Click **"Load Preset"** button
4. The scan configuration is updated with the preset settings

### Quick Load:
- Double-click any preset in the list to load it immediately

### From Quick Presets:
- Use the quick preset buttons in the Scan Configuration Dialog (Downloads, Photos, Documents, Media)

## Editing a Preset

1. Open the Preset Manager
2. Select a user-defined preset (📝 icon)
3. Click **"✏ Edit"** button
4. Modify the description
5. Click OK to save changes

**Note**: You cannot edit built-in presets (🔒 icon). To customize a built-in preset:
1. Load the built-in preset
2. Modify the settings as desired
3. Save as a new preset with a different name

## Deleting a Preset

1. Open the Preset Manager
2. Select a user-defined preset (📝 icon)
3. Click **"🗑 Delete"** button
4. Confirm the deletion

**Note**: Built-in presets cannot be deleted.

## Viewing Preset Details

When you select a preset in the Preset Manager, the right panel shows:
- Preset name and description
- Type (Built-in or User-defined)
- Target paths
- Detection mode
- Minimum file size
- Maximum depth
- Include options (hidden files, system files, symlinks, archives)
- Exclude patterns
- Exclude folders

## Preset Storage

Presets are stored in your system's application settings and persist across sessions. They are automatically loaded when you open the Preset Manager.

**Storage Location**: 
- Linux: `~/.config/DupFinder/presets/scan/`
- Windows: Registry or `%APPDATA%/DupFinder/presets/scan/`
- macOS: `~/Library/Preferences/com.dupfinder.presets/scan/`

## Best Practices

### Naming Presets
- Use descriptive names that indicate the purpose
- Examples: "Weekly Downloads Cleanup", "Photo Library Scan", "Project Folder Check"

### Organizing Presets
- Create presets for frequently used scan configurations
- Use descriptions to document the purpose and any special settings
- Delete unused presets to keep the list manageable

### Common Use Cases

**Regular Maintenance**:
- Create a preset for weekly/monthly cleanup scans
- Include common folders that accumulate duplicates

**Project-Specific**:
- Create presets for specific project directories
- Customize exclude patterns for build artifacts, dependencies, etc.

**File Type Specific**:
- Create presets for specific file types (images, videos, documents)
- Adjust detection modes and minimum sizes accordingly

**Deep vs. Quick Scans**:
- Quick preset: Fast scans with size+name matching
- Deep preset: Thorough scans with hash-based detection

## Keyboard Shortcuts

- **Double-click**: Load selected preset
- **Delete key**: Delete selected preset (with confirmation)

## Tips

1. **Start with Built-in Presets**: Use built-in presets as templates for your custom presets
2. **Test Before Saving**: Configure and test your settings before saving as a preset
3. **Use Descriptions**: Add meaningful descriptions to help remember preset purposes
4. **Regular Review**: Periodically review and clean up unused presets
5. **Backup**: Presets are stored in application settings, which can be backed up

## Troubleshooting

### Preset Not Loading
- Ensure the target paths in the preset still exist
- Check that you have read permissions for the target directories

### Cannot Edit/Delete Preset
- Built-in presets (🔒 icon) cannot be edited or deleted
- Create a new preset based on the built-in one instead

### Preset Not Appearing
- Ensure the preset was saved successfully
- Try closing and reopening the Preset Manager
- Check application settings storage location

## API Usage (For Developers)

```cpp
// Create and save a preset
ScanSetupDialog::ScanConfiguration config;
config.targetPaths << "/path/to/scan";
config.detectionMode = ScanSetupDialog::DetectionMode::Smart;

PresetManagerDialog::PresetInfo preset;
preset.name = "My Preset";
preset.description = "Custom scan configuration";
preset.config = config;
preset.isBuiltIn = false;

PresetManagerDialog manager;
manager.savePreset(preset);

// Load a preset
PresetManagerDialog::PresetInfo loaded = manager.getPreset("My Preset");
if (!loaded.name.isEmpty()) {
    // Use the loaded configuration
    scanDialog.setConfiguration(loaded.config);
}

// Get all user presets
QList<PresetManagerDialog::PresetInfo> userPresets = manager.getUserPresets();
```

## Related Documentation
- [Scan Configuration Guide](SCAN_CONFIGURATION.md)
- [Exclude Pattern Widget Usage](EXCLUDE_PATTERN_WIDGET_USAGE.md)
- [User Guide](USER_GUIDE.md)
