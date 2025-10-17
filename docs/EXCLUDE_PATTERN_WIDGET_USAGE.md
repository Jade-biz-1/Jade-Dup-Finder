# ExcludePatternWidget Usage Guide

## Overview

The `ExcludePatternWidget` is a Qt widget that provides a user-friendly interface for managing file exclusion patterns in the Duplicate File Finder application. It allows users to add, remove, validate, and test patterns that will be used to exclude files from scanning.

## Features

- **Pattern Validation**: Real-time validation of patterns with visual feedback
- **Pattern Testing**: Test patterns against sample filenames
- **Common Patterns**: Quick access to commonly used exclusion patterns
- **Persistence**: Save and load patterns using QSettings
- **Visual Feedback**: Color-coded validation messages (green for valid, red for invalid)

## Integration

### Including in Your Dialog

```cpp
#include "exclude_pattern_widget.h"

// In your dialog class
ExcludePatternWidget* m_excludePatternWidget;

// In constructor
m_excludePatternWidget = new ExcludePatternWidget(this);
layout->addWidget(m_excludePatternWidget);
```

### Getting Patterns

```cpp
QStringList patterns = m_excludePatternWidget->getPatterns();
```

### Setting Patterns

```cpp
QStringList patterns = {"*.tmp", "*.log", "Thumbs.db"};
m_excludePatternWidget->setPatterns(patterns);
```

### Connecting to Signals

```cpp
// Listen for pattern changes
connect(m_excludePatternWidget, &ExcludePatternWidget::patternsChanged,
        this, &MyDialog::onPatternsChanged);

// Listen for individual pattern additions
connect(m_excludePatternWidget, &ExcludePatternWidget::patternAdded,
        this, &MyDialog::onPatternAdded);

// Listen for individual pattern removals
connect(m_excludePatternWidget, &ExcludePatternWidget::patternRemoved,
        this, &MyDialog::onPatternRemoved);
```

## Pattern Syntax

### Wildcard Patterns

The widget supports standard wildcard patterns:

- `*` - Matches any sequence of characters
- `?` - Matches any single character
- `.` - Matches a literal dot

Examples:
- `*.tmp` - Matches all files ending with .tmp
- `test*.txt` - Matches test1.txt, test_file.txt, etc.
- `file?.dat` - Matches file1.dat, fileA.dat, etc.

### Exact Matches

You can also specify exact filenames:
- `Thumbs.db` - Matches only files named exactly "Thumbs.db"
- `.DS_Store` - Matches only files named exactly ".DS_Store"

### Pattern Validation

The widget validates patterns to ensure they:
- Are not empty
- Contain only valid characters (alphanumeric, *, ?, ., -, _, /, \, space)
- Are valid regular expressions (if they contain regex special characters)

## Common Patterns

The widget provides quick access to commonly used exclusion patterns:

- `*.tmp` - Temporary files
- `*.log` - Log files
- `*.bak` - Backup files
- `*.cache` - Cache files
- `*.swp` - Swap files (vim)
- `Thumbs.db` - Windows thumbnail cache
- `.DS_Store` - macOS folder metadata
- `desktop.ini` - Windows folder settings
- `*.temp` - Temporary files (alternative extension)
- `~*` - Backup files (Unix convention)
- `*.old` - Old file versions

## Testing Patterns

Users can test their patterns against sample filenames:

1. Click the "Test Pattern" button
2. Enter a filename to test
3. The widget will show whether the filename matches any of the current patterns

Example:
- Patterns: `*.tmp`, `*.log`
- Test filename: `debug.log`
- Result: ✓ Filename 'debug.log' MATCHES one or more patterns and will be excluded.

## Persistence

### Saving Patterns

```cpp
// Save to default key "excludePatterns"
m_excludePatternWidget->saveToSettings();

// Save to custom key
m_excludePatternWidget->saveToSettings("myCustomKey");
```

### Loading Patterns

```cpp
// Load from default key "excludePatterns"
m_excludePatternWidget->loadFromSettings();

// Load from custom key
m_excludePatternWidget->loadFromSettings("myCustomKey");
```

## API Reference

### Public Methods

#### Pattern Management

```cpp
QStringList getPatterns() const;
void setPatterns(const QStringList& patterns);
bool addPattern(const QString& pattern);
void removePattern(const QString& pattern);
void clearPatterns();
```

#### Pattern Validation

```cpp
static bool validatePattern(const QString& pattern, QString* errorMessage = nullptr);
bool matchesAnyPattern(const QString& filename) const;
```

#### Persistence

```cpp
void loadFromSettings(const QString& settingsKey = "excludePatterns");
void saveToSettings(const QString& settingsKey = "excludePatterns");
```

### Signals

```cpp
void patternsChanged(const QStringList& patterns);
void patternAdded(const QString& pattern);
void patternRemoved(const QString& pattern);
```

## Example: Integration with ScanSetupDialog

```cpp
// In ScanSetupDialog constructor
m_excludePatternWidget = new ExcludePatternWidget(this);
m_excludePatternWidget->setPatterns(QStringList{"*.tmp", "*.log", "Thumbs.db"});

// Connect to update scan configuration
connect(m_excludePatternWidget, &ExcludePatternWidget::patternsChanged,
        this, &ScanSetupDialog::onOptionsChanged);

// In getCurrentConfiguration()
config.excludePatterns = m_excludePatternWidget->getPatterns();

// In setConfiguration()
m_excludePatternWidget->setPatterns(config.excludePatterns);
```

## UI Components

The widget consists of:

1. **Title Label**: "Exclude Patterns"
2. **Pattern List**: Displays all current patterns
3. **Input Field**: For entering new patterns
4. **Add Button**: Adds the pattern from the input field
5. **Validation Label**: Shows validation feedback
6. **Remove Button**: Removes the selected pattern
7. **Test Pattern Button**: Opens dialog to test patterns
8. **Add Common Button**: Shows menu of common patterns

## Styling

The widget uses Qt stylesheets for consistent theming:

- List widget has rounded borders and alternating row colors
- Input field highlights on focus
- Buttons have hover effects
- Validation messages are color-coded (green/red)

## Best Practices

1. **Initialize with defaults**: Always set some default patterns
2. **Save on dialog accept**: Persist patterns when user confirms
3. **Load on dialog show**: Restore patterns when dialog opens
4. **Validate before use**: Use `validatePattern()` before adding patterns programmatically
5. **Test patterns**: Encourage users to test patterns before scanning

## Troubleshooting

### Pattern not matching expected files

- Use the "Test Pattern" feature to verify pattern behavior
- Remember that patterns are case-insensitive
- Check for typos in the pattern

### Pattern validation fails

- Ensure pattern doesn't contain invalid characters
- Check that regex patterns (if used) are valid
- Avoid empty or whitespace-only patterns

### Patterns not persisting

- Ensure `saveToSettings()` is called when dialog is accepted
- Verify the settings key is consistent between save and load
- Check QSettings configuration (organization/application name)

## Requirements

- Qt 6.5+
- C++17 or later
- QSettings configured for the application

## See Also

- [ScanSetupDialog Documentation](SCAN_DIALOG_USAGE.md)
- [Pattern Matching in FileScanner](FILESCANNER_EXAMPLES.md)
- [Qt QRegularExpression Documentation](https://doc.qt.io/qt-6/qregularexpression.html)
