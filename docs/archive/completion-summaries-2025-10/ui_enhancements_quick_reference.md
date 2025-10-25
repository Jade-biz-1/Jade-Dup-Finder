# UIEnhancements Quick Reference Guide

Quick reference for using the UIEnhancements utility class (Section 1.5).

---

## Basic Dialog Setup

Add to your dialog constructor after `initializeUI()`:

```cpp
#include "ui_enhancements.h"

MyDialog::MyDialog(QWidget* parent) : QDialog(parent)
{
    initializeUI();
    setupConnections();
    
    // Apply standard enhancements
    UIEnhancements::setupLogicalTabOrder(this);
    UIEnhancements::setupEscapeKeyHandler(this);
    UIEnhancements::setupEnterKeyHandler(this);
    UIEnhancements::applyConsistentSpacing(this);
}
```

---

## Visual Feedback

### Hover Effects
```cpp
// Single button
UIEnhancements::addButtonHoverEffect(myButton);

// All buttons in dialog
UIEnhancements::addHoverEffectsToButtons(this);
```

### Loading Indicators
```cpp
// Show loading
UIEnhancements::showLoadingIndicator(widget, tr("Processing..."));

// Hide loading
UIEnhancements::hideLoadingIndicator(widget);
```

### Disabled States
```cpp
// Enable/disable with visual feedback
UIEnhancements::setEnabledWithFeedback(button, false);
UIEnhancements::setEnabledWithFeedback(button, true);
```

### Drag & Drop
```cpp
UIEnhancements::addDragDropFeedback(widget, tr("Drop files here"));
```

---

## User Interactions

### Tooltips
```cpp
// Auto-generate tooltips
UIEnhancements::addComprehensiveTooltips(this);

// With custom tooltips
QMap<QString, QString> tooltips;
tooltips["saveButton"] = tr("Save settings");
tooltips["cancelButton"] = tr("Cancel changes");
UIEnhancements::addComprehensiveTooltips(this, tooltips);

// Single widget tooltip
UIEnhancements::addHoverTooltip(widget, tr("Click to perform action"));
```

### Focus Indicators
```cpp
UIEnhancements::addFocusIndicators(widget);
```

---

## Text Formatting

### File Paths
```cpp
// Short path with ellipsis
QString path = UIEnhancements::formatPathWithEllipsis(fullPath, 50);

// Elide in label
UIEnhancements::elideTextInLabel(label, longPath);
```

### File Sizes
```cpp
QString size = UIEnhancements::formatFileSize(bytes);
// Examples: "1.5 MB", "2.3 GB", "512 bytes"
```

### Numbers
```cpp
QString count = UIEnhancements::formatNumber(fileCount);
// Examples: "1,234", "1.234" (locale-dependent)
```

### Dates
```cpp
QString date = UIEnhancements::formatDateTime(dateTime);
// Uses locale's default format

QString custom = UIEnhancements::formatDateTime(dateTime, "yyyy-MM-dd");
```

---

## Common Patterns

### File List Display
```cpp
for (const FileInfo& file : files) {
    QString size = UIEnhancements::formatFileSize(file.size);
    QString path = UIEnhancements::formatPathWithEllipsis(file.path, 60);
    QString date = UIEnhancements::formatDateTime(file.modified);
    
    // Add to UI...
}
```

### Operation with Feedback
```cpp
void MyClass::performOperation()
{
    UIEnhancements::showLoadingIndicator(this, tr("Processing..."));
    
    try {
        // Do work
        processFiles();
        
        QMessageBox::information(this, tr("Success"), 
            tr("Processed %1 files").arg(
                UIEnhancements::formatNumber(count)));
    } catch (...) {
        QMessageBox::critical(this, tr("Error"), tr("Operation failed"));
    }
    
    UIEnhancements::hideLoadingIndicator(this);
}
```

### Button State Management
```cpp
void updateButtonStates()
{
    bool hasSelection = !selectedItems.isEmpty();
    UIEnhancements::setEnabledWithFeedback(deleteButton, hasSelection);
    UIEnhancements::setEnabledWithFeedback(moveButton, hasSelection);
}
```

---

## Best Practices

✅ **DO:**
- Apply enhancements in constructor after UI setup
- Use formatFileSize() for all file sizes
- Use formatNumber() for all large numbers
- Add tooltips to non-obvious controls
- Use setupLogicalTabOrder() for keyboard navigation

❌ **DON'T:**
- Apply hover effects if using custom themes
- Forget to hide loading indicators
- Hardcode file size units
- Skip ESC key handler for dialogs

---

## See Also

- Full documentation: `docs/section_1_5_completion_summary.md`
- Header file: `include/ui_enhancements.h`
- Example: `src/gui/about_dialog.cpp`
