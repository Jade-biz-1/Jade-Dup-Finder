# UI Automation Framework

This document describes the comprehensive UI automation framework for CloneClean testing, which provides high-level automation capabilities for interacting with Qt widgets and testing user interfaces.

## Overview

The UI automation framework consists of two main components:

1. **UIAutomation** - Core automation engine for widget interaction
2. **WidgetSelector** - Advanced widget selection and identification system

The framework provides a high-level, reliable way to automate UI testing with features like:
- Robust widget finding and selection
- Mouse and keyboard interaction simulation
- Form filling and validation
- Dialog handling and navigation
- Screenshot capture and visual verification
- Synchronization and wait conditions
- Accessibility support

## Components

### 1. UIAutomation Class

The `UIAutomation` class is the main automation engine that provides methods for interacting with Qt widgets.

#### Key Features

- **Widget Finding**: Reliable widget identification using multiple strategies
- **Mouse Interactions**: Click, double-click, right-click, hover, drag operations
- **Keyboard Interactions**: Text typing, key presses, keyboard shortcuts
- **Form Automation**: Automated form filling and validation
- **Dialog Handling**: Modal dialog interaction and management
- **Navigation**: Menu navigation, tab switching, tree expansion
- **Synchronization**: Wait conditions and timing control
- **Validation**: Widget state and content verification
- **Screenshots**: Widget and screen capture capabilities

#### Basic Usage

```cpp
#include "ui_automation.h"

// Create UI automation instance
UIAutomation uiAutomation;

// Configure automation
uiAutomation.setDefaultTimeout(5000);
uiAutomation.enableDetailedLogging(true);

// Find and click a button
auto buttonSelector = UIAutomation::byObjectName("myButton");
if (uiAutomation.clickWidget(buttonSelector)) {
    qDebug() << "Button clicked successfully";
}

// Type text in an input field
auto inputSelector = UIAutomation::byObjectName("textInput");
uiAutomation.typeText(inputSelector, "Hello, World!");

// Verify widget state
if (uiAutomation.verifyWidgetVisible(buttonSelector)) {
    qDebug() << "Button is visible";
}
```

#### Widget Selection Strategies

The framework supports multiple widget selection strategies:

| Strategy | Description | Example |
|----------|-------------|---------|
| `ObjectName` | Find by QObject::objectName() | `byObjectName("myButton")` |
| `Text` | Find by displayed text content | `byText("Click Me")` |
| `ClassName` | Find by widget class name | `byClassName("QPushButton")` |
| `Property` | Find by custom property value | `byProperty("enabled", true)` |
| `Position` | Find by screen position | `byPosition(QPoint(100, 200))` |
| `Custom` | Find using custom predicate | `byCustom([](QWidget* w) { return w->isVisible(); })` |

#### Mouse Interactions

```cpp
// Basic clicking
UIAutomation::MouseOptions options;
options.button = Qt::LeftButton;
options.modifiers = Qt::ControlModifier;
options.offset = QPoint(10, 5);  // Offset from center
uiAutomation.clickWidget(selector, options);

// Double-clicking
uiAutomation.doubleClickWidget(selector);

// Right-clicking
uiAutomation.rightClickWidget(selector);

// Hovering
uiAutomation.hoverWidget(selector, 2000);  // Hover for 2 seconds

// Drag and drop
auto sourceSelector = UIAutomation::byObjectName("source");
auto targetSelector = UIAutomation::byObjectName("target");
uiAutomation.dragWidget(sourceSelector, targetSelector);
```

#### Keyboard Interactions

```cpp
// Type text with options
UIAutomation::KeyboardOptions keyOptions;
keyOptions.clearFirst = true;
keyOptions.typingSpeedMs = 50;  // 50ms between characters
uiAutomation.typeText(selector, "Test input", keyOptions);

// Press individual keys
uiAutomation.pressKey(selector, Qt::Key_Enter);
uiAutomation.pressKey(selector, Qt::Key_A, {Qt::ControlModifier});

// Press key sequences
uiAutomation.pressKeySequence(selector, QKeySequence::Copy);
```

#### Form Automation

```cpp
// Define form fields
QList<UIAutomation::FormField> formFields = {
    {UIAutomation::byObjectName("nameInput"), "John Doe", "", true},
    {UIAutomation::byObjectName("emailInput"), "john@example.com", "", true},
    {UIAutomation::byObjectName("ageSpinBox"), 30, "", false},
    {UIAutomation::byObjectName("subscribeCheckBox"), true, "", false}
};

// Fill the entire form
if (uiAutomation.fillForm(formFields)) {
    qDebug() << "Form filled successfully";
}

// Submit form
auto submitSelector = UIAutomation::byText("Submit");
uiAutomation.submitForm(submitSelector);
```

#### Wait Conditions and Synchronization

```cpp
// Wait for widget to appear
auto dialogSelector = UIAutomation::byObjectName("confirmDialog");
if (uiAutomation.waitForWidget(dialogSelector, 5000)) {
    qDebug() << "Dialog appeared";
}

// Wait for specific conditions
UIAutomation::WaitSpec waitSpec;
waitSpec.condition = UIAutomation::WaitCondition::TextChanged;
waitSpec.selector = UIAutomation::byObjectName("statusLabel");
waitSpec.expectedValue = "Processing complete";
waitSpec.timeoutMs = 10000;

if (uiAutomation.waitForCondition(waitSpec)) {
    qDebug() << "Status updated";
}

// Wait for widget state changes
uiAutomation.waitForWidgetEnabled(buttonSelector, 3000);
uiAutomation.waitForWidgetVisible(resultSelector, 5000);
```

### 2. WidgetSelector Class

The `WidgetSelector` class provides advanced widget selection capabilities with CSS-like syntax and sophisticated matching algorithms.

#### Advanced Selection Features

- **CSS-like Selectors**: Familiar web-style selector syntax
- **Spatial Relationships**: Position-based widget finding
- **Constraint Matching**: Multiple criteria filtering
- **Fluent API**: Builder pattern for complex selectors
- **Performance Optimization**: Efficient widget tree traversal

#### Usage Examples

```cpp
#include "widget_selector.h"

// Basic selectors
auto selector1 = WidgetSelector::byObjectName("myButton");
auto selector2 = WidgetSelector::byText("Click Me");
auto selector3 = WidgetSelector::byClassName("QPushButton");

// Advanced selectors with constraints
auto selector4 = WidgetSelector::byObjectName("button")
    .requireVisible(true)
    .requireEnabled(true)
    .setIndex(0);

// Fluent API builder
auto selector5 = WIDGET_BUILDER()
    .objectName("dialog")
    .visible(true)
    .child("QPushButton")
    .text("OK")
    .build();

// CSS-like selectors
auto selector6 = WidgetSelector::byCss("QPushButton[text='Save']:enabled");

// Custom predicate selectors
auto selector7 = WidgetSelector::byCustom([](QWidget* widget) {
    return widget->isVisible() && 
           widget->size().width() > 100 &&
           widget->property("important").toBool();
});
```

#### Spatial Relationships

```cpp
// Find widgets based on position relative to other widgets
QWidget* referenceWidget = findSomeWidget();

auto selector = WidgetSelector::byClassName("QLabel")
    .addSpatialConstraint(WidgetSelector::SpatialRelation::Below, referenceWidget)
    .addSpatialConstraint(WidgetSelector::SpatialRelation::Near, referenceWidget, 50);

QWidget* foundWidget = selector.findWidget();
```

## Integration with Test Framework

The UI automation framework integrates seamlessly with the enhanced test framework:

### Using with TestBase

```cpp
#include "test_base.h"
#include "ui_automation.h"

DECLARE_TEST_CLASS(MyUITest, UI, High, "ui", "automation")

private:
    UIAutomation* m_uiAutomation;
    QWidget* m_testWidget;

private slots:
    void initTestCase() {
        TestBase::initTestCase();
        
        // Create test widget
        m_testWidget = createTestWidget();
        m_testWidget->show();
        QTest::qWaitForWindowActive(m_testWidget);
        
        // Setup UI automation
        m_uiAutomation = new UIAutomation(this);
        m_uiAutomation->enableDetailedLogging(true);
    }

    TEST_METHOD(test_buttonClick_updatesLabel_correctly) {
        // Use convenient macros
        UI_CLICK(UIAutomation::byObjectName("testButton"));
        UI_VERIFY_TEXT(UIAutomation::byObjectName("resultLabel"), "Button clicked");
    }

    TEST_METHOD(test_formFilling_submitsData_successfully) {
        // Fill form using automation
        QList<UIAutomation::FormField> fields = {
            {UIAutomation::byObjectName("nameField"), "Test User"},
            {UIAutomation::byObjectName("emailField"), "test@example.com"}
        };
        
        TEST_VERIFY_WITH_MSG(m_uiAutomation->fillForm(fields), 
                           "Should fill form successfully");
        
        UI_CLICK(UIAutomation::byText("Submit"));
        UI_WAIT_FOR(UIAutomation::byText("Success"), 5000);
    }

END_TEST_CLASS()
```

### Convenience Macros

The framework provides convenient macros for common operations:

```cpp
// Widget interaction macros
UI_CLICK(selector)                    // Click widget
UI_TYPE(selector, text)               // Type text
UI_WAIT_FOR(selector, timeout)        // Wait for widget

// Verification macros
UI_VERIFY_VISIBLE(selector)           // Verify widget is visible
UI_VERIFY_ENABLED(selector)           // Verify widget is enabled
UI_VERIFY_TEXT(selector, text)        // Verify widget text
```

## Advanced Features

### 1. Dialog Handling

```cpp
// Handle modal dialogs
QMap<QString, QVariant> dialogResponses = {
    {"nameField", "John Doe"},
    {"confirmCheckbox", true}
};

if (uiAutomation.handleDialog("Settings Dialog", dialogResponses)) {
    qDebug() << "Dialog handled successfully";
}

// Accept/reject dialogs
uiAutomation.acceptDialog("Confirmation");
uiAutomation.rejectDialog("Warning");
```

### 2. Menu Navigation

```cpp
// Navigate through menus
QStringList menuPath = {"File", "Recent Files", "document.txt"};
uiAutomation.clickMenuItem(menuPath);

// Context menus
uiAutomation.openContextMenu(UIAutomation::byObjectName("fileList"));
uiAutomation.selectContextMenuItem(selector, "Delete");
```

### 3. Screenshot and Visual Testing

```cpp
// Capture screenshots
QPixmap screenshot = uiAutomation.captureWidget(selector);
uiAutomation.saveScreenshot(selector, "test_result.png");

// Visual comparison (basic)
bool matches = uiAutomation.compareWidgetScreenshot(
    selector, "baseline.png", 0.95  // 95% similarity threshold
);
```

### 4. Accessibility Support

```cpp
// Navigate using keyboard
uiAutomation.navigateByTab(3);  // Tab 3 times
uiAutomation.navigateByArrowKeys(Qt::Key_Down, 2);  // Down arrow 2 times
uiAutomation.activateFocusedWidget();  // Activate current widget

// Get accessibility information
QStringList accessInfo = uiAutomation.getAccessibilityInfo(selector);
```

## Configuration and Customization

### Global Configuration

```cpp
// Configure timeouts and delays
uiAutomation.setDefaultTimeout(10000);  // 10 second timeout
uiAutomation.setDefaultDelay(100);      // 100ms delay between actions

// Configure screenshots
uiAutomation.setScreenshotDirectory("./test_screenshots");
uiAutomation.enableAutomaticScreenshots(true);

// Configure logging
uiAutomation.enableDetailedLogging(true);
uiAutomation.setRetryAttempts(3);
```

### Test Configuration Integration

```cpp
// Use test configuration for UI automation settings
TestConfig::TestSuiteConfig config = getTestConfig();
if (config.tags.contains("ui-slow")) {
    uiAutomation.setDefaultTimeout(15000);  // Longer timeout for slow UI tests
}

if (isRunningInCI()) {
    uiAutomation.enableAutomaticScreenshots(false);  // Disable screenshots in CI
}
```

## Best Practices

### 1. Reliable Widget Selection

```cpp
// Good: Use stable identifiers
auto selector = UIAutomation::byObjectName("saveButton");

// Better: Combine multiple criteria for robustness
auto selector = WIDGET_BUILDER()
    .objectName("saveButton")
    .text("Save")
    .enabled(true)
    .visible(true)
    .build();

// Best: Use semantic selectors
auto selector = UIAutomation::byText("Save Document")
    .requireEnabled(true);
```

### 2. Proper Synchronization

```cpp
// Good: Wait for elements before interacting
UI_WAIT_FOR(dialogSelector, 5000);
UI_CLICK(UIAutomation::byText("OK"));

// Better: Wait for specific conditions
uiAutomation.waitForWidgetEnabled(submitButton, 3000);
uiAutomation.clickWidget(submitButton);

// Best: Use appropriate wait conditions
UIAutomation::WaitSpec waitSpec;
waitSpec.condition = UIAutomation::WaitCondition::PropertyChanged;
waitSpec.selector = progressSelector;
waitSpec.propertyName = "value";
waitSpec.expectedValue = 100;
uiAutomation.waitForCondition(waitSpec);
```

### 3. Error Handling and Debugging

```cpp
// Enable detailed logging for debugging
uiAutomation.enableDetailedLogging(true);

// Use verification methods for better error messages
TEST_VERIFY_WITH_MSG(uiAutomation.verifyWidgetExists(selector),
                   "Widget should exist before interaction");

// Capture screenshots on failure
if (!uiAutomation.clickWidget(selector)) {
    uiAutomation.saveScreenshot(UIAutomation::byObjectName("mainWindow"), 
                               "failure_screenshot.png");
    QFAIL("Failed to click widget");
}
```

### 4. Performance Considerations

```cpp
// Cache selectors for repeated use
static auto buttonSelector = UIAutomation::byObjectName("frequentButton");

// Use specific parent contexts to limit search scope
QWidget* dialogWidget = findDialog();
auto buttonInDialog = UIAutomation::byText("OK");
QWidget* okButton = uiAutomation.findWidget(buttonInDialog, dialogWidget);

// Batch operations when possible
QList<UIAutomation::FormField> allFields = {
    // ... all form fields
};
uiAutomation.fillForm(allFields);  // More efficient than individual calls
```

## Troubleshooting

### Common Issues

1. **Widget Not Found**
   ```cpp
   // Check if widget exists
   if (!uiAutomation.verifyWidgetExists(selector)) {
       qDebug() << "Widget not found, checking alternatives...";
       // Try alternative selectors or wait longer
   }
   ```

2. **Timing Issues**
   ```cpp
   // Increase timeouts for slow operations
   uiAutomation.setDefaultTimeout(15000);
   
   // Use explicit waits
   uiAutomation.waitForWidgetVisible(selector, 10000);
   ```

3. **Widget Not Interactable**
   ```cpp
   // Check widget state before interaction
   if (!uiAutomation.verifyWidgetEnabled(selector)) {
       qDebug() << "Widget is disabled, waiting for it to be enabled...";
       uiAutomation.waitForWidgetEnabled(selector, 5000);
   }
   ```

### Debug Mode

```cpp
// Enable comprehensive debugging
uiAutomation.enableDetailedLogging(true);
uiAutomation.enableAutomaticScreenshots(true);

// Use widget hierarchy dumping (if available)
// dumpWidgetHierarchy(QApplication::activeWindow());
```

## Platform Considerations

### Cross-Platform Compatibility

The UI automation framework is designed to work across Windows, macOS, and Linux:

```cpp
// Platform-specific adjustments
#ifdef Q_OS_WIN
    uiAutomation.setDefaultDelay(150);  // Slightly longer delays on Windows
#elif defined(Q_OS_MAC)
    uiAutomation.setDefaultDelay(100);  // Standard delays on macOS
#else
    uiAutomation.setDefaultDelay(75);   // Faster on Linux
#endif
```

### CI/CD Integration

```cpp
// Configure for headless CI environments
if (isRunningInCI()) {
    uiAutomation.enableAutomaticScreenshots(false);
    uiAutomation.setDefaultTimeout(30000);  // Longer timeouts in CI
    
    // Use offscreen platform for testing
    // Set QT_QPA_PLATFORM=offscreen environment variable
}
```

## Future Enhancements

Planned improvements for the UI automation framework:

1. **Visual Testing**
   - Advanced image comparison algorithms
   - Baseline management system
   - Visual regression detection

2. **Enhanced Selectors**
   - XPath-like navigation syntax
   - Advanced CSS selector support
   - Fuzzy matching algorithms

3. **Performance Optimization**
   - Widget caching and indexing
   - Parallel test execution support
   - Smart retry mechanisms

4. **Advanced Interactions**
   - Touch and gesture simulation
   - Multi-monitor support
   - Native OS integration

5. **AI-Powered Features**
   - Intelligent widget identification
   - Self-healing selectors
   - Automated test generation