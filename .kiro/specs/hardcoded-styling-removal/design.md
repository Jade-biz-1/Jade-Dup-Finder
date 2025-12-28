# Design Document

## Overview

This design addresses the systematic removal of hardcoded styling throughout the CloneClean GUI components and establishes a comprehensive theme-aware styling system. The solution involves enhancing the existing ThemeManager to provide centralized styling methods, updating all GUI components to use theme-aware properties, and implementing minimum size constraints for UI controls.

## Architecture

### Enhanced ThemeManager Architecture

```
ThemeManager (Enhanced)
├── Core Theme System (Existing)
│   ├── Theme enumeration (Light, Dark, System)
│   ├── Application-wide stylesheet generation
│   └── Dialog registration and updates
├── Component-Specific Styling (New)
│   ├── Progress bar theme methods
│   ├── Custom widget styling
│   └── Status indicator colors
├── Minimum Size Management (New)
│   ├── Control size constraints
│   ├── Layout minimum requirements
│   └── Responsive sizing rules
└── Hardcoded Style Override (New)
    ├── Style conflict detection
    ├── Forced theme application
    └── Runtime style validation
```

### Component Integration Pattern

```
GUI Component
├── Constructor
│   ├── Create UI elements
│   ├── Register with ThemeManager
│   └── Apply initial theme-aware styling
├── Theme Update Handler
│   ├── Receive theme change notifications
│   ├── Update component-specific styling
│   └── Refresh child widgets
└── Minimum Size Enforcement
    ├── Set minimum dimensions
    ├── Handle layout constraints
    └── Maintain usability standards
```

## Components and Interfaces

### 1. Enhanced ThemeManager Interface

```cpp
class ThemeManager : public QObject {
public:
    // Existing methods...
    
    // New component-specific styling methods
    QString getProgressBarStyle(ProgressType type = Normal) const;
    QString getStatusIndicatorStyle(StatusType status) const;
    QString getCustomWidgetStyle(WidgetType widget) const;
    
    // Minimum size management
    QSize getMinimumControlSize(ControlType control) const;
    void enforceMinimumSizes(QWidget* parent);
    
    // Style override and validation
    void removeHardcodedStyles(QWidget* widget);
    void validateThemeCompliance(QWidget* widget);
    
    // Enhanced dialog management
    void applyComprehensiveTheme(QDialog* dialog);
    void registerCustomWidget(QWidget* widget, const QString& styleClass);
};

enum class ProgressType {
    Normal, Success, Warning, Error, Performance
};

enum class StatusType {
    Success, Warning, Error, Info, Neutral
};

enum class ControlType {
    Button, LineEdit, ComboBox, CheckBox, RadioButton, 
    Label, GroupBox, TabWidget, ProgressBar, Slider
};
```

### 2. Component Update Interface

```cpp
class ThemeAwareComponent {
public:
    virtual void onThemeChanged() = 0;
    virtual void applyMinimumSizes() = 0;
    virtual void removeHardcodedStyling() = 0;
};
```

### 3. Style Configuration Structure

```cpp
struct StyleConfiguration {
    QSize minimumButtonSize{80, 24};
    QSize minimumLineEditSize{100, 20};
    QSize minimumComboBoxSize{120, 24};
    QSize minimumCheckBoxSize{16, 16};
    QSize minimumLabelSize{50, 16};
    
    int standardPadding{8};
    int standardMargin{4};
    int standardBorderRadius{4};
    
    QString fontFamily{"Segoe UI, Ubuntu, sans-serif"};
    int baseFontSize{9};
    int titleFontSize{11};
};
```

## Data Models

### Theme Style Registry

```cpp
class StyleRegistry {
private:
    QMap<QString, QString> m_lightThemeStyles;
    QMap<QString, QString> m_darkThemeStyles;
    QMap<ControlType, QSize> m_minimumSizes;
    QMap<QString, StyleConfiguration> m_componentConfigs;
    
public:
    void registerComponentStyle(const QString& component, 
                               const QString& lightStyle, 
                               const QString& darkStyle);
    QString getComponentStyle(const QString& component, Theme theme) const;
    QSize getMinimumSize(ControlType control) const;
};
```

### Hardcoded Style Detection

```cpp
struct HardcodedStyleIssue {
    QString fileName;
    int lineNumber;
    QString issueType;  // "hex-color", "rgb-value", "inline-style"
    QString currentValue;
    QString suggestedFix;
};

class StyleValidator {
public:
    QList<HardcodedStyleIssue> detectHardcodedStyles(QWidget* widget);
    bool validateThemeCompliance(QWidget* widget);
    void generateComplianceReport();
};
```

## Error Handling

### Style Application Errors

1. **Missing Theme Styles**: Fallback to default palette colors
2. **Invalid Style Syntax**: Log warning and use base styles
3. **Component Registration Failures**: Continue with basic theme application
4. **Minimum Size Conflicts**: Prioritize usability over exact dimensions

### Runtime Style Validation

```cpp
class StyleErrorHandler {
public:
    void handleStyleError(const QString& component, const QString& error);
    void logStyleWarning(const QString& message);
    void reportThemeInconsistency(QWidget* widget);
};
```

## Testing Strategy

### 1. Automated Style Compliance Testing

```cpp
class StyleComplianceTests : public QObject {
    Q_OBJECT
private slots:
    void testNoHardcodedColors();
    void testThemeConsistency();
    void testMinimumSizes();
    void testProgressBarTheming();
    void testCustomWidgetStyling();
};
```

### 2. Visual Theme Testing

- **Theme Switching Tests**: Verify all components update correctly
- **Contrast Ratio Tests**: Ensure accessibility compliance
- **Size Constraint Tests**: Validate minimum size enforcement
- **Cross-Platform Tests**: Verify consistent appearance across platforms

### 3. Performance Testing

- **Theme Application Speed**: Measure time to apply themes to all components
- **Memory Usage**: Monitor theme-related memory consumption
- **Style Cache Efficiency**: Validate style caching performance

## Implementation Phases

### Phase 1: ThemeManager Enhancement
- Add component-specific styling methods
- Implement minimum size management
- Create style registry system
- Add hardcoded style detection

### Phase 2: Critical Component Updates
- Fix scan_progress_dialog.cpp (most hardcoded styles)
- Update smart_selection_dialog.cpp
- Fix scan_scope_preview_widget.cpp
- Update thumbnail_delegate.cpp

### Phase 3: Remaining Component Updates
- Fix preset_manager_dialog.cpp
- Update results_window.cpp
- Fix file_operation_progress_dialog.cpp
- Update all remaining GUI components

### Phase 4: Validation and Testing
- Implement style compliance validation
- Add automated testing
- Perform comprehensive theme testing
- Document theme-aware styling guidelines

## Design Decisions and Rationales

### 1. Centralized Styling Approach
**Decision**: Extend ThemeManager rather than create separate styling classes
**Rationale**: Maintains consistency with existing architecture and provides single source of truth for all styling

### 2. Progressive Enhancement Strategy
**Decision**: Update components incrementally rather than wholesale replacement
**Rationale**: Minimizes risk and allows for thorough testing of each component

### 3. Minimum Size Enforcement
**Decision**: Implement size constraints at the ThemeManager level
**Rationale**: Ensures consistent sizing across all components and prevents layout-related usability issues

### 4. Style Override Mechanism
**Decision**: Provide methods to detect and override hardcoded styles
**Rationale**: Allows for runtime correction of styling issues and provides debugging capabilities

### 5. Backward Compatibility
**Decision**: Maintain existing ThemeManager API while adding new functionality
**Rationale**: Ensures existing code continues to work while providing enhanced capabilities