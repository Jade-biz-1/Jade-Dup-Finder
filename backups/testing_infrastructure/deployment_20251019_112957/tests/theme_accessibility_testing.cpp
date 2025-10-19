#include "theme_accessibility_testing.h"
#include "ui_automation.h"
#include "visual_testing.h"
#include <QApplication>
#include <QWidget>
#include <QStyle>
#include <QStyleOption>
#include <QPainter>
#include <QFontMetrics>
#include <QAccessible>
#include <QDebug>
#include <QtMath>

ThemeAccessibilityTesting::ThemeAccessibilityTesting(QObject* parent)
    : QObject(parent)
    , m_uiAutomation(nullptr)
    , m_visualTesting(nullptr)
    , m_currentTheme(ThemeType::System)
{
    // Set default configuration
    m_config.complianceLevel = AccessibilityLevel::WCAG_AA;
    m_config.minContrastRatio = 4.5;
    m_config.minLargeTextRatio = 3.0;
    
    // Store original palette
    m_originalPalette = QApplication::palette();
    
    // Initialize themes
    initializeThemes();
}

ThemeAccessibilityTesting::~ThemeAccessibilityTesting() {
    restoreOriginalTheme();
}

void ThemeAccessibilityTesting::setAccessibilityConfig(const AccessibilityConfig& config) {
    m_config = config;
}

ThemeAccessibilityTesting::AccessibilityConfig ThemeAccessibilityTesting::getAccessibilityConfig() const {
    return m_config;
}

void ThemeAccessibilityTesting::setUIAutomation(UIAutomation* uiAutomation) {
    m_uiAutomation = uiAutomation;
}

void ThemeAccessibilityTesting::setVisualTesting(VisualTesting* visualTesting) {
    m_visualTesting = visualTesting;
}

bool ThemeAccessibilityTesting::switchToTheme(ThemeType themeType) {
    QPalette palette = createThemePalette(themeType);
    applyPaletteToApplication(palette);
    
    m_currentTheme = themeType;
    
    QString themeName;
    switch (themeType) {
        case ThemeType::Light: themeName = "Light"; break;
        case ThemeType::Dark: themeName = "Dark"; break;
        case ThemeType::HighContrast: themeName = "HighContrast"; break;
        case ThemeType::System: themeName = "System"; break;
        default: themeName = "Custom"; break;
    }
    
    m_currentThemeName = themeName;
    emit themeChanged(themeType, themeName);
    
    return true;
}

ThemeAccessibilityTesting::ContrastResult ThemeAccessibilityTesting::testColorContrast(QWidget* widget) {
    ContrastResult result;
    result.widget = widget;
    result.level = m_config.complianceLevel;
    result.elementDescription = generateWidgetDescription(widget);
    
    if (!widget) {
        return result;
    }
    
    // Extract colors
    result.foreground = extractForegroundColor(widget);
    result.background = extractBackgroundColor(widget);
    
    // Calculate contrast ratio
    result.contrastRatio = calculateContrastRatio(result.foreground, result.background);
    
    // Determine required ratio based on text size
    QFont font = widget->font();
    bool isLarge = isLargeText(font);
    result.requiredRatio = isLarge ? m_config.minLargeTextRatio : m_config.minContrastRatio;
    
    // Check if it passes
    result.passes = meetsContrastRequirement(result.contrastRatio, m_config.complianceLevel, isLarge);
    
    if (!result.passes) {
        emit contrastTestFailed(result);
    }
    
    return result;
}

ThemeAccessibilityTesting::ContrastResult ThemeAccessibilityTesting::testColorContrast(
    const QColor& foreground, const QColor& background, AccessibilityLevel level) {
    
    ContrastResult result;
    result.foreground = foreground;
    result.background = background;
    result.level = level;
    result.contrastRatio = calculateContrastRatio(foreground, background);
    
    // Determine required ratio based on level
    switch (level) {
        case AccessibilityLevel::WCAG_A:
            result.requiredRatio = 3.0;
            break;
        case AccessibilityLevel::WCAG_AA:
            result.requiredRatio = 4.5;
            break;
        case AccessibilityLevel::WCAG_AAA:
            result.requiredRatio = 7.0;
            break;
        default:
            result.requiredRatio = 4.5;
            break;
    }
    
    result.passes = result.contrastRatio >= result.requiredRatio;
    
    return result;
}

ThemeAccessibilityTesting::KeyboardNavResult ThemeAccessibilityTesting::testKeyboardNavigation(QWidget* widget) {
    KeyboardNavResult result;
    result.widget = widget;
    
    if (!widget) {
        return result;
    }
    
    // Test if widget can receive focus
    result.canReceiveFocus = canWidgetReceiveFocus(widget);
    
    // Test focus indicator
    if (result.canReceiveFocus) {
        widget->setFocus();
        QApplication::processEvents();
        
        // Check if focus is visually indicated (simplified check)
        result.focusIndicator = widget->hasFocus() ? "Has focus" : "No focus indicator";
    }
    
    // Test keyboard shortcuts (if UI automation is available)
    if (m_uiAutomation && result.canReceiveFocus) {
        // Test common shortcuts
        result.enterWorks = m_uiAutomation->pressKey(
            UIAutomation::byObjectName(widget->objectName()), Qt::Key_Return);
        result.escapeWorks = m_uiAutomation->pressKey(
            UIAutomation::byObjectName(widget->objectName()), Qt::Key_Escape);
    }
    
    // Get accessible actions
    QAccessibleInterface* interface = getAccessibleInterface(widget);
    if (interface) {
        for (int i = 0; i < interface->actionCount(); ++i) {
            result.accessibleActions.append(interface->actionNames().at(i));
        }
    }
    
    if (!result.canReceiveFocus || result.accessibleActions.isEmpty()) {
        emit keyboardNavTestFailed(result);
    }
    
    return result;
}

ThemeAccessibilityTesting::ScreenReaderResult ThemeAccessibilityTesting::testScreenReaderCompatibility(QWidget* widget) {
    ScreenReaderResult result;
    result.widget = widget;
    
    if (!widget) {
        return result;
    }
    
    QAccessibleInterface* interface = getAccessibleInterface(widget);
    if (!interface) {
        emit screenReaderTestFailed(result);
        return result;
    }
    
    // Test accessible name
    result.accessibleName = interface->text(QAccessible::Name);
    result.hasAccessibleName = !result.accessibleName.isEmpty();
    
    // Test accessible description
    result.accessibleDescription = interface->text(QAccessible::Description);
    result.hasAccessibleDescription = !result.accessibleDescription.isEmpty();
    
    // Test role
    result.role = interface->role();
    result.hasCorrectRole = (result.role != QAccessible::NoRole);
    
    // Test state
    result.state = interface->state();
    result.stateReported = true; // Simplified - assume state is reported if interface exists
    
    bool allTestsPassed = result.hasAccessibleName && result.hasCorrectRole && result.stateReported;
    
    if (!allTestsPassed) {
        emit screenReaderTestFailed(result);
    }
    
    return result;
}bool Theme
AccessibilityTesting::runFullAccessibilityAudit(QWidget* rootWidget) {
    if (!rootWidget) {
        return false;
    }
    
    int totalTests = 0;
    int passedTests = 0;
    int failedTests = 0;
    
    // Test color contrast
    if (m_config.testColorContrast) {
        QList<ContrastResult> contrastResults = testAllColorContrasts(rootWidget);
        totalTests += contrastResults.size();
        for (const ContrastResult& result : contrastResults) {
            if (result.passes) {
                passedTests++;
            } else {
                failedTests++;
            }
        }
    }
    
    // Test keyboard navigation
    if (m_config.testKeyboardNav) {
        QList<KeyboardNavResult> keyboardResults = testAllKeyboardNavigation(rootWidget);
        totalTests += keyboardResults.size();
        for (const KeyboardNavResult& result : keyboardResults) {
            if (result.canReceiveFocus && !result.accessibleActions.isEmpty()) {
                passedTests++;
            } else {
                failedTests++;
            }
        }
    }
    
    // Test screen reader compatibility
    if (m_config.testScreenReader) {
        QList<ScreenReaderResult> screenReaderResults = testAllScreenReaderCompatibility(rootWidget);
        totalTests += screenReaderResults.size();
        for (const ScreenReaderResult& result : screenReaderResults) {
            if (result.hasAccessibleName && result.hasCorrectRole) {
                passedTests++;
            } else {
                failedTests++;
            }
        }
    }
    
    emit accessibilityAuditCompleted(totalTests, passedTests, failedTests);
    
    return failedTests == 0;
}

// Static utility methods
double ThemeAccessibilityTesting::calculateContrastRatio(const QColor& color1, const QColor& color2) {
    double lum1 = calculateRelativeLuminance(color1);
    double lum2 = calculateRelativeLuminance(color2);
    
    // Ensure lum1 is the lighter color
    if (lum1 < lum2) {
        qSwap(lum1, lum2);
    }
    
    return (lum1 + 0.05) / (lum2 + 0.05);
}

double ThemeAccessibilityTesting::calculateRelativeLuminance(const QColor& color) {
    // Convert to linear RGB
    auto toLinear = [](double value) {
        value /= 255.0;
        return (value <= 0.03928) ? value / 12.92 : qPow((value + 0.055) / 1.055, 2.4);
    };
    
    double r = toLinear(color.red());
    double g = toLinear(color.green());
    double b = toLinear(color.blue());
    
    // Calculate relative luminance
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

bool ThemeAccessibilityTesting::isLargeText(const QFont& font) {
    QFontMetrics metrics(font);
    int pixelSize = metrics.height();
    
    // Large text is 18pt+ or 14pt+ bold
    // Approximate conversion: 1pt ≈ 1.33px
    bool isLarge = (pixelSize >= 24) || (font.bold() && pixelSize >= 19);
    
    return isLarge;
}

QAccessibleInterface* ThemeAccessibilityTesting::getAccessibleInterface(QWidget* widget) {
    if (!widget) {
        return nullptr;
    }
    
    return QAccessible::queryAccessibleInterface(widget);
}

// Private helper methods
void ThemeAccessibilityTesting::initializeThemes() {
    m_themeNames[ThemeType::Light] = "Light";
    m_themeNames[ThemeType::Dark] = "Dark";
    m_themeNames[ThemeType::HighContrast] = "HighContrast";
    m_themeNames[ThemeType::System] = "System";
}

void ThemeAccessibilityTesting::restoreOriginalTheme() {
    QApplication::setPalette(m_originalPalette);
}

QPalette ThemeAccessibilityTesting::createThemePalette(ThemeType themeType) const {
    QPalette palette = m_originalPalette;
    
    switch (themeType) {
        case ThemeType::Light:
            palette.setColor(QPalette::Window, QColor(240, 240, 240));
            palette.setColor(QPalette::WindowText, QColor(0, 0, 0));
            palette.setColor(QPalette::Base, QColor(255, 255, 255));
            palette.setColor(QPalette::Text, QColor(0, 0, 0));
            palette.setColor(QPalette::Button, QColor(225, 225, 225));
            palette.setColor(QPalette::ButtonText, QColor(0, 0, 0));
            break;
            
        case ThemeType::Dark:
            palette.setColor(QPalette::Window, QColor(45, 45, 45));
            palette.setColor(QPalette::WindowText, QColor(255, 255, 255));
            palette.setColor(QPalette::Base, QColor(35, 35, 35));
            palette.setColor(QPalette::Text, QColor(255, 255, 255));
            palette.setColor(QPalette::Button, QColor(60, 60, 60));
            palette.setColor(QPalette::ButtonText, QColor(255, 255, 255));
            break;
            
        case ThemeType::HighContrast:
            palette.setColor(QPalette::Window, QColor(0, 0, 0));
            palette.setColor(QPalette::WindowText, QColor(255, 255, 255));
            palette.setColor(QPalette::Base, QColor(0, 0, 0));
            palette.setColor(QPalette::Text, QColor(255, 255, 255));
            palette.setColor(QPalette::Button, QColor(0, 0, 0));
            palette.setColor(QPalette::ButtonText, QColor(255, 255, 255));
            palette.setColor(QPalette::Highlight, QColor(255, 255, 0));
            palette.setColor(QPalette::HighlightedText, QColor(0, 0, 0));
            break;
            
        case ThemeType::System:
        default:
            // Keep original palette
            break;
    }
    
    return palette;
}

void ThemeAccessibilityTesting::applyPaletteToApplication(const QPalette& palette) {
    QApplication::setPalette(palette);
    QApplication::processEvents();
}

QColor ThemeAccessibilityTesting::extractForegroundColor(QWidget* widget) const {
    if (!widget) {
        return QColor();
    }
    
    QPalette palette = widget->palette();
    return palette.color(QPalette::WindowText);
}

QColor ThemeAccessibilityTesting::extractBackgroundColor(QWidget* widget) const {
    if (!widget) {
        return QColor();
    }
    
    QPalette palette = widget->palette();
    return palette.color(QPalette::Window);
}

QList<ThemeAccessibilityTesting::ContrastResult> ThemeAccessibilityTesting::testAllColorContrasts(QWidget* rootWidget) {
    QList<ContrastResult> results;
    
    if (!rootWidget) {
        return results;
    }
    
    // Test root widget
    if (!isWidgetExempt(rootWidget)) {
        results.append(testColorContrast(rootWidget));
    }
    
    // Test all child widgets
    QList<QWidget*> children = rootWidget->findChildren<QWidget*>();
    for (QWidget* child : children) {
        if (!isWidgetExempt(child) && child->isVisible()) {
            results.append(testColorContrast(child));
        }
    }
    
    return results;
}

QList<ThemeAccessibilityTesting::KeyboardNavResult> ThemeAccessibilityTesting::testAllKeyboardNavigation(QWidget* rootWidget) {
    QList<KeyboardNavResult> results;
    
    if (!rootWidget) {
        return results;
    }
    
    QList<QWidget*> focusableWidgets = getFocusableWidgets(rootWidget);
    for (QWidget* widget : focusableWidgets) {
        if (!isWidgetExempt(widget)) {
            results.append(testKeyboardNavigation(widget));
        }
    }
    
    return results;
}

QList<ThemeAccessibilityTesting::ScreenReaderResult> ThemeAccessibilityTesting::testAllScreenReaderCompatibility(QWidget* rootWidget) {
    QList<ScreenReaderResult> results;
    
    if (!rootWidget) {
        return results;
    }
    
    // Test root widget
    if (!isWidgetExempt(rootWidget)) {
        results.append(testScreenReaderCompatibility(rootWidget));
    }
    
    // Test all child widgets
    QList<QWidget*> children = rootWidget->findChildren<QWidget*>();
    for (QWidget* child : children) {
        if (!isWidgetExempt(child) && child->isVisible()) {
            results.append(testScreenReaderCompatibility(child));
        }
    }
    
    return results;
}

bool ThemeAccessibilityTesting::canWidgetReceiveFocus(QWidget* widget) const {
    return widget && (widget->focusPolicy() != Qt::NoFocus);
}

QList<QWidget*> ThemeAccessibilityTesting::getFocusableWidgets(QWidget* rootWidget) const {
    QList<QWidget*> focusableWidgets;
    
    if (!rootWidget) {
        return focusableWidgets;
    }
    
    if (canWidgetReceiveFocus(rootWidget)) {
        focusableWidgets.append(rootWidget);
    }
    
    QList<QWidget*> children = rootWidget->findChildren<QWidget*>();
    for (QWidget* child : children) {
        if (canWidgetReceiveFocus(child) && child->isVisible()) {
            focusableWidgets.append(child);
        }
    }
    
    return focusableWidgets;
}

bool ThemeAccessibilityTesting::meetsContrastRequirement(double ratio, AccessibilityLevel level, bool isLargeText) const {
    double requiredRatio;
    
    switch (level) {
        case AccessibilityLevel::WCAG_A:
            requiredRatio = isLargeText ? 3.0 : 3.0;
            break;
        case AccessibilityLevel::WCAG_AA:
            requiredRatio = isLargeText ? 3.0 : 4.5;
            break;
        case AccessibilityLevel::WCAG_AAA:
            requiredRatio = isLargeText ? 4.5 : 7.0;
            break;
        default:
            requiredRatio = 4.5;
            break;
    }
    
    return ratio >= requiredRatio;
}

bool ThemeAccessibilityTesting::isWidgetExempt(QWidget* widget) const {
    if (!widget) {
        return true;
    }
    
    QString className = widget->metaObject()->className();
    return m_config.exemptWidgets.contains(className) || 
           m_config.exemptWidgets.contains(widget->objectName());
}

QString ThemeAccessibilityTesting::generateWidgetDescription(QWidget* widget) const {
    if (!widget) {
        return "Unknown widget";
    }
    
    QString description = widget->metaObject()->className();
    if (!widget->objectName().isEmpty()) {
        description += QString(" (%1)").arg(widget->objectName());
    }
    
    return description;
}