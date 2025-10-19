#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QColor>
#include <QFont>
#include <QSize>
#include <QMap>
#include <QVariant>
#include <QKeySequence>
#include <QAccessible>
#include <QAccessibleInterface>
#include <QPalette>
#include <QStyle>

class QWidget;
class QApplication;
class UIAutomation;
class VisualTesting;

/**
 * @brief Comprehensive theme and accessibility testing framework
 * 
 * Provides automated testing capabilities for UI themes, accessibility compliance,
 * keyboard navigation, screen reader compatibility, and color contrast validation.
 */
class ThemeAccessibilityTesting : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Theme types for testing
     */
    enum class ThemeType {
        Light,              ///< Light theme
        Dark,               ///< Dark theme
        HighContrast,       ///< High contrast theme
        Custom,             ///< Custom theme
        System              ///< System default theme
    };

    /**
     * @brief Accessibility compliance levels
     */
    enum class AccessibilityLevel {
        WCAG_A,             ///< WCAG 2.1 Level A
        WCAG_AA,            ///< WCAG 2.1 Level AA (recommended)
        WCAG_AAA,           ///< WCAG 2.1 Level AAA
        Section508,         ///< Section 508 compliance
        Custom              ///< Custom accessibility requirements
    };

    /**
     * @brief Color contrast test result
     */
    struct ContrastResult {
        QColor foreground;              ///< Foreground color
        QColor background;              ///< Background color
        double contrastRatio = 0.0;     ///< Calculated contrast ratio
        double requiredRatio = 4.5;     ///< Required contrast ratio
        bool passes = false;            ///< Whether test passes
        AccessibilityLevel level;       ///< Compliance level tested
        QString elementDescription;     ///< Description of UI element
        QWidget* widget = nullptr;      ///< Associated widget
    };

    /**
     * @brief Keyboard navigation test result
     */
    struct KeyboardNavResult {
        bool canReceiveFocus = false;   ///< Widget can receive keyboard focus
        bool tabOrderCorrect = false;   ///< Tab order is logical
        bool shortcutsWork = false;     ///< Keyboard shortcuts function
        bool escapeWorks = false;       ///< Escape key works appropriately
        bool enterWorks = false;        ///< Enter key works appropriately
        QStringList accessibleActions; ///< Available accessible actions
        QString focusIndicator;         ///< Focus indicator description
        QWidget* widget = nullptr;      ///< Associated widget
    };

    /**
     * @brief Screen reader compatibility result
     */
    struct ScreenReaderResult {
        bool hasAccessibleName = false; ///< Has accessible name
        bool hasAccessibleDescription = false; ///< Has accessible description
        bool hasCorrectRole = false;    ///< Has correct accessibility role
        bool stateReported = false;     ///< State changes are reported
        QString accessibleName;         ///< Accessible name text
        QString accessibleDescription;  ///< Accessible description text
        QAccessible::Role role;         ///< Accessibility role
        QAccessible::State state;       ///< Accessibility state
        QWidget* widget = nullptr;      ///< Associated widget
    };

    /**
     * @brief Theme validation result
     */
    struct ThemeValidationResult {
        ThemeType themeType;            ///< Theme type tested
        bool colorsConsistent = false;  ///< Colors are consistent
        bool fontsAppropriate = false;  ///< Fonts are appropriate
        bool sizingCorrect = false;     ///< Widget sizing is correct
        bool iconsVisible = false;      ///< Icons are visible and appropriate
        QMap<QString, QVariant> metrics; ///< Additional theme metrics
        QStringList issues;             ///< List of identified issues
    };

    /**
     * @brief Accessibility test configuration
     */
    struct AccessibilityConfig {
        AccessibilityLevel complianceLevel = AccessibilityLevel::WCAG_AA;
        bool testColorContrast = true;      ///< Test color contrast ratios
        bool testKeyboardNav = true;        ///< Test keyboard navigation
        bool testScreenReader = true;       ///< Test screen reader compatibility
        bool testFocusManagement = true;    ///< Test focus management
        bool testAriaLabels = true;         ///< Test ARIA labels and roles
        double minContrastRatio = 4.5;     ///< Minimum contrast ratio
        double minLargeTextRatio = 3.0;     ///< Minimum ratio for large text
        QStringList exemptWidgets;          ///< Widgets exempt from testing
        QMap<QString, QVariant> customRules; ///< Custom accessibility rules
    };

    explicit ThemeAccessibilityTesting(QObject* parent = nullptr);
    ~ThemeAccessibilityTesting();

    // Configuration
    void setAccessibilityConfig(const AccessibilityConfig& config);
    AccessibilityConfig getAccessibilityConfig() const;
    void setUIAutomation(UIAutomation* uiAutomation);
    void setVisualTesting(VisualTesting* visualTesting);

    // Theme testing
    bool switchToTheme(ThemeType themeType);
    bool switchToTheme(const QString& themeName);
    ThemeValidationResult validateCurrentTheme();
    QMap<ThemeType, ThemeValidationResult> validateAllThemes(QWidget* testWidget);
    bool compareThemeVisuals(ThemeType theme1, ThemeType theme2, QWidget* widget);

    // Color contrast testing
    ContrastResult testColorContrast(QWidget* widget);
    ContrastResult testColorContrast(const QColor& foreground, const QColor& background, 
                                   AccessibilityLevel level = AccessibilityLevel::WCAG_AA);
    QList<ContrastResult> testAllColorContrasts(QWidget* rootWidget);
    bool validateColorContrastCompliance(QWidget* rootWidget, AccessibilityLevel level = AccessibilityLevel::WCAG_AA);

    // Keyboard navigation testing
    KeyboardNavResult testKeyboardNavigation(QWidget* widget);
    bool testTabOrder(const QList<QWidget*>& widgets);
    bool testKeyboardShortcuts(QWidget* widget, const QMap<QKeySequence, QString>& shortcuts);
    bool testFocusManagement(QWidget* widget);
    QList<KeyboardNavResult> testAllKeyboardNavigation(QWidget* rootWidget);

    // Screen reader compatibility testing
    ScreenReaderResult testScreenReaderCompatibility(QWidget* widget);
    bool testAccessibleNames(QWidget* rootWidget);
    bool testAccessibleRoles(QWidget* rootWidget);
    bool testAccessibleStates(QWidget* rootWidget);
    QList<ScreenReaderResult> testAllScreenReaderCompatibility(QWidget* rootWidget);

    // Comprehensive accessibility testing
    bool runFullAccessibilityAudit(QWidget* rootWidget);
    QMap<QString, QVariant> generateAccessibilityReport(QWidget* rootWidget);
    bool validateWCAGCompliance(QWidget* rootWidget, AccessibilityLevel level = AccessibilityLevel::WCAG_AA);

    // Theme switching and validation
    QStringList getAvailableThemes() const;
    QString getCurrentTheme() const;
    bool applyThemeToWidget(QWidget* widget, ThemeType themeType);
    QPixmap captureThemeScreenshot(QWidget* widget, ThemeType themeType);

    // Utility methods
    static double calculateContrastRatio(const QColor& color1, const QColor& color2);
    static double calculateRelativeLuminance(const QColor& color);
    static bool isLargeText(const QFont& font);
    static QAccessibleInterface* getAccessibleInterface(QWidget* widget);
    static QString describeAccessibilityRole(QAccessible::Role role);
    static QString describeAccessibilityState(const QAccessible::State& state);

    // Color analysis utilities
    QColor extractForegroundColor(QWidget* widget) const;
    QColor extractBackgroundColor(QWidget* widget) const;
    QList<QColor> extractAllColors(QWidget* widget) const;
    bool isColorBlindnessFriendly(const QList<QColor>& colors) const;

    // Font and sizing analysis
    bool validateFontSizes(QWidget* rootWidget) const;
    bool validateMinimumSizes(QWidget* rootWidget) const;
    QMap<QString, QFont> analyzeFontUsage(QWidget* rootWidget) const;

signals:
    void themeChanged(ThemeType themeType, const QString& themeName);
    void accessibilityTestCompleted(const QString& testName, bool passed);
    void contrastTestFailed(const ContrastResult& result);
    void keyboardNavTestFailed(const KeyboardNavResult& result);
    void screenReaderTestFailed(const ScreenReaderResult& result);
    void accessibilityAuditCompleted(int totalTests, int passedTests, int failedTests);

private:
    AccessibilityConfig m_config;
    UIAutomation* m_uiAutomation;
    VisualTesting* m_visualTesting;
    ThemeType m_currentTheme;
    QString m_currentThemeName;
    
    // Theme management
    QMap<ThemeType, QString> m_themeNames;
    QMap<QString, QPalette> m_themePalettes;
    QPalette m_originalPalette;
    
    // Internal helper methods
    void initializeThemes();
    void restoreOriginalTheme();
    QPalette createThemePalette(ThemeType themeType) const;
    void applyPaletteToApplication(const QPalette& palette);
    
    // Color analysis helpers
    QColor getEffectiveColor(QWidget* widget, QPalette::ColorRole role) const;
    QList<QPair<QColor, QColor>> findColorPairs(QWidget* widget) const;
    bool meetsContrastRequirement(double ratio, AccessibilityLevel level, bool isLargeText = false) const;
    
    // Keyboard navigation helpers
    bool canWidgetReceiveFocus(QWidget* widget) const;
    QList<QWidget*> getFocusableWidgets(QWidget* rootWidget) const;
    bool testTabSequence(const QList<QWidget*>& widgets) const;
    bool simulateKeyboardNavigation(QWidget* startWidget, const QList<QWidget*>& expectedOrder) const;
    
    // Screen reader helpers
    QAccessibleInterface* createAccessibleInterface(QWidget* widget) const;
    bool hasValidAccessibleName(QAccessibleInterface* interface) const;
    bool hasValidAccessibleRole(QAccessibleInterface* interface) const;
    bool reportsStateChanges(QWidget* widget) const;
    
    // Validation helpers
    bool isWidgetExempt(QWidget* widget) const;
    QString generateWidgetDescription(QWidget* widget) const;
    void logAccessibilityIssue(const QString& issue, QWidget* widget) const;
};

/**
 * @brief Theme manager for automated theme switching during testing
 */
class ThemeManager : public QObject {
    Q_OBJECT

public:
    explicit ThemeManager(QObject* parent = nullptr);

    // Theme registration and management
    void registerTheme(const QString& name, const QPalette& palette);
    void registerTheme(const QString& name, const QString& styleSheet);
    bool loadThemeFromFile(const QString& name, const QString& filePath);
    QStringList getRegisteredThemes() const;

    // Theme application
    bool applyTheme(const QString& name);
    bool applyThemeToWidget(const QString& name, QWidget* widget);
    void restoreDefaultTheme();

    // Theme utilities
    QPalette getCurrentPalette() const;
    QString getCurrentStyleSheet() const;
    bool saveTheme(const QString& name, const QString& filePath) const;

signals:
    void themeApplied(const QString& themeName);
    void themeRestored();

private:
    QMap<QString, QPalette> m_palettes;
    QMap<QString, QString> m_styleSheets;
    QPalette m_defaultPalette;
    QString m_defaultStyleSheet;
    QString m_currentTheme;
};

/**
 * @brief Convenience macros for theme and accessibility testing
 */
#define ACCESSIBILITY_VERIFY(widget) \
    do { \
        if (!themeAccessibilityTesting.runFullAccessibilityAudit(widget)) { \
            QFAIL("Accessibility audit failed"); \
        } \
    } while(0)

#define CONTRAST_VERIFY(widget, level) \
    do { \
        if (!themeAccessibilityTesting.validateColorContrastCompliance(widget, level)) { \
            QFAIL("Color contrast validation failed"); \
        } \
    } while(0)

#define KEYBOARD_NAV_VERIFY(widget) \
    do { \
        auto result = themeAccessibilityTesting.testKeyboardNavigation(widget); \
        if (!result.canReceiveFocus || !result.tabOrderCorrect) { \
            QFAIL("Keyboard navigation test failed"); \
        } \
    } while(0)

#define THEME_SWITCH_VERIFY(themeType, widget) \
    do { \
        if (!themeAccessibilityTesting.switchToTheme(themeType)) { \
            QFAIL("Failed to switch theme"); \
        } \
        auto result = themeAccessibilityTesting.validateCurrentTheme(); \
        if (!result.colorsConsistent || !result.fontsAppropriate) { \
            QFAIL("Theme validation failed"); \
        } \
    } while(0)