#ifndef THEME_MANAGER_H
#define THEME_MANAGER_H

#include <QObject>
#include <QWidget>
#include <QDialog>
#include <QApplication>
#include <QStyleFactory>
#include <QPalette>
#include <QSettings>
#include <QPointer>
#include <QList>
#include <QSize>
#include <QMap>
#include <QString>
#include <QDateTime>
#include <QColor>
#include <QJsonObject>
#include <QMutex>

// Forward declarations for UI types
class QPushButton;
class QCheckBox;
class QLabel;
class QTreeWidget;
class QComboBox;
class QLineEdit;

// Forward declarations
class ThemeEditor;
class ComponentRegistry;
class StyleValidator;
class ThemeErrorHandler;
class ThemePerformanceOptimizer;

// Theme data structures
struct ThemeData {
    QString name;
    QString description;
    QDateTime created;
    QDateTime modified;
    
    struct ColorScheme {
        QColor background{255, 255, 255};
        QColor foreground{0, 0, 0};
        QColor accent{0, 120, 215};
        QColor border{200, 200, 200};
        QColor hover{230, 230, 230};
        QColor disabled{150, 150, 150};
        QColor success{40, 167, 69};
        QColor warning{255, 193, 7};
        QColor error{220, 53, 69};
        QColor info{23, 162, 184};
    } colors;
    
    struct Typography {
        QString fontFamily{"Segoe UI, Ubuntu, sans-serif"};
        int baseFontSize{9};
        int titleFontSize{11};
        int smallFontSize{8};
        bool boldTitles{true};
    } typography;
    
    struct Spacing {
        int padding{8};
        int margin{4};
        int borderRadius{4};
        int borderWidth{1};
    } spacing;
    
    // Validation methods
    bool isValid() const;
    bool meetsAccessibilityStandards() const;
    double getContrastRatio(const QColor& fg, const QColor& bg) const;
};

struct StyleViolation {
    QString componentName;
    QString fileName;
    int lineNumber;
    QString violationType;  // "hardcoded-color", "inline-style", "accessibility"
    QString currentValue;
    QString suggestedFix;
    QString severity;       // "critical", "warning", "info"
};

struct ValidationResult {
    bool isCompliant;
    QList<StyleViolation> violations;
    double accessibilityScore;
    QString summary;
};

struct ComplianceReport {
    QDateTime generated;
    int totalComponents;
    int compliantComponents;
    int violationCount;
    QList<StyleViolation> criticalViolations;
    QList<StyleViolation> warnings;
    double overallScore;
    QString recommendations;
};

class ThemeManager : public QObject
{
    Q_OBJECT

public:
    enum Theme {
        SystemDefault,
        Light,
        Dark,
        HighContrast,
        Custom
    };
    
    enum class ComponentType {
        Dialog, Widget, ProgressBar, CheckBox, Button,
        LineEdit, ComboBox, Label, TreeView, TableView,
        GroupBox, TabWidget, RadioButton, Slider, ScrollBar
    };

    enum class ProgressType {
        Normal,
        Success,
        Warning,
        Error,
        Performance,
        Queue
    };

    enum class StatusType {
        Success,
        Warning,
        Error,
        Info,
        Neutral
    };

    enum class ControlType {
        Button,
        LineEdit,
        ComboBox,
        CheckBox,
        RadioButton,
        Label,
        GroupBox,
        TabWidget,
        ProgressBar,
        Slider
    };

    static ThemeManager* instance();
    
    // Core theme management
    Q_INVOKABLE void setTheme(Theme theme, const QString& customThemeName = QString());
    Theme currentTheme() const { return m_currentTheme; }
    QString currentThemeName() const;
    QString currentThemeString() const;
    
    // Component registration and styling
    void registerComponent(QWidget* component, ComponentType type);
    void unregisterComponent(QWidget* component);
    QString getComponentStyle(ComponentType type) const;
    void applyThemeToComponent(QWidget* component);
    
    // Style application
    QString getApplicationStyleSheet() const;
    void applyToWidget(QWidget* widget);
    void applyToDialog(QDialog* dialog);
    void applyToApplication();
    
    // Dialog registration for automatic theme updates
    void registerDialog(QDialog* dialog);
    void unregisterDialog(QDialog* dialog);
    
    // Component-specific styling methods
    QString getProgressBarStyle(ProgressType type = ProgressType::Normal) const;
    QString getStatusIndicatorStyle(StatusType status) const;
    QString getCustomWidgetStyle(const QString& widgetClass) const;
    
    // Theme editing and customization
    ThemeEditor* createThemeEditor(QWidget* parent = nullptr);
    bool saveCustomTheme(const QString& name, const ThemeData& themeData);
    QStringList getCustomThemeNames() const;
    ThemeData getThemeData(const QString& themeName) const;
    bool deleteCustomTheme(const QString& name);
    
    // Validation and compliance
    ValidationResult validateThemeCompliance(QWidget* component);
    QList<StyleViolation> detectHardcodedStyles(QWidget* component);
    bool performAccessibilityValidation(const ThemeData& theme);
    ComplianceReport generateComplianceReport();
    
    // Enhanced validation system
    void enableRuntimeValidation(bool enabled = true);
    void setValidationScanInterval(int milliseconds = 5000);
    ComplianceReport performComprehensiveValidation();
    QList<StyleViolation> scanAllComponents();
    void generateDetailedValidationReport(const QString& outputPath = QString());
    
    // Source code validation
    QList<StyleViolation> validateSourceCode(const QString& sourceDirectory = QString());
    void logValidationResults(const QList<StyleViolation>& violations);
    
    // Validation statistics
    int getValidationScansPerformed() const;
    QDateTime getLastValidationScan() const;
    QStringList getValidationSummary() const;
    
    // Minimum size management
    QSize getMinimumControlSize(ControlType control) const;
    void enforceMinimumSizes(QWidget* parent);
    
    // Convenience methods to reduce duplicate styling code (Task 2.1.6)
    // These methods combine common operations into single calls
    
    /**
     * @brief Apply theme-aware style and minimum size to a button
     * @param button Button to style
     */
    void styleButton(QPushButton* button);
    
    /**
     * @brief Apply theme-aware style and minimum size to multiple buttons
     * @param buttons List of buttons to style
     */
    void styleButtons(const QList<QPushButton*>& buttons);
    
    /**
     * @brief Apply theme-aware style and minimum size to a checkbox
     * @param checkbox Checkbox to style
     */
    void styleCheckBox(QCheckBox* checkbox);
    
    /**
     * @brief Apply theme-aware style and minimum size to multiple checkboxes
     * @param checkboxes List of checkboxes to style
     */
    void styleCheckBoxes(const QList<QCheckBox*>& checkboxes);
    
    /**
     * @brief Apply theme-aware style to a label
     * @param label Label to style
     */
    void styleLabel(QLabel* label);
    
    /**
     * @brief Apply theme-aware style to a tree widget
     * @param tree Tree widget to style
     */
    void styleTreeWidget(QTreeWidget* tree);
    
    /**
     * @brief Apply theme-aware style to a combo box
     * @param combo Combo box to style
     */
    void styleComboBox(QComboBox* combo);
    
    /**
     * @brief Apply theme-aware style to a line edit
     * @param lineEdit Line edit to style
     */
    void styleLineEdit(QLineEdit* lineEdit);
    
    // Style override and validation (legacy methods)
    void removeHardcodedStyles(QWidget* widget);
    
    // Comprehensive validation system (legacy methods)
    void validateApplicationCompliance();
    QStringList scanForHardcodedStyles();
    
    // Comprehensive theme testing
    void performThemeComplianceTest();
    bool testThemeSwitching();
    
    // Enhanced dialog management
    void applyComprehensiveTheme(QDialog* dialog);
    void registerCustomWidget(QWidget* widget, const QString& styleClass);
    
    // Theme detection
    bool isSystemDarkMode() const;
    bool isHighContrastModeEnabled() const;
    bool isSystemHighContrastMode() const;
    
    // Accessibility features
    void enableHighContrastMode(bool enabled = true);
    void enableEnhancedFocusIndicators(bool enabled = true);
    void setMinimumContrastRatio(double ratio = 4.5);
    double getMinimumContrastRatio() const;
    bool validateAccessibilityCompliance() const;
    QStringList getAccessibilityViolations() const;
    void applyAccessibilityEnhancements(QWidget* widget);
    
    // Focus indicator management
    void enableFocusIndicators(QWidget* widget, bool enabled = true);
    void setFocusIndicatorStyle(const QString& style);
    QString getFocusIndicatorStyle() const;
    
    // Alternative indicators for color-only information
    void enableAlternativeIndicators(bool enabled = true);
    bool hasAlternativeIndicators() const;
    void addIconIndicator(QWidget* widget, const QString& iconPath, const QString& description);
    void addTextIndicator(QWidget* widget, const QString& text);
    
    // Keyboard navigation and shortcuts
    void setupAccessibleTabOrder(QWidget* parent);
    void setupAccessibleKeyboardShortcuts(QWidget* parent);
    void enableKeyboardNavigation(QWidget* widget, bool enabled = true);
    
    // Theme data access
    ThemeData getCurrentThemeData() const;
    ThemeData getHighContrastThemeData() const;
    
    // Persistence
    void saveThemePreference();
    void loadThemePreference();
    
    // Real-time theme updates
    void enableRealTimeThemeUpdates(bool enabled = true);
    void setThemeUpdateInterval(int milliseconds = 100);
    void enableComponentMonitoring(bool enabled = true);
    
    // Error recovery
    void attemptThemeRecovery();
    QStringList getFailedThemeComponents() const;
    
    // Performance optimization
    void enablePerformanceOptimization(bool enabled = true);
    void enableStyleSheetCaching(bool enabled = true);
    void enableBatchUpdates(bool enabled = true);
    void setPerformanceTarget(int maxSwitchTimeMs = 100);
    qint64 getLastThemeSwitchTime() const;
    qint64 getAverageThemeSwitchTime() const;
    int getCacheHitRate() const;
    QString generatePerformanceReport() const;
    void resetPerformanceMetrics();

signals:
    void themeChanged(Theme theme, const QString& themeName);
    void themeValidationCompleted(const ComplianceReport& report);
    void componentRegistered(QWidget* component);
    void componentUnregistered(QWidget* component);

private slots:
    void onSystemThemeChanged();
    void propagateThemeChangeWithRecovery();

private:
    explicit ThemeManager(QObject* parent = nullptr);
    ~ThemeManager() = default;
    
    // Style generation
    QString generateLightThemeStyles() const;
    QString generateDarkThemeStyles() const;
    QString generateCommonStyles() const;
    QString generateStyleSheet(const ThemeData& theme) const;
    void validateAndApplyTheme(const ThemeData& theme);
    
    // Component-specific style generation
    QString generateProgressBarStyles(ProgressType type, Theme theme) const;
    QString generateStatusIndicatorStyles(StatusType status, Theme theme) const;
    QString generateCustomWidgetStyles(const QString& widgetClass, Theme theme) const;
    
    // Minimum size configuration
    void initializeMinimumSizes();
    QSize getDefaultMinimumSize(ControlType control) const;
    
    // System theme detection
    void setupSystemThemeDetection();
    Theme detectSystemTheme() const;
    
    // Internal methods
    void updateRegisteredDialogs();
    void propagateThemeChange();
    void handleSystemThemeChange();
    QString generateComponentStyleSheet(ComponentType type, const ThemeData& theme) const;
    QString generateHighContrastThemeStyles() const;
    QString generateFocusIndicatorStyles() const;
    
    // Theme persistence helpers
    QString getThemeStoragePath() const;
    QString getPreferencesKey() const;
    QJsonObject themeToJson(const ThemeData& theme) const;
    ThemeData themeFromJson(const QJsonObject& json) const;
    ThemeData getDefaultTheme(Theme theme) const;
    
    static ThemeManager* s_instance;
    Theme m_currentTheme;
    QString m_currentCustomThemeName;
    bool m_followSystemTheme;
    
    // Dialog tracking for automatic theme updates
    QList<QPointer<QDialog>> m_registeredDialogs;
    
    // Component styling registry
    QMap<QString, QString> m_lightCustomStyles;
    QMap<QString, QString> m_darkCustomStyles;
    QMap<ControlType, QSize> m_minimumSizes;
    QList<QPointer<QWidget>> m_registeredCustomWidgets;
    
    // Component registry
    ComponentRegistry* m_componentRegistry;
    
    // Style validator
    StyleValidator* m_styleValidator;
    
    // Performance optimizer
    ThemePerformanceOptimizer* m_performanceOptimizer;
    
    // Accessibility features
    bool m_highContrastModeEnabled;
    bool m_enhancedFocusIndicatorsEnabled;
    bool m_alternativeIndicatorsEnabled;
    double m_minimumContrastRatio;
    QString m_focusIndicatorStyle;
    QMap<QWidget*, QString> m_widgetIconIndicators;
    QMap<QWidget*, QString> m_widgetTextIndicators;
    
    // Prevent infinite theme application loop
    bool m_isApplyingTheme;
};

// Helper macros for theme-aware styling
#define THEME_AWARE_STYLE(lightStyle, darkStyle) \
    (ThemeManager::instance()->currentTheme() == ThemeManager::Dark ? (darkStyle) : (lightStyle))

#define APPLY_THEME_TO_WIDGET(widget) \
    ThemeManager::instance()->applyToWidget(widget)

#endif // THEME_MANAGER_H