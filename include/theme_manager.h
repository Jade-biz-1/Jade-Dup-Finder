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

class ThemeManager : public QObject
{
    Q_OBJECT

public:
    enum Theme {
        SystemDefault,
        Light,
        Dark
    };

    enum class ProgressType {
        Normal,
        Success,
        Warning,
        Error,
        Performance
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
    
    // Theme management
    void setTheme(Theme theme);
    Theme currentTheme() const { return m_currentTheme; }
    QString currentThemeString() const;
    
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
    
    // Minimum size management
    QSize getMinimumControlSize(ControlType control) const;
    void enforceMinimumSizes(QWidget* parent);
    
    // Style override and validation
    void removeHardcodedStyles(QWidget* widget);
    void validateThemeCompliance(QWidget* widget);
    
    // Comprehensive validation system
    void validateApplicationCompliance();
    QStringList scanForHardcodedStyles();
    void generateComplianceReport();
    
    // Comprehensive theme testing
    void performThemeComplianceTest();
    bool testThemeSwitching();
    
    // Enhanced dialog management
    void applyComprehensiveTheme(QDialog* dialog);
    void registerCustomWidget(QWidget* widget, const QString& styleClass);
    
    // Theme detection
    bool isSystemDarkMode() const;
    
    // Settings integration
    void loadFromSettings();
    void saveToSettings();

signals:
    void themeChanged(Theme newTheme);

private slots:
    void onSystemThemeChanged();

private:
    explicit ThemeManager(QObject* parent = nullptr);
    ~ThemeManager() = default;
    
    // Style generation
    QString generateLightThemeStyles() const;
    QString generateDarkThemeStyles() const;
    QString generateCommonStyles() const;
    
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
    
    static ThemeManager* s_instance;
    Theme m_currentTheme;
    bool m_followSystemTheme;
    
    // Dialog tracking for automatic theme updates
    QList<QPointer<QDialog>> m_registeredDialogs;
    
    // Component styling registry
    QMap<QString, QString> m_lightCustomStyles;
    QMap<QString, QString> m_darkCustomStyles;
    QMap<ControlType, QSize> m_minimumSizes;
    QList<QPointer<QWidget>> m_registeredCustomWidgets;
};

// Helper macros for theme-aware styling
#define THEME_AWARE_STYLE(lightStyle, darkStyle) \
    (ThemeManager::instance()->currentTheme() == ThemeManager::Dark ? (darkStyle) : (lightStyle))

#define APPLY_THEME_TO_WIDGET(widget) \
    ThemeManager::instance()->applyToWidget(widget)

#endif // THEME_MANAGER_H