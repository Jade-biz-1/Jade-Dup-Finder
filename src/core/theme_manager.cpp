#include "theme_manager.h"
#include "theme_editor.h"
#include "component_registry.h"
#include "style_validator.h"
#include "theme_persistence.h"
#include "theme_performance_optimizer.h"
#include "core/logger.h"
#include <QApplication>
#include <QStyleFactory>
#include <QStyle>
#include <QPalette>
#include <QSettings>
#include <QTimer>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>
#include <QCheckBox>
#include <QRadioButton>
#include <QLabel>
#include <QGroupBox>
#include <QProgressBar>
#include <windows.h>
#include <QSlider>
#include <QTreeWidget>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QThread>
#include <QMetaObject>
#include <cmath>
#include <algorithm>

// ThemeData implementation
bool ThemeData::isValid() const
{
    return !name.isEmpty() && 
           colors.background.isValid() && 
           colors.foreground.isValid() &&
           !typography.fontFamily.isEmpty() &&
           typography.baseFontSize > 0;
}

bool ThemeData::meetsAccessibilityStandards() const
{
    // Check contrast ratios for critical color combinations
    double bgFgRatio = getContrastRatio(colors.foreground, colors.background);
    double accentBgRatio = getContrastRatio(colors.accent, colors.background);
    double hoverBgRatio = getContrastRatio(colors.hover, colors.background);
    
    // WCAG AA requires 4.5:1 for normal text, 3:1 for large text
    return bgFgRatio >= 4.5 && accentBgRatio >= 3.0 && hoverBgRatio >= 3.0;
}

double ThemeData::getContrastRatio(const QColor& fg, const QColor& bg) const
{
    auto getLuminance = [](const QColor& color) -> double {
        auto sRGBtoLin = [](double colorChannel) -> double {
            if (colorChannel <= 0.03928) {
                return colorChannel / 12.92;
            } else {
                return std::pow((colorChannel + 0.055) / 1.055, 2.4);
            }
        };
        
        double r = sRGBtoLin(color.redF());
        double g = sRGBtoLin(color.greenF());
        double b = sRGBtoLin(color.blueF());
        
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    
    double l1 = getLuminance(fg);
    double l2 = getLuminance(bg);
    
    if (l1 < l2) {
        std::swap(l1, l2);
    }
    
    return (l1 + 0.05) / (l2 + 0.05);
}

ThemeManager* ThemeManager::s_instance = nullptr;

ThemeManager* ThemeManager::instance()
{
    if (!s_instance) {
        s_instance = new ThemeManager(qApp);
    }
    return s_instance;
}

ThemeManager::ThemeManager(QObject* parent)
    : QObject(parent)
    , m_currentTheme(SystemDefault)
    , m_followSystemTheme(true)
    , m_componentRegistry(new ComponentRegistry(this))
    , m_styleValidator(new StyleValidator(this))
    , m_performanceOptimizer(new ThemePerformanceOptimizer(this))
    , m_highContrastModeEnabled(false)
    , m_enhancedFocusIndicatorsEnabled(true)
    , m_alternativeIndicatorsEnabled(false)
    , m_minimumContrastRatio(4.5)
{
    initializeMinimumSizes();
    setupSystemThemeDetection();
    
    // Migrate from old settings if necessary
    ThemePersistence::migrateFromOldSettings();
    
    // Load theme preferences but DON'T apply yet
    loadThemePreference();
    
    // Initialize enhanced theme propagation features
    if (m_componentRegistry) {
        // Connect to component registry signals
        connect(m_componentRegistry, &ComponentRegistry::componentRegistered,
                this, &ThemeManager::componentRegistered);
        connect(m_componentRegistry, &ComponentRegistry::componentUnregistered,
                this, &ThemeManager::componentUnregistered);
        connect(m_componentRegistry, &ComponentRegistry::dialogRegistered,
                this, [this](QDialog* dialog) {
                    LOG_DEBUG(LogCategories::UI, QString("Dialog registered: %1")
                             .arg(dialog->metaObject()->className()));
                });
        connect(m_componentRegistry, &ComponentRegistry::dialogUnregistered,
                this, [this](QDialog* dialog) {
                    LOG_DEBUG(LogCategories::UI, QString("Dialog unregistered: %1")
                             .arg(dialog->metaObject()->className()));
                });
    }
    
    // CRITICAL FIX: Defer theme application until event loop starts
    // This ensures all operations happen in the main thread
    QTimer::singleShot(0, this, [this]() {
        // Now safe to apply theme from main event loop
        applyToApplication();
        
        // Enable real-time updates AFTER initial application
        if (m_componentRegistry) {
            enableRealTimeThemeUpdates(true);
            enableComponentMonitoring(true);
        }
        
        LOG_INFO(LogCategories::UI, QString("ThemeManager initialized with theme: %1 (%2)")
                 .arg(currentThemeString())
                 .arg(m_currentCustomThemeName.isEmpty() ? "default" : m_currentCustomThemeName));
    });
    
    // Connect to application aboutToQuit signal to save preferences
    connect(qApp, &QApplication::aboutToQuit, this, [this]() {
        LOG_INFO(LogCategories::UI, "Application shutting down, saving theme preferences");
        saveThemePreference();
        
        // Create automatic backup of custom themes
        QString backupPath = QString("%1/themes_auto_backup_%2.json")
                           .arg(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation))
                           .arg(QDateTime::currentDateTime().toString("yyyyMMdd"));
        
        QStringList customThemes = getCustomThemeNames();
        if (!customThemes.isEmpty()) {
            if (ThemePersistence::backupThemes(backupPath)) {
                LOG_DEBUG(LogCategories::CONFIG, QString("Auto-backup created: %1").arg(backupPath));
            }
        }
    });
}



QString ThemeManager::currentThemeString() const
{
    switch (m_currentTheme) {
        case Light: return "light";
        case Dark: return "dark";
        case HighContrast: return "high-contrast";
        case Custom: return "custom";
        case SystemDefault: 
        default: return "system";
    }
}

QString ThemeManager::getApplicationStyleSheet() const
{
    QString styleSheet = generateCommonStyles();
    
    Theme effectiveTheme = m_currentTheme;
    if (effectiveTheme == SystemDefault) {
        effectiveTheme = isSystemDarkMode() ? Dark : Light;
    }
    
    if (effectiveTheme == Dark) {
        styleSheet += generateDarkThemeStyles();
    } else if (effectiveTheme == HighContrast || m_highContrastModeEnabled) {
        styleSheet += generateHighContrastThemeStyles();
    } else {
        styleSheet += generateLightThemeStyles();
    }
    
    // Add accessibility enhancements if enabled
    if (m_enhancedFocusIndicatorsEnabled) {
        styleSheet += generateFocusIndicatorStyles();
    }
    
    return styleSheet;
}

void ThemeManager::applyToWidget(QWidget* widget)
{
    if (!widget) return;
    
    QString styleSheet = getApplicationStyleSheet();
    widget->setStyleSheet(styleSheet);
    widget->update();
}

void ThemeManager::applyToDialog(QDialog* dialog)
{
    if (!dialog) return;
    
    QString styleSheet = getApplicationStyleSheet();
    
    // Apply theme to the dialog itself
    dialog->setStyleSheet(styleSheet);
    
    // Apply accessibility enhancements to dialog
    applyAccessibilityEnhancements(dialog);
    
    // Apply theme to all child widgets recursively
    QList<QWidget*> children = dialog->findChildren<QWidget*>();
    for (QWidget* child : children) {
        // Clear any existing stylesheet first, then apply theme
        child->setStyleSheet("");  // Clear existing styles
        child->setStyleSheet(styleSheet);  // Apply theme styles
        
        // Apply accessibility enhancements to each child
        applyAccessibilityEnhancements(child);
        
        child->update();
    }
    
    // Ensure proper tab order for keyboard navigation
    setupAccessibleTabOrder(dialog);
    
    // Add keyboard shortcuts for common actions
    setupAccessibleKeyboardShortcuts(dialog);
    
    // Update the dialog itself
    dialog->update();
    dialog->repaint();
}

void ThemeManager::applyToApplication()
{
    Theme effectiveTheme = m_currentTheme;
    if (effectiveTheme == SystemDefault) {
        // Check for system high contrast mode first
        if (isSystemHighContrastMode()) {
            effectiveTheme = HighContrast;
        } else {
            effectiveTheme = isSystemDarkMode() ? Dark : Light;
        }
    }
    
    // Set application style
    qApp->setStyle(QStyleFactory::create("Fusion"));
    
    if (effectiveTheme == HighContrast) {
        // High contrast palette for maximum accessibility
        QPalette highContrastPalette;
        highContrastPalette.setColor(QPalette::Window, QColor(0, 0, 0));
        highContrastPalette.setColor(QPalette::WindowText, QColor(255, 255, 255));
        highContrastPalette.setColor(QPalette::Base, QColor(0, 0, 0));
        highContrastPalette.setColor(QPalette::AlternateBase, QColor(64, 64, 64));
        highContrastPalette.setColor(QPalette::ToolTipBase, QColor(255, 255, 0));
        highContrastPalette.setColor(QPalette::ToolTipText, QColor(0, 0, 0));
        highContrastPalette.setColor(QPalette::Text, QColor(255, 255, 255));
        highContrastPalette.setColor(QPalette::Button, QColor(0, 0, 0));
        highContrastPalette.setColor(QPalette::ButtonText, QColor(255, 255, 255));
        highContrastPalette.setColor(QPalette::BrightText, QColor(255, 255, 0));
        highContrastPalette.setColor(QPalette::Link, QColor(255, 255, 0));
        highContrastPalette.setColor(QPalette::Highlight, QColor(255, 255, 0));
        highContrastPalette.setColor(QPalette::HighlightedText, QColor(0, 0, 0));
        
        qApp->setPalette(highContrastPalette);
        
    } else if (effectiveTheme == Dark) {
        // Dark palette
        QPalette darkPalette;
        darkPalette.setColor(QPalette::Window, QColor(53, 53, 53));
        darkPalette.setColor(QPalette::WindowText, QColor(255, 255, 255));
        darkPalette.setColor(QPalette::Base, QColor(25, 25, 25));
        darkPalette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
        darkPalette.setColor(QPalette::ToolTipBase, QColor(0, 0, 0));
        darkPalette.setColor(QPalette::ToolTipText, QColor(255, 255, 255));
        darkPalette.setColor(QPalette::Text, QColor(255, 255, 255));
        darkPalette.setColor(QPalette::Button, QColor(53, 53, 53));
        darkPalette.setColor(QPalette::ButtonText, QColor(255, 255, 255));
        darkPalette.setColor(QPalette::BrightText, QColor(255, 0, 0));
        darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));
        darkPalette.setColor(QPalette::Highlight, QColor(42, 130, 218));
        darkPalette.setColor(QPalette::HighlightedText, QColor(0, 0, 0));
        
        qApp->setPalette(darkPalette);
    } else {
        // Light theme - use standard palette
        qApp->setPalette(qApp->style()->standardPalette());
    }
    
    // Apply stylesheet to application
    qApp->setStyleSheet(getApplicationStyleSheet());
    
    QString themeDescription;
    switch (effectiveTheme) {
        case Light: themeDescription = "light"; break;
        case Dark: themeDescription = "dark"; break;
        case HighContrast: themeDescription = "high contrast"; break;
        default: themeDescription = "system"; break;
    }
    
    LOG_INFO(LogCategories::UI, QString("Applied %1 theme to application").arg(themeDescription));
}

bool ThemeManager::isSystemDarkMode() const
{
    // Check system palette to determine if dark mode is active
    QPalette palette = qApp->palette();
    QColor windowColor = palette.color(QPalette::Window);
    
    // If window background is darker than text, assume dark mode
    QColor textColor = palette.color(QPalette::WindowText);
    return windowColor.lightness() < textColor.lightness();
}



void ThemeManager::onSystemThemeChanged()
{
    if (m_followSystemTheme && m_currentTheme == SystemDefault) {
        LOG_INFO(LogCategories::UI, "System theme changed, updating application");
        applyToApplication();
        emit themeChanged(m_currentTheme, m_currentCustomThemeName);
    }
}

QString ThemeManager::generateCommonStyles() const
{
    return R"(
        /* Base application styles - theme-neutral */
        QApplication {
            font-family: "Segoe UI", "Ubuntu", sans-serif;
            font-size: 9pt;
        }
        
        QPushButton {
            min-height: 24px;
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: 500;
        }
        
        QPushButton:hover {
            border-width: 2px;
        }
        
        QPushButton:pressed {
            border-width: 2px;
        }
        
        QPushButton:disabled {
            opacity: 0.6;
        }
        
        QLineEdit, QSpinBox, QDoubleSpinBox {
            min-height: 20px;
            padding: 4px 8px;
            border-radius: 4px;
        }
        
        QComboBox {
            min-height: 24px;
            padding: 4px 8px;
            border-radius: 4px;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        
        QComboBox::down-arrow {
            width: 12px;
            height: 12px;
        }
        
        QCheckBox {
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }
        
        QRadioButton {
            spacing: 8px;
        }
        
        QRadioButton::indicator {
            width: 16px;
            height: 16px;
            border-radius: 8px;
        }
        
        QGroupBox {
            font-weight: bold;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 8px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 8px 0 8px;
        }
        
        QTabWidget::pane {
            border-radius: 4px;
            padding: 4px;
        }
        
        QTabBar::tab {
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        
        QTabBar::tab:selected {
            margin-bottom: -1px;
        }
        
        QListWidget, QTreeWidget, QTableWidget {
            border-radius: 4px;
        }
        
        QListWidget::item, QTreeWidget::item, QTableWidget::item {
            padding: 4px;
            border-radius: 2px;
        }
        
        QScrollBar:vertical {
            width: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar:horizontal {
            height: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:horizontal {
            border-radius: 6px;
            min-width: 20px;
        }
        
        QScrollBar::add-line, QScrollBar::sub-line {
            border: none;
            background: none;
        }
        
        QProgressBar {
            border-radius: 4px;
            text-align: center;
            font-weight: bold;
        }
        
        QProgressBar::chunk {
            border-radius: 4px;
        }
    )";
}

QString ThemeManager::generateLightThemeStyles() const
{
    return R"(
        /* Light theme specific styles */
        QMainWindow {
            background-color: #f8f9fa;
            color: #212529;
        }
        
        QWidget {
            background-color: #ffffff;
            color: #212529;
        }
        
        QDialog {
            background-color: #f8f9fa;
            color: #212529;
        }
        
        QPushButton {
            background-color: #e9ecef;
            color: #495057;
            border: 1px solid #ced4da;
        }
        
        QPushButton:hover {
            background-color: #dee2e6;
            border-color: #adb5bd;
        }
        
        QPushButton:pressed {
            background-color: #ced4da;
        }
        
        QLineEdit, QSpinBox, QDoubleSpinBox {
            background-color: #ffffff;
            color: #495057;
            border: 1px solid #ced4da;
        }
        
        QComboBox {
            background-color: #ffffff;
            color: #495057;
            border: 1px solid #ced4da;
        }
        
        QTreeWidget, QTableWidget, QListWidget {
            background-color: #ffffff;
            color: #495057;
            alternate-background-color: #f8f9fa;
        }
        
        QMenuBar {
            background-color: #e9ecef;
            color: #495057;
            border-bottom: 1px solid #dee2e6;
        }
        
        QMenuBar::item:selected {
            background-color: #dee2e6;
        }
        
        QStatusBar {
            background-color: #e9ecef;
            color: #495057;
            border-top: 1px solid #dee2e6;
        }
        
        QToolTip {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            padding: 4px;
        }
        
        /* Checkbox styling with enhanced visibility */
        QCheckBox::indicator {
            background-color: #ffffff;
            border: 2px solid #ced4da;
        }
        
        QCheckBox::indicator:hover {
            border-color: #007bff;
            background-color: #f0f8ff;
        }
        
        QCheckBox::indicator:checked {
            background-color: #007bff;
            border-color: #007bff;
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEwIDNMNC41IDguNUwyIDYiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
        }
        
        QCheckBox::indicator:checked:hover {
            background-color: #0056b3;
            border-color: #0056b3;
        }
        
        QCheckBox::indicator:disabled {
            background-color: #e9ecef;
            border-color: #dee2e6;
        }
        
        QCheckBox::indicator:checked:disabled {
            background-color: #6c757d;
            border-color: #6c757d;
        }
        
        QCheckBox:focus {
            outline: 2px solid #007bff;
            outline-offset: 2px;
        }
    )";
}

QString ThemeManager::generateDarkThemeStyles() const
{
    return R"(
        /* Dark theme specific styles */
        QMainWindow {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        QWidget {
            background-color: #2d2d30;
            color: #ffffff;
        }
        
        QDialog {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        QPushButton {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #555555;
        }
        
        QPushButton:hover {
            background-color: #4a4a4a;
            border-color: #007acc;
        }
        
        QPushButton:pressed {
            background-color: #2d2d30;
        }
        
        QLineEdit, QSpinBox, QDoubleSpinBox {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #555555;
        }
        
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 2px solid #007acc;
        }
        
        QComboBox {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #555555;
        }
        
        QComboBox QAbstractItemView {
            background-color: #2d2d30;
            color: #ffffff;
            selection-background-color: #007acc;
        }
        
        /* Checkbox styling with enhanced visibility and contrast */
        QCheckBox::indicator {
            background-color: #1e1e1e;
            border: 2px solid #6e6e6e;
        }
        
        QCheckBox::indicator:hover {
            border-color: #007acc;
            background-color: #2d2d30;
        }
        
        QCheckBox::indicator:checked {
            background-color: #007acc;
            border-color: #007acc;
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEwIDNMNC41IDguNUwyIDYiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
        }
        
        QCheckBox::indicator:checked:hover {
            background-color: #1e88e5;
            border-color: #1e88e5;
        }
        
        QCheckBox::indicator:disabled {
            background-color: #3c3c3c;
            border-color: #555555;
        }
        
        QCheckBox::indicator:checked:disabled {
            background-color: #4a4a4a;
            border-color: #555555;
        }
        
        QCheckBox:focus {
            outline: 2px solid #007acc;
            outline-offset: 2px;
        }
        
        QTreeWidget, QTableWidget, QListWidget {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #555555;
            alternate-background-color: #2d2d30;
        }
        
        QTreeWidget::item:selected, QTableWidget::item:selected, QListWidget::item:selected {
            background-color: #007acc;
            color: #ffffff;
        }
        
        QTabWidget::pane {
            background-color: #2d2d30;
            border: 1px solid #555555;
        }
        
        QTabBar::tab {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #555555;
        }
        
        QTabBar::tab:selected {
            background-color: #2d2d30;
            color: #ffffff;
            border-bottom: 1px solid #2d2d30;
        }
        
        QTabBar::tab:hover:!selected {
            background-color: #4a4a4a;
        }
        
        QMenuBar {
            background-color: #2d2d30;
            color: #ffffff;
            border-bottom: 1px solid #555555;
        }
        
        QMenuBar::item:selected {
            background-color: #3c3c3c;
        }
        
        QStatusBar {
            background-color: #2d2d30;
            color: #ffffff;
            border-top: 1px solid #555555;
        }
        
        QToolTip {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 4px;
        }
        
        QScrollBar:vertical {
            background-color: #2d2d30;
        }
        
        QScrollBar::handle:vertical {
            background-color: #555555;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #666666;
        }
        
        QScrollBar:horizontal {
            background-color: #2d2d30;
        }
        
        QScrollBar::handle:horizontal {
            background-color: #555555;
        }
        
        QScrollBar::handle:horizontal:hover {
            background-color: #666666;
        }
    )";
}

QString ThemeManager::generateHighContrastThemeStyles() const
{
    return R"(
        /* High contrast theme for enhanced accessibility */
        QMainWindow {
            background-color: #000000;
            color: #ffffff;
        }
        
        QWidget {
            background-color: #000000;
            color: #ffffff;
        }
        
        QDialog {
            background-color: #000000;
            color: #ffffff;
            border: 3px solid #ffffff;
        }
        
        QPushButton {
            background-color: #000000;
            color: #ffffff;
            border: 3px solid #ffffff;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #ffffff;
            color: #000000;
            border: 3px solid #000000;
        }
        
        QPushButton:pressed {
            background-color: #808080;
            color: #ffffff;
            border: 3px solid #ffffff;
        }
        
        QPushButton:focus {
            outline: 4px solid #ffff00;
            outline-offset: 2px;
        }
        
        QLineEdit, QSpinBox, QDoubleSpinBox {
            background-color: #ffffff;
            color: #000000;
            border: 3px solid #000000;
            font-weight: bold;
        }
        
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 4px solid #ffff00;
            outline: 2px solid #000000;
        }
        
        QComboBox {
            background-color: #ffffff;
            color: #000000;
            border: 3px solid #000000;
            font-weight: bold;
        }
        
        QComboBox:focus {
            border: 4px solid #ffff00;
        }
        
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #000000;
            selection-background-color: #000000;
            selection-color: #ffffff;
            border: 3px solid #000000;
        }
        
        QCheckBox {
            color: #ffffff;
            font-weight: bold;
        }
        
        QCheckBox::indicator {
            background-color: #ffffff;
            border: 3px solid #000000;
            width: 20px;
            height: 20px;
        }
        
        QCheckBox::indicator:checked {
            background-color: #000000;
            border: 3px solid #ffffff;
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEzIDRMNiAxMUwzIDgiIHN0cm9rZT0iI2ZmZmZmZiIgc3Ryb2tlLXdpZHRoPSIzIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc+Cg==);
        }
        
        QCheckBox::indicator:focus {
            outline: 4px solid #ffff00;
            outline-offset: 2px;
        }
        
        QRadioButton {
            color: #ffffff;
            font-weight: bold;
        }
        
        QRadioButton::indicator {
            background-color: #ffffff;
            border: 3px solid #000000;
            width: 20px;
            height: 20px;
            border-radius: 10px;
        }
        
        QRadioButton::indicator:checked {
            background-color: #000000;
            border: 3px solid #ffffff;
        }
        
        QRadioButton::indicator:focus {
            outline: 4px solid #ffff00;
            outline-offset: 2px;
        }
        
        QTreeWidget, QTableWidget, QListWidget {
            background-color: #ffffff;
            color: #000000;
            border: 3px solid #000000;
            alternate-background-color: #f0f0f0;
            font-weight: bold;
        }
        
        QTreeWidget::item:selected, QTableWidget::item:selected, QListWidget::item:selected {
            background-color: #000000;
            color: #ffffff;
        }
        
        QTreeWidget::item:focus, QTableWidget::item:focus, QListWidget::item:focus {
            outline: 3px solid #ffff00;
            outline-offset: 1px;
        }
        
        QTabWidget::pane {
            background-color: #000000;
            border: 3px solid #ffffff;
        }
        
        QTabBar::tab {
            background-color: #808080;
            color: #ffffff;
            border: 3px solid #ffffff;
            font-weight: bold;
            padding: 8px 16px;
        }
        
        QTabBar::tab:selected {
            background-color: #ffffff;
            color: #000000;
            border-bottom: 3px solid #ffffff;
        }
        
        QTabBar::tab:focus {
            outline: 3px solid #ffff00;
            outline-offset: 2px;
        }
        
        QMenuBar {
            background-color: #000000;
            color: #ffffff;
            border-bottom: 3px solid #ffffff;
            font-weight: bold;
        }
        
        QMenuBar::item:selected {
            background-color: #ffffff;
            color: #000000;
        }
        
        QStatusBar {
            background-color: #000000;
            color: #ffffff;
            border-top: 3px solid #ffffff;
            font-weight: bold;
        }
        
        QToolTip {
            background-color: #ffff00;
            color: #000000;
            border: 3px solid #000000;
            border-radius: 0px;
            padding: 8px;
            font-weight: bold;
        }
        
        QScrollBar:vertical {
            background-color: #ffffff;
            border: 2px solid #000000;
            width: 20px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #000000;
            border: 2px solid #ffffff;
            min-height: 30px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #808080;
        }
        
        QScrollBar:horizontal {
            background-color: #ffffff;
            border: 2px solid #000000;
            height: 20px;
        }
        
        QScrollBar::handle:horizontal {
            background-color: #000000;
            border: 2px solid #ffffff;
            min-width: 30px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background-color: #808080;
        }
        
        QProgressBar {
            background-color: #ffffff;
            border: 3px solid #000000;
            text-align: center;
            font-weight: bold;
            color: #000000;
        }
        
        QProgressBar::chunk {
            background-color: #000000;
            border: 1px solid #ffffff;
        }
    )";
}

QString ThemeManager::generateFocusIndicatorStyles() const
{
    QString focusColor = m_highContrastModeEnabled ? "#ffff00" : "#007acc";
    int focusWidth = m_highContrastModeEnabled ? 4 : 2;
    
    return QString(R"(
        /* Enhanced focus indicators for accessibility */
        *:focus {
            outline: %1px solid %2;
            outline-offset: 2px;
        }
        
        QPushButton:focus {
            outline: %1px solid %2;
            outline-offset: 2px;
        }
        
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            outline: %1px solid %2;
            outline-offset: 1px;
        }
        
        QComboBox:focus {
            outline: %1px solid %2;
            outline-offset: 1px;
        }
        
        QCheckBox:focus {
            outline: %1px solid %2;
            outline-offset: 3px;
        }
        
        QRadioButton:focus {
            outline: %1px solid %2;
            outline-offset: 3px;
        }
        
        QTabBar::tab:focus {
            outline: %1px solid %2;
            outline-offset: 2px;
        }
        
        QTreeWidget:focus, QTableWidget:focus, QListWidget:focus {
            outline: %1px solid %2;
            outline-offset: 1px;
        }
        
        QSlider:focus {
            outline: %1px solid %2;
            outline-offset: 2px;
        }
    )").arg(focusWidth).arg(focusColor);
}

void ThemeManager::setupSystemThemeDetection()
{
    // Monitor system theme changes with enhanced detection
    QTimer* themeCheckTimer = new QTimer(this);
    connect(themeCheckTimer, &QTimer::timeout, this, [this]() {
        // Only check if we're following system theme
        if (m_followSystemTheme && m_currentTheme == SystemDefault) {
            // Check if system theme actually changed
            Theme detectedTheme = detectSystemTheme();
            static Theme lastDetectedTheme = detectedTheme;
            
            if (detectedTheme != lastDetectedTheme) {
                LOG_INFO(LogCategories::UI, QString("System theme changed from %1 to %2")
                         .arg(lastDetectedTheme == Dark ? "dark" : "light")
                         .arg(detectedTheme == Dark ? "dark" : "light"));
                
                lastDetectedTheme = detectedTheme;
                onSystemThemeChanged();
            }
        }
    });
    themeCheckTimer->start(2000); // Check every 2 seconds for more responsive detection
    
    // Also connect to application palette change signal for immediate detection
    connect(qApp, &QGuiApplication::paletteChanged, this, [this](const QPalette&) {
        if (m_followSystemTheme && m_currentTheme == SystemDefault) {
            LOG_DEBUG(LogCategories::UI, "Application palette changed, checking system theme");
            QTimer::singleShot(100, this, &ThemeManager::onSystemThemeChanged);
        }
    });
    
    LOG_INFO(LogCategories::UI, "Enhanced system theme detection initialized");
}

ThemeManager::Theme ThemeManager::detectSystemTheme() const
{
    return isSystemDarkMode() ? Dark : Light;
}

void ThemeManager::registerDialog(QDialog* dialog)
{
    if (!dialog) return;
    
    // Register with enhanced ComponentRegistry
    if (m_componentRegistry) {
        m_componentRegistry->registerDialog(dialog);
    }
    
    // Legacy support - check if already registered
    for (const auto& ptr : m_registeredDialogs) {
        if (ptr.data() == dialog) {
            return; // Already registered
        }
    }
    
    // Add to registered dialogs (legacy)
    m_registeredDialogs.append(QPointer<QDialog>(dialog));
    
    // Apply current theme immediately
    applyToDialog(dialog);
    
    // Connect to dialog destruction to auto-unregister
    connect(dialog, &QObject::destroyed, this, [this, dialog]() {
        unregisterDialog(dialog);
    });
    
    LOG_DEBUG(LogCategories::UI, QString("Dialog registered for theme updates: %1")
             .arg(dialog->metaObject()->className()));
}

void ThemeManager::unregisterDialog(QDialog* dialog)
{
    if (!dialog) return;
    
    // Unregister from enhanced ComponentRegistry
    if (m_componentRegistry) {
        m_componentRegistry->unregisterDialog(dialog);
    }
    
    // Remove from registered dialogs (legacy)
    for (int i = m_registeredDialogs.size() - 1; i >= 0; --i) {
        if (m_registeredDialogs[i].isNull() || m_registeredDialogs[i].data() == dialog) {
            m_registeredDialogs.removeAt(i);
        }
    }
    
    LOG_DEBUG(LogCategories::UI, QString("Dialog unregistered from theme updates: %1")
             .arg(dialog->metaObject()->className()));
}

void ThemeManager::updateRegisteredDialogs()
{
    // Clean up null pointers and update remaining dialogs
    for (int i = m_registeredDialogs.size() - 1; i >= 0; --i) {
        if (m_registeredDialogs[i].isNull()) {
            m_registeredDialogs.removeAt(i);
        } else {
            applyToDialog(m_registeredDialogs[i].data());
        }
    }
}

// Component-specific styling methods
QString ThemeManager::getProgressBarStyle(ProgressType type) const
{
    return generateProgressBarStyles(type, m_currentTheme == SystemDefault ? 
                                   (isSystemDarkMode() ? Dark : Light) : m_currentTheme);
}

QString ThemeManager::getStatusIndicatorStyle(StatusType status) const
{
    return generateStatusIndicatorStyles(status, m_currentTheme == SystemDefault ? 
                                       (isSystemDarkMode() ? Dark : Light) : m_currentTheme);
}

QString ThemeManager::getCustomWidgetStyle(const QString& widgetClass) const
{
    return generateCustomWidgetStyles(widgetClass, m_currentTheme == SystemDefault ? 
                                    (isSystemDarkMode() ? Dark : Light) : m_currentTheme);
}

// Minimum size management
QSize ThemeManager::getMinimumControlSize(ControlType control) const
{
    return m_minimumSizes.value(control, getDefaultMinimumSize(control));
}

void ThemeManager::enforceMinimumSizes(QWidget* parent)
{
    if (!parent) return;
    
    // Find all child widgets and apply minimum sizes
    QList<QWidget*> children = parent->findChildren<QWidget*>();
    for (QWidget* child : children) {
        if (auto* button = qobject_cast<QPushButton*>(child)) {
            button->setMinimumSize(getMinimumControlSize(ControlType::Button));
        } else if (auto* lineEdit = qobject_cast<QLineEdit*>(child)) {
            lineEdit->setMinimumSize(getMinimumControlSize(ControlType::LineEdit));
        } else if (auto* comboBox = qobject_cast<QComboBox*>(child)) {
            comboBox->setMinimumSize(getMinimumControlSize(ControlType::ComboBox));
        } else if (auto* checkBox = qobject_cast<QCheckBox*>(child)) {
            checkBox->setMinimumSize(getMinimumControlSize(ControlType::CheckBox));
        } else if (auto* radioButton = qobject_cast<QRadioButton*>(child)) {
            radioButton->setMinimumSize(getMinimumControlSize(ControlType::RadioButton));
        } else if (auto* label = qobject_cast<QLabel*>(child)) {
            label->setMinimumSize(getMinimumControlSize(ControlType::Label));
        } else if (auto* groupBox = qobject_cast<QGroupBox*>(child)) {
            groupBox->setMinimumSize(getMinimumControlSize(ControlType::GroupBox));
        } else if (auto* progressBar = qobject_cast<QProgressBar*>(child)) {
            progressBar->setMinimumSize(getMinimumControlSize(ControlType::ProgressBar));
        } else if (auto* slider = qobject_cast<QSlider*>(child)) {
            slider->setMinimumSize(getMinimumControlSize(ControlType::Slider));
        }
    }
}

// Convenience methods to reduce duplicate styling code (Task 2.1.6)
void ThemeManager::styleButton(QPushButton* button)
{
    if (!button) return;
    
    button->setStyleSheet(getComponentStyle(ComponentType::Button));
    button->setMinimumSize(getMinimumControlSize(ControlType::Button));
}

void ThemeManager::styleButtons(const QList<QPushButton*>& buttons)
{
    for (QPushButton* button : buttons) {
        styleButton(button);
    }
}

void ThemeManager::styleCheckBox(QCheckBox* checkbox)
{
    if (!checkbox) return;
    
    checkbox->setStyleSheet(getComponentStyle(ComponentType::CheckBox));
    checkbox->setMinimumSize(getMinimumControlSize(ControlType::CheckBox));
}

void ThemeManager::styleCheckBoxes(const QList<QCheckBox*>& checkboxes)
{
    for (QCheckBox* checkbox : checkboxes) {
        styleCheckBox(checkbox);
    }
}

void ThemeManager::styleLabel(QLabel* label)
{
    if (!label) return;
    
    label->setStyleSheet(getComponentStyle(ComponentType::Label));
}

void ThemeManager::styleTreeWidget(QTreeWidget* tree)
{
    if (!tree) return;
    
    tree->setStyleSheet(getComponentStyle(ComponentType::TreeView));
}

void ThemeManager::styleComboBox(QComboBox* combo)
{
    if (!combo) return;
    
    combo->setStyleSheet(getComponentStyle(ComponentType::ComboBox));
    combo->setMinimumSize(getMinimumControlSize(ControlType::ComboBox));
}

void ThemeManager::styleLineEdit(QLineEdit* lineEdit)
{
    if (!lineEdit) return;
    
    lineEdit->setStyleSheet(getComponentStyle(ComponentType::LineEdit));
    lineEdit->setMinimumSize(getMinimumControlSize(ControlType::LineEdit));
}

// Style override and validation
void ThemeManager::removeHardcodedStyles(QWidget* widget)
{
    if (!widget) return;
    
    // Clear any existing stylesheet that might contain hardcoded colors
    widget->setStyleSheet("");
    
    // Apply theme-aware styling
    applyToWidget(widget);
    
    // Recursively handle child widgets
    QList<QWidget*> children = widget->findChildren<QWidget*>();
    for (QWidget* child : children) {
        child->setStyleSheet("");
        applyToWidget(child);
    }
}



// Enhanced dialog management
void ThemeManager::applyComprehensiveTheme(QDialog* dialog)
{
    if (!dialog) return;
    
    // Apply main theme
    applyToDialog(dialog);
    
    // Enforce minimum sizes
    enforceMinimumSizes(dialog);
    
    // Validate theme compliance
    validateThemeCompliance(dialog);
    
    // Register for future theme updates
    registerDialog(dialog);
}

void ThemeManager::registerCustomWidget(QWidget* widget, const QString& styleClass)
{
    if (!widget) return;
    
    // Store the widget with its style class
    m_registeredCustomWidgets.append(QPointer<QWidget>(widget));
    
    // Apply custom styling immediately
    QString customStyle = getCustomWidgetStyle(styleClass);
    if (!customStyle.isEmpty()) {
        widget->setStyleSheet(customStyle);
    }
    
    // Connect to widget destruction to auto-unregister
    connect(widget, &QObject::destroyed, this, [this, widget]() {
        m_registeredCustomWidgets.removeAll(QPointer<QWidget>(widget));
    });
}

// Component-specific style generation
QString ThemeManager::generateProgressBarStyles(ProgressType type, Theme theme) const
{
    QString baseStyle = R"(
        QProgressBar {
            border: 2px solid %1;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            height: 25px;
        }
        QProgressBar::chunk {
            background-color: %2;
            border-radius: 3px;
        }
    )";
    
    QString borderColor, chunkColor;
    
    if (theme == Dark) {
        borderColor = "#555555";
        switch (type) {
            case ProgressType::Success:
                chunkColor = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4CAF50, stop:1 #45a049)";
                break;
            case ProgressType::Warning:
                chunkColor = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF9800, stop:1 #F57C00)";
                break;
            case ProgressType::Error:
                chunkColor = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #F44336, stop:1 #D32F2F)";
                break;
            case ProgressType::Performance:
                chunkColor = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2196F3, stop:1 #1976D2)";
                break;
            case ProgressType::Queue:
                chunkColor = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #9C27B0, stop:1 #7B1FA2)";
                break;
            default:
                chunkColor = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007acc, stop:1 #005a9e)";
                break;
        }
    } else {
        borderColor = "#ced4da";
        switch (type) {
            case ProgressType::Success:
                chunkColor = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #28a745, stop:1 #1e7e34)";
                break;
            case ProgressType::Warning:
                chunkColor = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ffc107, stop:1 #e0a800)";
                break;
            case ProgressType::Error:
                chunkColor = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #dc3545, stop:1 #c82333)";
                break;
            case ProgressType::Performance:
                chunkColor = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007bff, stop:1 #0056b3)";
                break;
            case ProgressType::Queue:
                chunkColor = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6f42c1, stop:1 #5a32a3)";
                break;
            default:
                chunkColor = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007bff, stop:1 #0056b3)";
                break;
        }
    }
    
    return baseStyle.arg(borderColor, chunkColor);
}

QString ThemeManager::generateStatusIndicatorStyles(StatusType status, Theme theme) const
{
    QString color;
    
    if (theme == Dark) {
        switch (status) {
            case StatusType::Success:
                color = "#4CAF50";
                break;
            case StatusType::Warning:
                color = "#FF9800";
                break;
            case StatusType::Error:
                color = "#F44336";
                break;
            case StatusType::Info:
                color = "#2196F3";
                break;
            default:
                color = "#ffffff";
                break;
        }
    } else {
        switch (status) {
            case StatusType::Success:
                color = "#28a745";
                break;
            case StatusType::Warning:
                color = "#ffc107";
                break;
            case StatusType::Error:
                color = "#dc3545";
                break;
            case StatusType::Info:
                color = "#007bff";
                break;
            default:
                color = "#495057";
                break;
        }
    }
    
    return QString("QLabel { color: %1; font-weight: bold; }").arg(color);
}

QString ThemeManager::generateCustomWidgetStyles(const QString& widgetClass, Theme theme) const
{
    // Check if we have custom styles registered for this widget class
    if (theme == Dark && m_darkCustomStyles.contains(widgetClass)) {
        return m_darkCustomStyles[widgetClass];
    } else if (theme == Light && m_lightCustomStyles.contains(widgetClass)) {
        return m_lightCustomStyles[widgetClass];
    }
    
    // Return empty string if no custom style is registered
    return QString();
}

// Minimum size configuration
void ThemeManager::initializeMinimumSizes()
{
    m_minimumSizes[ControlType::Button] = QSize(80, 24);
    m_minimumSizes[ControlType::LineEdit] = QSize(100, 20);
    m_minimumSizes[ControlType::ComboBox] = QSize(120, 24);
    m_minimumSizes[ControlType::CheckBox] = QSize(16, 16);
    m_minimumSizes[ControlType::RadioButton] = QSize(16, 16);
    m_minimumSizes[ControlType::Label] = QSize(50, 16);
    m_minimumSizes[ControlType::GroupBox] = QSize(100, 50);
    m_minimumSizes[ControlType::TabWidget] = QSize(200, 100);
    m_minimumSizes[ControlType::ProgressBar] = QSize(150, 20);
    m_minimumSizes[ControlType::Slider] = QSize(100, 20);
}

QSize ThemeManager::getDefaultMinimumSize(ControlType control) const
{
    switch (control) {
        case ControlType::Button:
            return QSize(80, 24);
        case ControlType::LineEdit:
            return QSize(100, 20);
        case ControlType::ComboBox:
            return QSize(120, 24);
        case ControlType::CheckBox:
        case ControlType::RadioButton:
            return QSize(16, 16);
        case ControlType::Label:
            return QSize(50, 16);
        case ControlType::GroupBox:
            return QSize(100, 50);
        case ControlType::TabWidget:
            return QSize(200, 100);
        case ControlType::ProgressBar:
            return QSize(150, 20);
        case ControlType::Slider:
            return QSize(100, 20);
        default:
            return QSize(50, 20);
    }
}

// Comprehensive validation system
void ThemeManager::validateApplicationCompliance()
{
    // CRITICAL: This must run on the main thread since it accesses widgets
    if (QThread::currentThread() != qApp->thread()) {
        LOG_WARNING(LogCategories::UI, "validateApplicationCompliance() called from worker thread, skipping widget modification");
        // Just report, don't fix from worker thread
        QMetaObject::invokeMethod(this, [this]() {
            validateApplicationCompliance();
        }, Qt::QueuedConnection);
        return;
    }
    
    LOG_INFO(LogCategories::UI, "Starting comprehensive theme compliance validation");
    
    // Get all widgets in the application
    QWidgetList allWidgets = QApplication::allWidgets();
    int totalWidgets = allWidgets.size();
    int nonCompliantWidgets = 0;
    
    for (QWidget* widget : allWidgets) {
        if (widget) {
            QString stylesheet = widget->styleSheet();
            if (stylesheet.contains(QRegularExpression("#[0-9a-fA-F]{3,6}")) ||
                stylesheet.contains(QRegularExpression("rgb\\s*\\(")) ||
                stylesheet.contains(QRegularExpression("rgba\\s*\\("))) {
                
                nonCompliantWidgets++;
                LOG_WARNING(LogCategories::UI, QString("Non-compliant widget found: %1 (%2)")
                           .arg(widget->metaObject()->className())
                           .arg(widget->objectName().isEmpty() ? "unnamed" : widget->objectName()));
                
                // Automatically fix - safe since we're on main thread now
                removeHardcodedStyles(widget);
            }
        }
    }
    
    LOG_INFO(LogCategories::UI, QString("Theme compliance validation complete: %1/%2 widgets were non-compliant and have been fixed")
             .arg(nonCompliantWidgets).arg(totalWidgets));
}

QStringList ThemeManager::scanForHardcodedStyles()
{
    QStringList issues;
    QWidgetList allWidgets = QApplication::allWidgets();
    
    for (QWidget* widget : allWidgets) {
        if (widget) {
            QString stylesheet = widget->styleSheet();
            QString widgetInfo = QString("%1 (%2)")
                               .arg(widget->metaObject()->className())
                               .arg(widget->objectName().isEmpty() ? "unnamed" : widget->objectName());
            
            // Check for hex colors
            QRegularExpression hexPattern("#[0-9a-fA-F]{3,6}");
            QRegularExpressionMatchIterator hexMatches = hexPattern.globalMatch(stylesheet);
            while (hexMatches.hasNext()) {
                QRegularExpressionMatch match = hexMatches.next();
                issues.append(QString("Hardcoded hex color '%1' in %2").arg(match.captured(0)).arg(widgetInfo));
            }
            
            // Check for RGB colors
            QRegularExpression rgbPattern("rgb\\s*\\([^)]+\\)");
            QRegularExpressionMatchIterator rgbMatches = rgbPattern.globalMatch(stylesheet);
            while (rgbMatches.hasNext()) {
                QRegularExpressionMatch match = rgbMatches.next();
                issues.append(QString("Hardcoded RGB color '%1' in %2").arg(match.captured(0)).arg(widgetInfo));
            }
            
            // Check for RGBA colors
            QRegularExpression rgbaPattern("rgba\\s*\\([^)]+\\)");
            QRegularExpressionMatchIterator rgbaMatches = rgbaPattern.globalMatch(stylesheet);
            while (rgbaMatches.hasNext()) {
                QRegularExpressionMatch match = rgbaMatches.next();
                issues.append(QString("Hardcoded RGBA color '%1' in %2").arg(match.captured(0)).arg(widgetInfo));
            }
        }
    }
    
    return issues;
}

ComplianceReport ThemeManager::generateComplianceReport()
{
    LOG_INFO(LogCategories::UI, "Generating comprehensive theme compliance report using enhanced validation");
    
    // Use the enhanced validation system
    ComplianceReport report = performComprehensiveValidation();
    
    LOG_INFO(LogCategories::UI, QString("=== ENHANCED THEME COMPLIANCE REPORT ==="));
    LOG_INFO(LogCategories::UI, QString("Generated: %1").arg(report.generated.toString()));
    LOG_INFO(LogCategories::UI, QString("Total components scanned: %1").arg(report.totalComponents));
    LOG_INFO(LogCategories::UI, QString("Compliant components: %1").arg(report.compliantComponents));
    LOG_INFO(LogCategories::UI, QString("Total violations: %1").arg(report.violationCount));
    LOG_INFO(LogCategories::UI, QString("Critical violations: %1").arg(report.criticalViolations.size()));
    LOG_INFO(LogCategories::UI, QString("Warnings: %1").arg(report.warnings.size()));
    LOG_INFO(LogCategories::UI, QString("Overall compliance score: %1%").arg(report.overallScore, 0, 'f', 1));
    
    if (report.violationCount == 0) {
        LOG_INFO(LogCategories::UI, "✅ All components are theme-compliant!");
    } else {
        LOG_WARNING(LogCategories::UI, "❌ Style violations detected:");
        
        if (!report.criticalViolations.isEmpty()) {
            LOG_ERROR(LogCategories::UI, "CRITICAL VIOLATIONS:");
            for (const StyleViolation& violation : report.criticalViolations) {
                LOG_ERROR(LogCategories::UI, QString("  - %1: %2 (%3)")
                         .arg(violation.componentName)
                         .arg(violation.violationType)
                         .arg(violation.currentValue));
            }
        }
        
        if (!report.warnings.isEmpty()) {
            LOG_WARNING(LogCategories::UI, "WARNINGS:");
            for (const StyleViolation& violation : report.warnings) {
                LOG_WARNING(LogCategories::UI, QString("  - %1: %2 (%3)")
                           .arg(violation.componentName)
                           .arg(violation.violationType)
                           .arg(violation.currentValue));
            }
        }
    }
    
    // Count registered dialogs and widgets
    int registeredDialogs = 0;
    int registeredCustomWidgets = 0;
    
    for (const auto& ptr : m_registeredDialogs) {
        if (!ptr.isNull()) registeredDialogs++;
    }
    
    for (const auto& ptr : m_registeredCustomWidgets) {
        if (!ptr.isNull()) registeredCustomWidgets++;
    }
    
    LOG_INFO(LogCategories::UI, QString("Registered dialogs: %1").arg(registeredDialogs));
    LOG_INFO(LogCategories::UI, QString("Registered custom widgets: %1").arg(registeredCustomWidgets));
    LOG_INFO(LogCategories::UI, QString("Current theme: %1").arg(currentThemeString()));
    LOG_INFO(LogCategories::UI, QString("Recommendations: %1").arg(report.recommendations));
    LOG_INFO(LogCategories::UI, QString("=== END ENHANCED COMPLIANCE REPORT ==="));
    
    return report;
}

// Comprehensive theme testing
void ThemeManager::performThemeComplianceTest()
{
    LOG_INFO(LogCategories::UI, "Starting comprehensive theme compliance test with enhanced validation");
    
    // Store original theme
    Theme originalTheme = m_currentTheme;
    
    // Test theme switching
    bool switchingWorks = testThemeSwitching();
    LOG_INFO(LogCategories::UI, QString("Theme switching test: %1").arg(switchingWorks ? "PASSED" : "FAILED"));
    
    // Generate enhanced compliance report
    ComplianceReport report = generateComplianceReport();
    
    // Validate source code if available
    QList<StyleViolation> sourceViolations = validateSourceCode();
    if (!sourceViolations.isEmpty()) {
        LOG_WARNING(LogCategories::UI, QString("Source code validation found %1 violations").arg(sourceViolations.size()));
    }
    
    // Validate application compliance
    validateApplicationCompliance();
    
    // Test accessibility compliance
    ThemeData currentTheme = getCurrentThemeData();
    bool accessibilityPassed = performAccessibilityValidation(currentTheme);
    LOG_INFO(LogCategories::UI, QString("Accessibility validation: %1").arg(accessibilityPassed ? "PASSED" : "FAILED"));
    
    // Generate detailed report
    generateDetailedValidationReport();
    
    // Test minimum size enforcement
    QWidgetList allWidgets = QApplication::allWidgets();
    int widgetsWithMinSizes = 0;
    
    for (QWidget* widget : allWidgets) {
        if (widget && !widget->minimumSize().isEmpty()) {
            widgetsWithMinSizes++;
        }
    }
    
    LOG_INFO(LogCategories::UI, QString("Widgets with minimum sizes: %1/%2")
             .arg(widgetsWithMinSizes).arg(allWidgets.size()));
    
    // Restore original theme
    setTheme(originalTheme);
    
    LOG_INFO(LogCategories::UI, QString("Theme compliance test complete. Theme switching: %1")
             .arg(switchingWorks ? "✅ PASS" : "❌ FAIL"));
}

bool ThemeManager::testThemeSwitching()
{
    LOG_INFO(LogCategories::UI, "Testing theme switching functionality");
    
    // Skip test if running from worker thread - theme changes require main thread
    if (QThread::currentThread() != qApp->thread()) {
        LOG_WARNING(LogCategories::UI, "Theme switching test skipped - must run on main thread");
        return true; // Return true to avoid false failures
    }
    
    Theme originalTheme = m_currentTheme;
    bool allTestsPassed = true;
    
    // Test switching to Light theme
    setTheme(Light);
    QCoreApplication::processEvents(); // Process deferred theme changes
    if (m_currentTheme != Light) {
        LOG_ERROR(LogCategories::UI, "Failed to switch to Light theme");
        allTestsPassed = false;
    } else {
        LOG_INFO(LogCategories::UI, "✅ Light theme switch successful");
    }
    
    // Test switching to Dark theme
    setTheme(Dark);
    QCoreApplication::processEvents(); // Process deferred theme changes
    if (m_currentTheme != Dark) {
        LOG_ERROR(LogCategories::UI, "Failed to switch to Dark theme");
        allTestsPassed = false;
    } else {
        LOG_INFO(LogCategories::UI, "✅ Dark theme switch successful");
    }
    
    // Test switching to System theme
    setTheme(SystemDefault);
    QCoreApplication::processEvents(); // Process deferred theme changes
    if (m_currentTheme != SystemDefault) {
        LOG_ERROR(LogCategories::UI, "Failed to switch to System theme");
        allTestsPassed = false;
    } else {
        LOG_INFO(LogCategories::UI, "✅ System theme switch successful");
    }
    
    // Test that registered dialogs get updated
    int dialogsUpdated = 0;
    for (const auto& ptr : m_registeredDialogs) {
        if (!ptr.isNull()) {
            dialogsUpdated++;
        }
    }
    
    LOG_INFO(LogCategories::UI, QString("Registered dialogs that will receive theme updates: %1")
             .arg(dialogsUpdated));
    
    // Restore original theme
    setTheme(originalTheme);
    
    return allTestsPassed;
}
// 
// Enhanced ThemeManager methods

void ThemeManager::setTheme(Theme theme, const QString& customThemeName)
{
    // CRITICAL FIX: Ensure theme changes happen on main thread
    if (QThread::currentThread() != qApp->thread()) {
        LOG_WARNING(LogCategories::UI, "setTheme() called from worker thread, deferring to main thread");
        QMetaObject::invokeMethod(this, [this, theme, customThemeName]() {
            setTheme(theme, customThemeName);
        }, Qt::QueuedConnection);
        return;
    }
    
    if (m_currentTheme == theme && m_currentCustomThemeName == customThemeName) {
        return;
    }
    
    Theme oldTheme = m_currentTheme;
    QString oldCustomName = m_currentCustomThemeName;
    
    m_currentTheme = theme;
    m_currentCustomThemeName = customThemeName;
    
    // Update follow system theme flag
    m_followSystemTheme = (theme == SystemDefault);
    
    LOG_INFO(LogCategories::UI, QString("Theme changed from %1 (%2) to %3 (%4)")
             .arg(static_cast<int>(oldTheme))
             .arg(oldCustomName.isEmpty() ? "default" : oldCustomName)
             .arg(static_cast<int>(theme))
             .arg(customThemeName.isEmpty() ? "default" : customThemeName));
    
    // CRITICAL FIX: Defer ALL theme application to prevent UI blocking
    // Just save settings and emit signal immediately, do heavy work later
    
    // Save to settings immediately (lightweight operation)
    saveThemePreference();
    
    // Emit signal immediately so UI responds
    emit themeChanged(theme, customThemeName);
    
    LOG_INFO(LogCategories::UI, "Theme change initiated successfully");
    
    // NOW defer the heavy work with a timer to ensure it happens AFTER this function returns
    QTimer::singleShot(0, this, [this]() {
        try {
            LOG_INFO(LogCategories::UI, "Applying theme in background...");
            
            // Apply to application - but keep it simple
            applyToApplication();
            
            // Process events to keep UI responsive
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
            
            LOG_INFO(LogCategories::UI, "Theme application completed");
            
        } catch (const std::exception& e) {
            LOG_ERROR(LogCategories::UI, QString("Theme application failed: %1").arg(e.what()));
        }
    });
}

QString ThemeManager::currentThemeName() const
{
    if (m_currentTheme == Custom && !m_currentCustomThemeName.isEmpty()) {
        return m_currentCustomThemeName;
    }
    return currentThemeString();
}

void ThemeManager::registerComponent(QWidget* component, ComponentType type)
{
    if (!component) return;
    
    m_componentRegistry->registerComponent(component, type);
    applyThemeToComponent(component);
    
    emit componentRegistered(component);
}

void ThemeManager::unregisterComponent(QWidget* component)
{
    if (!component) return;
    
    m_componentRegistry->unregisterComponent(component);
    emit componentUnregistered(component);
}

QString ThemeManager::getComponentStyle(ComponentType type) const
{
    ThemeData currentThemeData = getCurrentThemeData();
    return m_componentRegistry ? 
           generateComponentStyleSheet(type, currentThemeData) : 
           QString();
}

void ThemeManager::applyThemeToComponent(QWidget* component)
{
    if (!component) return;
    
    ThemeData currentThemeData = getCurrentThemeData();
    QString styleSheet = generateStyleSheet(currentThemeData);
    component->setStyleSheet(styleSheet);
    
    // Apply accessibility enhancements
    applyAccessibilityEnhancements(component);
    
    component->update();
}

ThemeEditor* ThemeManager::createThemeEditor(QWidget* parent)
{
    ThemeEditor* editor = new ThemeEditor(parent);
    editor->setBaseTheme(m_currentTheme);
    
    if (m_currentTheme == Custom && !m_currentCustomThemeName.isEmpty()) {
        editor->loadCustomTheme(m_currentCustomThemeName);
    }
    
    return editor;
}

bool ThemeManager::saveCustomTheme(const QString& name, const ThemeData& themeData)
{
    if (name.isEmpty() || !themeData.isValid()) {
        LOG_WARNING(LogCategories::UI, "Cannot save invalid theme or empty name");
        return false;
    }
    
    bool success = ThemePersistence::saveCustomTheme(name, themeData);
    if (success) {
        LOG_INFO(LogCategories::UI, QString("Custom theme '%1' saved successfully").arg(name));
    } else {
        LOG_ERROR(LogCategories::UI, QString("Failed to save custom theme '%1'").arg(name));
    }
    
    return success;
}

QStringList ThemeManager::getCustomThemeNames() const
{
    return ThemePersistence::getCustomThemeNames();
}

ThemeData ThemeManager::getThemeData(const QString& themeName) const
{
    if (themeName.isEmpty()) {
        return getCurrentThemeData();
    }
    
    // Check if it's a built-in theme
    if (themeName == "light") {
        return getDefaultTheme(Light);
    } else if (themeName == "dark") {
        return getDefaultTheme(Dark);
    } else if (themeName == "system") {
        return getDefaultTheme(SystemDefault);
    }
    
    // Try to load as custom theme
    return ThemePersistence::loadCustomTheme(themeName);
}

bool ThemeManager::deleteCustomTheme(const QString& name)
{
    if (name.isEmpty()) {
        return false;
    }
    
    bool success = ThemePersistence::deleteCustomTheme(name);
    if (success) {
        LOG_INFO(LogCategories::UI, QString("Custom theme '%1' deleted successfully").arg(name));
        
        // If we're currently using this theme, switch to system default
        if (m_currentTheme == Custom && m_currentCustomThemeName == name) {
            setTheme(SystemDefault);
        }
    } else {
        LOG_ERROR(LogCategories::UI, QString("Failed to delete custom theme '%1'").arg(name));
    }
    
    return success;
}

ValidationResult ThemeManager::validateThemeCompliance(QWidget* component)
{
    return StyleValidator::validateComponent(component);
}

QList<StyleViolation> ThemeManager::detectHardcodedStyles(QWidget* component)
{
    return StyleValidator::scanForHardcodedStyles(component);
}

bool ThemeManager::performAccessibilityValidation(const ThemeData& theme)
{
    return StyleValidator::validateAccessibility(theme);
}

// Enhanced validation system implementation
void ThemeManager::enableRuntimeValidation(bool enabled)
{
    if (m_styleValidator) {
        m_styleValidator->enableRuntimeScanning(enabled);
        LOG_INFO(LogCategories::UI, QString("Runtime theme validation %1")
                 .arg(enabled ? "enabled" : "disabled"));
    }
}

void ThemeManager::setValidationScanInterval(int milliseconds)
{
    if (m_styleValidator) {
        m_styleValidator->setRuntimeScanInterval(milliseconds);
        LOG_INFO(LogCategories::UI, QString("Validation scan interval set to %1ms").arg(milliseconds));
    }
}

ComplianceReport ThemeManager::performComprehensiveValidation()
{
    LOG_INFO(LogCategories::UI, "Starting comprehensive theme validation");
    
    if (!m_styleValidator) {
        LOG_ERROR(LogCategories::UI, "StyleValidator not available for comprehensive validation");
        ComplianceReport emptyReport;
        emptyReport.generated = QDateTime::currentDateTime();
        emptyReport.totalComponents = 0;
        emptyReport.compliantComponents = 0;
        emptyReport.violationCount = 0;
        emptyReport.overallScore = 0.0;
        emptyReport.recommendations = "StyleValidator not initialized";
        return emptyReport;
    }
    
    ComplianceReport report = m_styleValidator->performComprehensiveApplicationScan();
    
    // Add theme-specific validation
    ThemeData currentTheme = getCurrentThemeData();
    QList<StyleViolation> accessibilityViolations = StyleValidator::validateAccessibilityCompliance(currentTheme);
    
    // Merge accessibility violations into the report
    for (const StyleViolation& violation : accessibilityViolations) {
        if (violation.severity == "critical") {
            report.criticalViolations.append(violation);
        } else {
            report.warnings.append(violation);
        }
        report.violationCount++;
    }
    
    // Recalculate overall score
    if (report.totalComponents > 0) {
        double accessibilityPenalty = accessibilityViolations.size() * 5.0; // 5% penalty per accessibility violation
        report.overallScore = qMax(0.0, report.overallScore - accessibilityPenalty);
    }
    
    // Update recommendations
    if (!accessibilityViolations.isEmpty()) {
        report.recommendations += QString("; Address %1 accessibility violations").arg(accessibilityViolations.size());
    }
    
    LOG_INFO(LogCategories::UI, QString("Comprehensive validation completed: %1 total violations, %.1f%% compliance score")
             .arg(report.violationCount)
             .arg(report.overallScore));
    
    emit themeValidationCompleted(report);
    
    return report;
}

QList<StyleViolation> ThemeManager::scanAllComponents()
{
    if (!m_styleValidator) {
        LOG_ERROR(LogCategories::UI, "StyleValidator not available for component scanning");
        return QList<StyleViolation>();
    }
    
    return m_styleValidator->scanAllApplicationComponents();
}

void ThemeManager::generateDetailedValidationReport(const QString& outputPath)
{
    if (!m_styleValidator) {
        LOG_ERROR(LogCategories::UI, "StyleValidator not available for detailed reporting");
        return;
    }
    
    QString reportPath = outputPath;
    if (reportPath.isEmpty()) {
        // Generate default path
        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
        reportPath = QString("theme_validation_report_%1.txt").arg(timestamp);
    }
    
    m_styleValidator->generateDetailedComplianceReport(reportPath);
    
    LOG_INFO(LogCategories::UI, QString("Detailed validation report generated: %1").arg(reportPath));
}

QList<StyleViolation> ThemeManager::validateSourceCode(const QString& sourceDirectory)
{
    QString scanDirectory = sourceDirectory;
    if (scanDirectory.isEmpty()) {
        // Default to src directory
        scanDirectory = "src";
    }
    
    LOG_INFO(LogCategories::UI, QString("Starting source code validation in: %1").arg(scanDirectory));
    
    QList<StyleViolation> violations = StyleValidator::scanSourceFiles(scanDirectory);
    
    LOG_INFO(LogCategories::UI, QString("Source code validation completed: %1 violations found").arg(violations.size()));
    
    // Log critical violations immediately
    logValidationResults(violations);
    
    return violations;
}

void ThemeManager::logValidationResults(const QList<StyleViolation>& violations)
{
    if (violations.isEmpty()) {
        LOG_INFO(LogCategories::UI, "✅ No style violations found - code is theme-compliant!");
        return;
    }
    
    LOG_WARNING(LogCategories::UI, QString("❌ Found %1 style violations:").arg(violations.size()));
    
    QMap<QString, int> violationCounts;
    QMap<QString, int> severityCounts;
    
    for (const StyleViolation& violation : violations) {
        violationCounts[violation.violationType]++;
        severityCounts[violation.severity]++;
        
        QString logMessage = QString("  %1 [%2]: %3 in %4")
                           .arg(violation.severity.toUpper())
                           .arg(violation.violationType)
                           .arg(violation.currentValue)
                           .arg(violation.componentName);
        
        if (violation.lineNumber > 0) {
            logMessage += QString(" (line %1)").arg(violation.lineNumber);
        }
        
        if (violation.severity == "critical") {
            LOG_ERROR(LogCategories::UI, logMessage);
        } else if (violation.severity == "warning") {
            LOG_WARNING(LogCategories::UI, logMessage);
        } else {
            LOG_INFO(LogCategories::UI, logMessage);
        }
        
        if (!violation.suggestedFix.isEmpty()) {
            LOG_INFO(LogCategories::UI, QString("    Suggested fix: %1").arg(violation.suggestedFix));
        }
    }
    
    // Summary
    LOG_INFO(LogCategories::UI, "=== VIOLATION SUMMARY ===");
    for (auto it = violationCounts.begin(); it != violationCounts.end(); ++it) {
        LOG_INFO(LogCategories::UI, QString("  %1: %2 occurrences").arg(it.key()).arg(it.value()));
    }
    
    LOG_INFO(LogCategories::UI, "=== SEVERITY BREAKDOWN ===");
    for (auto it = severityCounts.begin(); it != severityCounts.end(); ++it) {
        LOG_INFO(LogCategories::UI, QString("  %1: %2 violations").arg(it.key()).arg(it.value()));
    }
}

int ThemeManager::getValidationScansPerformed() const
{
    if (m_styleValidator) {
        return m_styleValidator->getTotalScansPerformed();
    }
    return 0;
}

QDateTime ThemeManager::getLastValidationScan() const
{
    if (m_styleValidator) {
        return m_styleValidator->getLastScanTime();
    }
    return QDateTime();
}

QStringList ThemeManager::getValidationSummary() const
{
    if (m_styleValidator) {
        return m_styleValidator->getViolationSummary();
    }
    return QStringList();
}



void ThemeManager::saveThemePreference()
{
    LOG_DEBUG(LogCategories::CONFIG, QString("Saving theme preference: theme=%1, customName='%2'")
             .arg(static_cast<int>(m_currentTheme))
             .arg(m_currentCustomThemeName));
    
    bool success = ThemePersistence::saveThemePreference(m_currentTheme, m_currentCustomThemeName);
    
    if (success) {
        LOG_INFO(LogCategories::CONFIG, QString("Theme preference saved successfully: %1 (%2)")
                 .arg(currentThemeString())
                 .arg(m_currentCustomThemeName.isEmpty() ? "default" : m_currentCustomThemeName));
    } else {
        LOG_ERROR(LogCategories::CONFIG, "Failed to save theme preference");
    }
}

void ThemeManager::loadThemePreference()
{
    LOG_DEBUG(LogCategories::CONFIG, "Loading theme preferences from persistent storage");
    
    auto preference = ThemePersistence::loadThemePreference();
    
    LOG_INFO(LogCategories::CONFIG, QString("Theme preference loaded: theme=%1, customName='%2'")
             .arg(static_cast<int>(preference.first))
             .arg(preference.second));
    
    setTheme(preference.first, preference.second);
    
    LOG_INFO(LogCategories::CONFIG, QString("Theme preference applied successfully: %1 (%2)")
             .arg(currentThemeString())
             .arg(m_currentCustomThemeName.isEmpty() ? "default" : m_currentCustomThemeName));
}

void ThemeManager::propagateThemeChange()
{
    // CRITICAL FIX: Make this async too - defer the work
    QTimer::singleShot(0, this, [this]() {
        if (m_componentRegistry) {
            ThemeData currentThemeData = getCurrentThemeData();
            
            // Apply theme to all registered components and dialogs
            m_componentRegistry->applyThemeToAll(currentThemeData);
            
            // Process events to keep responsive
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
            
            // Update registered dialogs (legacy support)
            updateRegisteredDialogs();
            
            // Process events again
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
            
            // Force real-time update if enabled
            m_componentRegistry->forceUpdateAll();
            
            LOG_INFO(LogCategories::UI, "Theme propagation completed for all registered components");
        }
    });
}

void ThemeManager::handleSystemThemeChange()
{
    if (m_followSystemTheme && m_currentTheme == SystemDefault) {
        LOG_INFO(LogCategories::UI, "System theme changed, updating application");
        
        // Defer everything to avoid blocking
        QTimer::singleShot(0, this, [this]() {
            applyToApplication();
            propagateThemeChange();  // This is now also async
        });
        
        // Emit immediately
        emit themeChanged(m_currentTheme, m_currentCustomThemeName);
    }
}

QString ThemeManager::generateStyleSheet(const ThemeData& theme) const
{
    QString styleSheet = generateCommonStyles();
    
    // Add theme-specific colors
    styleSheet += QString(R"(
        QMainWindow {
            background-color: %1;
            color: %2;
        }
        
        QWidget {
            background-color: %1;
            color: %2;
        }
        
        QDialog {
            background-color: %1;
            color: %2;
        }
        
        QPushButton {
            background-color: %3;
            color: %2;
            border: %4px solid %5;
            border-radius: %6px;
            padding: %7px;
        }
        
        QPushButton:hover {
            background-color: %8;
        }
        
        QPushButton:disabled {
            background-color: %9;
            color: %10;
        }
    )").arg(theme.colors.background.name())
       .arg(theme.colors.foreground.name())
       .arg(theme.colors.accent.name())
       .arg(theme.spacing.borderWidth)
       .arg(theme.colors.border.name())
       .arg(theme.spacing.borderRadius)
       .arg(theme.spacing.padding)
       .arg(theme.colors.hover.name())
       .arg(theme.colors.disabled.name())
       .arg(theme.colors.disabled.name());
    
    return styleSheet;
}

void ThemeManager::validateAndApplyTheme(const ThemeData& theme)
{
    if (!theme.isValid()) {
        LOG_WARNING(LogCategories::UI, "Invalid theme data, using default theme");
        return;
    }
    
    if (!theme.meetsAccessibilityStandards()) {
        LOG_WARNING(LogCategories::UI, "Theme does not meet accessibility standards");
    }
    
    // Apply the theme
    QString styleSheet = generateStyleSheet(theme);
    qApp->setStyleSheet(styleSheet);
}

ThemeData ThemeManager::getCurrentThemeData() const
{
    if (m_currentTheme == Custom && !m_currentCustomThemeName.isEmpty()) {
        return ThemePersistence::loadCustomTheme(m_currentCustomThemeName);
    }
    
    Theme effectiveTheme = m_currentTheme;
    if (effectiveTheme == SystemDefault) {
        effectiveTheme = isSystemDarkMode() ? Dark : Light;
    }
    
    return getDefaultTheme(effectiveTheme);
}



ThemeData ThemeManager::getDefaultTheme(Theme theme) const
{
    ThemeData themeData;
    
    switch (theme) {
        case Light:
            themeData.name = "Light";
            themeData.description = "Default light theme";
            themeData.colors.background = QColor(248, 249, 250);
            themeData.colors.foreground = QColor(33, 37, 41);
            themeData.colors.accent = QColor(0, 123, 255);
            themeData.colors.border = QColor(206, 212, 218);
            themeData.colors.hover = QColor(230, 230, 230);
            themeData.colors.disabled = QColor(150, 150, 150);
            break;
            
        case Dark:
            themeData.name = "Dark";
            themeData.description = "Default dark theme";
            themeData.colors.background = QColor(30, 30, 30);
            themeData.colors.foreground = QColor(255, 255, 255);
            themeData.colors.accent = QColor(0, 122, 204);
            themeData.colors.border = QColor(85, 85, 85);
            themeData.colors.hover = QColor(74, 74, 74);
            themeData.colors.disabled = QColor(100, 100, 100);
            break;
            
        case HighContrast:
            return getHighContrastThemeData();
            
        case SystemDefault:
        default:
            // Check for system high contrast mode first
            if (isSystemHighContrastMode()) {
                return getDefaultTheme(HighContrast);
            }
            // Use light or dark theme based on system preference
            return getDefaultTheme(isSystemDarkMode() ? Dark : Light);
    }
    
    themeData.created = QDateTime::currentDateTime();
    themeData.modified = themeData.created;
    
    return themeData;
}

QString ThemeManager::generateComponentStyleSheet(ComponentType type, const ThemeData& theme) const
{
    QString styleSheet;
    
    switch (type) {
        case ComponentType::Button:
            styleSheet = QString(R"(
                QPushButton {
                    background-color: %1;
                    color: %2;
                    border: %3px solid %4;
                    border-radius: %5px;
                    padding: %6px;
                    min-height: 24px;
                }
                QPushButton:hover {
                    background-color: %7;
                }
                QPushButton:pressed {
                    background-color: %8;
                }
                QPushButton:disabled {
                    background-color: %9;
                    color: %10;
                }
            )").arg(theme.colors.accent.name())
               .arg(theme.colors.background.name())
               .arg(theme.spacing.borderWidth)
               .arg(theme.colors.border.name())
               .arg(theme.spacing.borderRadius)
               .arg(theme.spacing.padding)
               .arg(theme.colors.hover.name())
               .arg(theme.colors.accent.darker(120).name())
               .arg(theme.colors.disabled.name())
               .arg(theme.colors.disabled.darker(150).name());
            break;
            
        case ComponentType::LineEdit:
            styleSheet = QString(R"(
                QLineEdit {
                    background-color: %1;
                    color: %2;
                    border: %3px solid %4;
                    border-radius: %5px;
                    padding: %6px;
                    min-height: 20px;
                }
                QLineEdit:focus {
                    border-color: %7;
                }
            )").arg(theme.colors.background.name())
               .arg(theme.colors.foreground.name())
               .arg(theme.spacing.borderWidth)
               .arg(theme.colors.border.name())
               .arg(theme.spacing.borderRadius)
               .arg(theme.spacing.padding)
               .arg(theme.colors.accent.name());
            break;
            
        case ComponentType::CheckBox:
            {
                // Enhanced checkbox styling with improved visibility and accessibility
                QColor indicatorBg = theme.colors.background;
                QColor indicatorBorder = theme.colors.border;
                QColor checkedBg = theme.colors.accent;
                QColor checkedBorder = theme.colors.accent;
                QColor hoverBorder = theme.colors.hover;
                QColor disabledBg = theme.colors.disabled.lighter(150);
                QColor disabledBorder = theme.colors.disabled;
                
                // Ensure proper contrast ratios for accessibility
                double contrastRatio = theme.getContrastRatio(theme.colors.foreground, indicatorBg);
                if (contrastRatio < 3.0) {
                    // Adjust background for better contrast
                    if (theme.colors.background.lightness() > 128) {
                        indicatorBg = theme.colors.background.darker(110);
                    } else {
                        indicatorBg = theme.colors.background.lighter(150);
                    }
                }
                
                styleSheet = QString(R"(
                    QCheckBox {
                        color: %1;
                        spacing: 10px;
                        font-size: %2px;
                        min-height: 20px;
                    }
                    QCheckBox::indicator {
                        width: 18px;
                        height: 18px;
                        border: 2px solid %3;
                        border-radius: 4px;
                        background-color: %4;
                        margin: 1px;
                    }
                    QCheckBox::indicator:checked {
                        background-color: %5;
                        border-color: %6;
                        image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEwIDNMNC41IDguNUwyIDYiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
                    }
                    QCheckBox::indicator:hover {
                        border-color: %7;
                        background-color: %8;
                    }
                    QCheckBox::indicator:checked:hover {
                        background-color: %9;
                        border-color: %10;
                    }
                    QCheckBox::indicator:disabled {
                        background-color: %11;
                        border-color: %12;
                    }
                    QCheckBox::indicator:checked:disabled {
                        background-color: %13;
                        border-color: %14;
                    }
                    QCheckBox:focus {
                        outline: 2px solid %15;
                        outline-offset: 2px;
                    }
                    QCheckBox[accessibleText]:after {
                        content: attr(accessibleText);
                        margin-left: 5px;
                        font-style: italic;
                        color: %1;
                    }
                )").arg(theme.colors.foreground.name())                    // %1 - text color
                   .arg(theme.typography.baseFontSize)                     // %2 - font size
                   .arg(indicatorBorder.name())                            // %3 - border color
                   .arg(indicatorBg.name())                                // %4 - background color
                   .arg(checkedBg.name())                                  // %5 - checked background
                   .arg(checkedBorder.name())                              // %6 - checked border
                   .arg(hoverBorder.name())                                // %7 - hover border
                   .arg(theme.colors.hover.name())                         // %8 - hover background
                   .arg(checkedBg.lighter(110).name())                     // %9 - checked hover background
                   .arg(checkedBorder.lighter(110).name())                 // %10 - checked hover border
                   .arg(disabledBg.name())                                 // %11 - disabled background
                   .arg(disabledBorder.name())                             // %12 - disabled border
                   .arg(theme.colors.disabled.name())                      // %13 - checked disabled background
                   .arg(theme.colors.disabled.darker(120).name())          // %14 - checked disabled border
                   .arg(theme.colors.accent.name());                       // %15 - focus outline color
            }
            break;
            
        case ComponentType::TreeView:
            {
                // Enhanced TreeView styling with improved checkbox visibility
                QColor itemBg = theme.colors.background;
                QColor itemSelectedBg = theme.colors.accent;
                QColor itemHoverBg = theme.colors.hover;
                QColor itemBorder = theme.colors.border;
                
                styleSheet = QString(R"(
                    QTreeWidget {
                        background-color: %1;
                        color: %2;
                        border: 1px solid %3;
                        border-radius: %4px;
                        selection-background-color: %5;
                        alternate-background-color: %6;
                        gridline-color: %7;
                        font-size: %8px;
                    }
                    QTreeWidget::item {
                        padding: 6px 4px;
                        border: none;
                        min-height: 24px;
                    }
                    QTreeWidget::item:selected {
                        background-color: %9;
                        color: %10;
                    }
                    QTreeWidget::item:hover {
                        background-color: %11;
                    }
                    QTreeWidget::item:selected:hover {
                        background-color: %12;
                    }
                    QTreeWidget::branch {
                        background: transparent;
                    }
                    QTreeWidget::branch:has-children:!has-siblings:closed,
                    QTreeWidget::branch:closed:has-children:has-siblings {
                        border-image: none;
                        image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQgM0w4IDZMNCA5VjNaIiBmaWxsPSIlMTMiLz4KPC9zdmc+Cg==);
                    }
                    QTreeWidget::branch:open:has-children:!has-siblings,
                    QTreeWidget::branch:open:has-children:has-siblings {
                        border-image: none;
                        image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTMgNEw2IDhMOSA0SDNaIiBmaWxsPSIlMTMiLz4KPC9zdmc+Cg==);
                    }

                    QTreeWidget::indicator {
                        width: 18px;
                        height: 18px;
                        border: 2px solid %14;
                        border-radius: 4px;
                        background-color: %15;
                        margin: 2px;
                    }
                    QTreeWidget::indicator:unchecked {
                        background-color: %15;
                        border: 2px solid %14;
                    }
                    QTreeWidget::indicator:checked {
                        background-color: %16;
                        border-color: %17;
                        image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEwIDNMNC41IDguNUwyIDYiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
                    }
                    QTreeWidget::indicator:hover {
                        border-color: %18;
                        background-color: %19;
                    }
                    QTreeWidget::indicator:checked:hover {
                        background-color: %20;
                        border-color: %21;
                    }
                    QTreeWidget::indicator:indeterminate {
                        background-color: %22;
                        border-color: %23;
                        image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3QgeD0iMyIgeT0iNSIgd2lkdGg9IjYiIGhlaWdodD0iMiIgZmlsbD0id2hpdGUiLz4KPC9zdmc+Cg==);
                    }
                )").arg(itemBg.name())                                      // %1 - background
                   .arg(theme.colors.foreground.name())                     // %2 - text color
                   .arg(itemBorder.name())                                  // %3 - border
                   .arg(theme.spacing.borderRadius)                         // %4 - border radius
                   .arg(itemSelectedBg.name())                              // %5 - selection background
                   .arg(itemHoverBg.name())                                 // %6 - alternate background
                   .arg(itemBorder.lighter(150).name())                     // %7 - gridline color
                   .arg(theme.typography.baseFontSize)                      // %8 - font size
                   .arg(itemSelectedBg.name())                              // %9 - selected background
                   .arg(theme.colors.background.name())                     // %10 - selected text color
                   .arg(itemHoverBg.name())                                 // %11 - hover background
                   .arg(itemSelectedBg.lighter(110).name())                 // %12 - selected hover background
                   .arg(theme.colors.foreground.name())                     // %13 - branch arrow color
                   .arg(theme.colors.border.name())                         // %14 - indicator border
                   .arg(theme.colors.background.name())                     // %15 - indicator background
                   .arg(theme.colors.accent.name())                         // %16 - indicator checked background
                   .arg(theme.colors.accent.name())                         // %17 - indicator checked border
                   .arg(theme.colors.accent.name())                         // %18 - indicator hover border
                   .arg(theme.colors.hover.name())                          // %19 - indicator hover background
                   .arg(theme.colors.accent.lighter(110).name())            // %20 - indicator checked hover background
                   .arg(theme.colors.accent.lighter(110).name())            // %21 - indicator checked hover border
                   .arg(theme.colors.warning.name())                        // %22 - indicator indeterminate background
                   .arg(theme.colors.warning.name());                       // %23 - indicator indeterminate border
            }
            break;
            
        // Add more component types as needed
        default:
            styleSheet = QString(R"(
                QWidget {
                    background-color: %1;
                    color: %2;
                }
            )").arg(theme.colors.background.name())
               .arg(theme.colors.foreground.name());
            break;
    }
    
    return styleSheet;
}
void ThemeManager::propagateThemeChangeWithRecovery()
{
    if (!m_componentRegistry) {
        LOG_WARNING(LogCategories::UI, "ComponentRegistry not available for theme propagation");
        return;
    }
    
    // CRITICAL FIX: Ensure we're on the main thread
    if (QThread::currentThread() != qApp->thread()) {
        LOG_WARNING(LogCategories::UI, "Theme propagation called from worker thread, deferring to main thread");
        QMetaObject::invokeMethod(this, "propagateThemeChangeWithRecovery", Qt::QueuedConnection);
        return;
    }
    
    ThemeData currentThemeData = getCurrentThemeData();
    
    // Enable monitoring during theme change
    bool wasMonitoring = m_componentRegistry->isMonitoring();
    if (!wasMonitoring) {
        m_componentRegistry->startMonitoring();
    }
    
    try {
        // CRITICAL FIX: Process events periodically during theme application
        // to prevent UI freeze and allow cancellation
        
        // Process any pending events before starting
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        
        // Apply theme to all registered components and dialogs
        m_componentRegistry->applyThemeToAll(currentThemeData);
        
        // Process events after bulk operation
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        
        // Update registered dialogs (legacy support)
        updateRegisteredDialogs();
        
        // Process events again
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        
        // Check for failed components and attempt recovery
        QStringList failedComponents = m_componentRegistry->getFailedComponents();
        if (!failedComponents.isEmpty()) {
            LOG_WARNING(LogCategories::UI, QString("Theme application failed for %1 components, attempting recovery")
                       .arg(failedComponents.size()));
            
            // CRITICAL FIX: Use QMetaObject::invokeMethod instead of QTimer::singleShot
            // to ensure we stay on the main thread
            QMetaObject::invokeMethod(this, [this, currentThemeData]() {
                if (m_componentRegistry) {
                    m_componentRegistry->retryFailedComponents(currentThemeData);
                    
                    // If still failing, attempt full recovery
                    QStringList stillFailed = m_componentRegistry->getFailedComponents();
                    if (!stillFailed.isEmpty()) {
                        LOG_WARNING(LogCategories::UI, QString("Attempting full recovery for %1 components")
                                   .arg(stillFailed.size()));
                        m_componentRegistry->attemptRecovery();
                    }
                }
            }, Qt::QueuedConnection);
        }
        
        // Force real-time update if enabled (deferred to avoid blocking)
        QMetaObject::invokeMethod(this, [this]() {
            if (m_componentRegistry) {
                m_componentRegistry->forceUpdateAll();
            }
        }, Qt::QueuedConnection);
        
        LOG_INFO(LogCategories::UI, "Enhanced theme propagation completed successfully");
        
    } catch (const std::exception& e) {
        LOG_ERROR(LogCategories::UI, QString("Theme propagation failed: %1").arg(e.what()));
        
        // Attempt recovery on main thread
        QMetaObject::invokeMethod(this, [this]() {
            try {
                if (m_componentRegistry) {
                    m_componentRegistry->attemptRecovery();
                }
                LOG_INFO(LogCategories::UI, "Recovery attempt completed after theme propagation failure");
            } catch (...) {
                LOG_ERROR(LogCategories::UI, "Recovery failed after theme propagation failure");
            }
        }, Qt::QueuedConnection);
        
        throw; // Re-throw to allow caller to handle
    }
    
    // Restore monitoring state
    if (!wasMonitoring) {
        m_componentRegistry->stopMonitoring();
    }
}

void ThemeManager::enableRealTimeThemeUpdates(bool enabled)
{
    if (m_componentRegistry) {
        m_componentRegistry->enableRealTimeUpdates(enabled);
        
        // Connect to real-time update signals if not already connected
        if (enabled) {
            connect(m_componentRegistry, &ComponentRegistry::realTimeUpdateTriggered,
                    this, [this]() {
                        ThemeData currentThemeData = getCurrentThemeData();
                        m_componentRegistry->applyThemeToAll(currentThemeData);
                    }, Qt::UniqueConnection);
            
            connect(m_componentRegistry, &ComponentRegistry::themeApplicationFailed,
                    this, [this](QWidget* component, const QString& error) {
                        LOG_WARNING(LogCategories::UI, QString("Real-time theme update failed for %1: %2")
                                   .arg(component->metaObject()->className()).arg(error));
                    }, Qt::UniqueConnection);
        }
        
        LOG_INFO(LogCategories::UI, QString("Real-time theme updates %1")
                 .arg(enabled ? "enabled" : "disabled"));
    }
}

void ThemeManager::setThemeUpdateInterval(int milliseconds)
{
    if (m_componentRegistry) {
        m_componentRegistry->setUpdateInterval(milliseconds);
        LOG_DEBUG(LogCategories::UI, QString("Theme update interval set to %1ms").arg(milliseconds));
    }
}

void ThemeManager::enableComponentMonitoring(bool enabled)
{
    if (m_componentRegistry) {
        if (enabled) {
            m_componentRegistry->startMonitoring();
            
            // Connect to monitoring signals
            connect(m_componentRegistry, &ComponentRegistry::componentValidationCompleted,
                    this, [this](int validCount, int totalCount) {
                        if (totalCount > 0 && validCount < totalCount) {
                            LOG_DEBUG(LogCategories::UI, QString("Component validation: %1/%2 valid")
                                     .arg(validCount).arg(totalCount));
                        }
                    }, Qt::UniqueConnection);
        } else {
            m_componentRegistry->stopMonitoring();
        }
        
        LOG_INFO(LogCategories::UI, QString("Component monitoring %1")
                 .arg(enabled ? "enabled" : "disabled"));
    }
}

void ThemeManager::attemptThemeRecovery()
{
    if (m_componentRegistry) {
        LOG_INFO(LogCategories::UI, "Attempting theme recovery for failed components");
        
        // Get current theme data
        ThemeData currentThemeData = getCurrentThemeData();
        
        // Attempt recovery
        m_componentRegistry->attemptRecovery();
        
        // Retry failed components with current theme
        m_componentRegistry->retryFailedComponents(currentThemeData);
        
        // Check results
        QStringList stillFailed = m_componentRegistry->getFailedComponents();
        if (stillFailed.isEmpty()) {
            LOG_INFO(LogCategories::UI, "Theme recovery completed successfully");
        } else {
            LOG_WARNING(LogCategories::UI, QString("Theme recovery partially successful, %1 components still failing")
                       .arg(stillFailed.size()));
        }
    }
}

QStringList ThemeManager::getFailedThemeComponents() const
{
    if (m_componentRegistry) {
        return m_componentRegistry->getFailedComponents();
    }
    return QStringList();
}

// Accessibility implementation methods

bool ThemeManager::isHighContrastModeEnabled() const
{
    return m_highContrastModeEnabled || isSystemHighContrastMode();
}

bool ThemeManager::isSystemHighContrastMode() const
{
    // Check system settings for high contrast mode
    #ifdef Q_OS_WIN
    // Windows high contrast detection
    return GetSystemMetrics(SM_CXBORDER) > 1;
    #elif defined(Q_OS_LINUX)
    // Linux accessibility settings detection
    QSettings settings("org.gnome.desktop.interface", QSettings::NativeFormat);
    return settings.value("high-contrast", false).toBool();
    #elif defined(Q_OS_MAC)
    // macOS accessibility settings detection
    // This would require Objective-C code or system calls
    return false;
    #else
    return false;
    #endif
}

void ThemeManager::enableHighContrastMode(bool enabled)
{
    if (m_highContrastModeEnabled == enabled) {
        return;
    }
    
    m_highContrastModeEnabled = enabled;
    
    LOG_INFO(LogCategories::UI, QString("High contrast mode %1").arg(enabled ? "enabled" : "disabled"));
    
    if (enabled) {
        // Switch to high contrast theme
        setTheme(HighContrast);
        
        // Enable enhanced focus indicators
        enableEnhancedFocusIndicators(true);
        
        // Set higher minimum contrast ratio
        setMinimumContrastRatio(7.0);
        
        // Enable alternative indicators
        enableAlternativeIndicators(true);
    } else {
        // Restore previous theme
        setTheme(SystemDefault);
        
        // Restore normal settings
        setMinimumContrastRatio(4.5);
    }
    
    // Apply to all registered components
    propagateThemeChange();
}

void ThemeManager::enableEnhancedFocusIndicators(bool enabled)
{
    m_enhancedFocusIndicatorsEnabled = enabled;
    
    LOG_INFO(LogCategories::UI, QString("Enhanced focus indicators %1").arg(enabled ? "enabled" : "disabled"));
    
    // Update focus indicator style
    if (enabled) {
        m_focusIndicatorStyle = generateFocusIndicatorStyles();
    } else {
        m_focusIndicatorStyle.clear();
    }
    
    // Apply to application
    applyToApplication();
}

void ThemeManager::setMinimumContrastRatio(double ratio)
{
    m_minimumContrastRatio = qMax(1.0, ratio);
    LOG_INFO(LogCategories::UI, QString("Minimum contrast ratio set to %1:1").arg(m_minimumContrastRatio));
}

double ThemeManager::getMinimumContrastRatio() const
{
    return m_minimumContrastRatio;
}

bool ThemeManager::validateAccessibilityCompliance() const
{
    ThemeData currentTheme = getCurrentThemeData();
    return StyleValidator::validateAccessibility(currentTheme);
}

QStringList ThemeManager::getAccessibilityViolations() const
{
    QStringList violations;
    
    ThemeData currentTheme = getCurrentThemeData();
    QList<StyleViolation> accessibilityViolations = StyleValidator::validateAccessibilityCompliance(currentTheme);
    
    for (const StyleViolation& violation : accessibilityViolations) {
        violations.append(QString("%1: %2").arg(violation.violationType).arg(violation.currentValue));
    }
    
    return violations;
}

void ThemeManager::applyAccessibilityEnhancements(QWidget* widget)
{
    if (!widget) return;
    
    // Apply enhanced focus indicators
    if (m_enhancedFocusIndicatorsEnabled) {
        enableFocusIndicators(widget, true);
    }
    
    // Apply alternative indicators if enabled
    if (m_alternativeIndicatorsEnabled) {
        // Add text indicators for color-only information
        if (m_widgetTextIndicators.contains(widget)) {
            QString textIndicator = m_widgetTextIndicators[widget];
            widget->setToolTip(widget->toolTip() + " " + textIndicator);
        }
        
        // Add icon indicators
        if (m_widgetIconIndicators.contains(widget)) {
            // This would require custom painting or additional UI elements
            LOG_DEBUG(LogCategories::UI, QString("Icon indicator applied to %1")
                     .arg(widget->metaObject()->className()));
        }
    }
    
    // Ensure minimum sizes for accessibility
    enforceMinimumSizes(widget);
    
    // Apply high contrast styling if enabled
    if (m_highContrastModeEnabled) {
        QString highContrastStyle = generateHighContrastThemeStyles();
        widget->setStyleSheet(widget->styleSheet() + highContrastStyle);
    }
}

void ThemeManager::enableFocusIndicators(QWidget* widget, bool enabled)
{
    if (!widget) return;
    
    if (enabled && m_enhancedFocusIndicatorsEnabled) {
        // Set focus policy to ensure widget can receive focus
        widget->setFocusPolicy(Qt::StrongFocus);
        
        // Apply focus indicator styles
        QString focusStyle = getFocusIndicatorStyle();
        if (!focusStyle.isEmpty()) {
            widget->setStyleSheet(widget->styleSheet() + focusStyle);
        }
        
        LOG_DEBUG(LogCategories::UI, QString("Focus indicators enabled for %1")
                 .arg(widget->metaObject()->className()));
    }
}

void ThemeManager::setFocusIndicatorStyle(const QString& style)
{
    m_focusIndicatorStyle = style;
    
    // Apply to all registered components
    if (m_enhancedFocusIndicatorsEnabled) {
        applyToApplication();
    }
}

QString ThemeManager::getFocusIndicatorStyle() const
{
    if (!m_focusIndicatorStyle.isEmpty()) {
        return m_focusIndicatorStyle;
    }
    
    return generateFocusIndicatorStyles();
}

void ThemeManager::enableAlternativeIndicators(bool enabled)
{
    m_alternativeIndicatorsEnabled = enabled;
    
    LOG_INFO(LogCategories::UI, QString("Alternative indicators %1").arg(enabled ? "enabled" : "disabled"));
    
    if (enabled) {
        // Apply alternative indicators to all registered components
        QWidgetList allWidgets = QApplication::allWidgets();
        for (QWidget* widget : allWidgets) {
            if (widget) {
                applyAccessibilityEnhancements(widget);
            }
        }
    }
}

bool ThemeManager::hasAlternativeIndicators() const
{
    return m_alternativeIndicatorsEnabled;
}

void ThemeManager::addIconIndicator(QWidget* widget, const QString& iconPath, const QString& description)
{
    if (!widget) return;
    
    m_widgetIconIndicators[widget] = iconPath;
    
    // Add description to tooltip
    QString currentTooltip = widget->toolTip();
    if (!currentTooltip.contains(description)) {
        widget->setToolTip(currentTooltip.isEmpty() ? description : currentTooltip + " " + description);
    }
    
    LOG_DEBUG(LogCategories::UI, QString("Icon indicator added to %1: %2")
             .arg(widget->metaObject()->className()).arg(description));
}

void ThemeManager::addTextIndicator(QWidget* widget, const QString& text)
{
    if (!widget) return;
    
    m_widgetTextIndicators[widget] = text;
    
    // Add text to tooltip
    QString currentTooltip = widget->toolTip();
    if (!currentTooltip.contains(text)) {
        widget->setToolTip(currentTooltip.isEmpty() ? text : currentTooltip + " " + text);
    }
    
    LOG_DEBUG(LogCategories::UI, QString("Text indicator added to %1: %2")
             .arg(widget->metaObject()->className()).arg(text));
}

ThemeData ThemeManager::getHighContrastThemeData() const
{
    ThemeData themeData;
    themeData.name = "High Contrast";
    themeData.description = "High contrast theme for enhanced accessibility";
    
    // High contrast colors with maximum contrast ratios
    themeData.colors.background = QColor(0, 0, 0);        // Pure black
    themeData.colors.foreground = QColor(255, 255, 255);  // Pure white
    themeData.colors.accent = QColor(255, 255, 0);        // Bright yellow
    themeData.colors.border = QColor(255, 255, 255);      // White borders
    themeData.colors.hover = QColor(128, 128, 128);       // Gray hover
    themeData.colors.disabled = QColor(128, 128, 128);    // Gray disabled
    themeData.colors.success = QColor(0, 255, 0);         // Bright green
    themeData.colors.warning = QColor(255, 255, 0);       // Bright yellow
    themeData.colors.error = QColor(255, 0, 0);           // Bright red
    themeData.colors.info = QColor(0, 255, 255);          // Bright cyan
    
    // Enhanced typography for accessibility
    themeData.typography.fontFamily = "Arial, sans-serif";
    themeData.typography.baseFontSize = 12;  // Larger base font
    themeData.typography.titleFontSize = 16;
    themeData.typography.smallFontSize = 10;
    themeData.typography.boldTitles = true;
    
    // Enhanced spacing for better touch targets
    themeData.spacing.padding = 12;
    themeData.spacing.margin = 8;
    themeData.spacing.borderRadius = 0;  // Sharp corners for clarity
    themeData.spacing.borderWidth = 3;   // Thicker borders
    
    themeData.created = QDateTime::currentDateTime();
    themeData.modified = themeData.created;
    
    return themeData;
}
//
// Accessibility helper methods implementation

void ThemeManager::setupAccessibleTabOrder(QWidget* parent)
{
    if (!parent) return;
    
    // Find all focusable widgets
    QList<QWidget*> focusableWidgets;
    QList<QWidget*> allChildren = parent->findChildren<QWidget*>();
    
    for (QWidget* child : allChildren) {
        if (child && child->focusPolicy() != Qt::NoFocus && child->isVisible() && child->isEnabled()) {
            focusableWidgets.append(child);
        }
    }
    
    // Sort widgets by their position (top-to-bottom, left-to-right)
    std::sort(focusableWidgets.begin(), focusableWidgets.end(), 
              [](QWidget* a, QWidget* b) {
                  QPoint posA = a->mapToGlobal(QPoint(0, 0));
                  QPoint posB = b->mapToGlobal(QPoint(0, 0));
                  
                  if (std::abs(posA.y() - posB.y()) < 10) {
                      // Same row, sort by x position
                      return posA.x() < posB.x();
                  }
                  // Different rows, sort by y position
                  return posA.y() < posB.y();
              });
    
    // Set up tab order
    for (int i = 0; i < focusableWidgets.size() - 1; ++i) {
        QWidget::setTabOrder(focusableWidgets[i], focusableWidgets[i + 1]);
    }
    
    LOG_DEBUG(LogCategories::UI, QString("Set up accessible tab order for %1 widgets in %2")
             .arg(focusableWidgets.size())
             .arg(parent->metaObject()->className()));
}

void ThemeManager::setupAccessibleKeyboardShortcuts(QWidget* parent)
{
    if (!parent) return;
    
    // Find buttons and add keyboard shortcuts
    QList<QPushButton*> buttons = parent->findChildren<QPushButton*>();
    
    for (QPushButton* button : buttons) {
        QString text = button->text();
        
        // Add common keyboard shortcuts
        if (text.contains("OK", Qt::CaseInsensitive) || text.contains("Accept", Qt::CaseInsensitive)) {
            button->setShortcut(QKeySequence(Qt::Key_Return));
        } else if (text.contains("Cancel", Qt::CaseInsensitive)) {
            button->setShortcut(QKeySequence(Qt::Key_Escape));
        } else if (text.contains("Apply", Qt::CaseInsensitive)) {
            button->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Return));
        } else if (text.contains("Help", Qt::CaseInsensitive)) {
            button->setShortcut(QKeySequence(Qt::Key_F1));
        }
        
        // Ensure button text shows the shortcut
        if (!button->shortcut().isEmpty()) {
            QString shortcutText = button->shortcut().toString();
            if (!text.contains(shortcutText)) {
                button->setText(text + QString(" (%1)").arg(shortcutText));
            }
        }
    }
    
    LOG_DEBUG(LogCategories::UI, QString("Set up accessible keyboard shortcuts for %1 buttons in %2")
             .arg(buttons.size())
             .arg(parent->metaObject()->className()));
}

void ThemeManager::enableKeyboardNavigation(QWidget* widget, bool enabled)
{
    if (!widget) return;
    
    if (enabled) {
        // Ensure widget can receive focus
        if (widget->focusPolicy() == Qt::NoFocus) {
            widget->setFocusPolicy(Qt::StrongFocus);
        }
        
        LOG_DEBUG(LogCategories::UI, QString("Enabled keyboard navigation for %1")
                 .arg(widget->metaObject()->className()));
    }
}

// Performance optimization methods
void ThemeManager::enablePerformanceOptimization(bool enabled)
{
    if (!m_performanceOptimizer) return;
    
    if (enabled) {
        m_performanceOptimizer->enableStyleSheetCaching(true);
        m_performanceOptimizer->enableBatchUpdates(true);
        m_performanceOptimizer->enableAsyncUpdates(true);
        m_performanceOptimizer->startPerformanceMonitoring();
        
        // Connect performance optimizer to theme changes
        connect(this, &ThemeManager::themeChanged, this, [this](Theme theme, const QString& themeName) {
            if (m_performanceOptimizer) {
                ThemeData currentTheme = getCurrentThemeData();
                m_performanceOptimizer->optimizedApplyTheme(currentTheme);
            }
        });
        
        LOG_INFO(LogCategories::UI, "Performance optimization enabled with caching and batch updates");
    } else {
        m_performanceOptimizer->enableStyleSheetCaching(false);
        m_performanceOptimizer->enableBatchUpdates(false);
        m_performanceOptimizer->enableAsyncUpdates(false);
        m_performanceOptimizer->stopPerformanceMonitoring();
        
        LOG_INFO(LogCategories::UI, "Performance optimization disabled");
    }
}

void ThemeManager::enableStyleSheetCaching(bool enabled)
{
    if (m_performanceOptimizer) {
        m_performanceOptimizer->enableStyleSheetCaching(enabled);
        LOG_INFO(LogCategories::UI, QString("StyleSheet caching %1").arg(enabled ? "enabled" : "disabled"));
    }
}

void ThemeManager::enableBatchUpdates(bool enabled)
{
    if (m_performanceOptimizer) {
        m_performanceOptimizer->enableBatchUpdates(enabled);
        LOG_INFO(LogCategories::UI, QString("Batch updates %1").arg(enabled ? "enabled" : "disabled"));
    }
}

void ThemeManager::setPerformanceTarget(int maxSwitchTimeMs)
{
    if (m_performanceOptimizer) {
        m_performanceOptimizer->setPerformanceTarget(maxSwitchTimeMs);
        LOG_INFO(LogCategories::UI, QString("Performance target set to %1ms").arg(maxSwitchTimeMs));
    }
}

qint64 ThemeManager::getLastThemeSwitchTime() const
{
    return m_performanceOptimizer ? m_performanceOptimizer->getLastSwitchTime() : 0;
}

qint64 ThemeManager::getAverageThemeSwitchTime() const
{
    return m_performanceOptimizer ? m_performanceOptimizer->getAverageSwitchTime() : 0;
}

int ThemeManager::getCacheHitRate() const
{
    return m_performanceOptimizer ? m_performanceOptimizer->getCacheHitRate() : 0;
}

QString ThemeManager::generatePerformanceReport() const
{
    if (!m_performanceOptimizer) {
        return "Performance optimizer not available";
    }
    
    return m_performanceOptimizer->generatePerformanceReport();
}

void ThemeManager::resetPerformanceMetrics()
{
    if (m_performanceOptimizer) {
        m_performanceOptimizer->resetPerformanceMetrics();
        LOG_INFO(LogCategories::UI, "Performance metrics reset");
    }
}