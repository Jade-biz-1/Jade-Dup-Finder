#ifndef THEME_EDITOR_H
#define THEME_EDITOR_H

#include <QDialog>
#include <QColorDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QLineEdit>
#include <QTextEdit>
#include <QSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QGroupBox>
#include <QScrollArea>
#include <QTabWidget>
#include <QRadioButton>
#include <QProgressBar>
#include "theme_manager.h"

class ThemeEditor : public QDialog
{
    Q_OBJECT
    
public:
    explicit ThemeEditor(QWidget* parent = nullptr);
    
    void setBaseTheme(ThemeManager::Theme baseTheme);
    void loadCustomTheme(const QString& themeName);
    
private slots:
    void onColorChanged();
    void onTypographyChanged();
    void onSpacingChanged();
    void onPreviewRequested();
    void onSaveTheme();
    void onLoadTheme();
    void onExportTheme();
    void onResetToDefaults();
    void onAccessibilityCheck();
    void onNameChanged();
    
private:
    void setupUI();
    void createColorSection();
    void createTypographySection();
    void createSpacingSection();
    void createPreviewSection();
    QWidget* createButtonSection();
    void updatePreview();
    bool validateAccessibility();
    void updateColorPickers();
    void updateTypographyControls();
    void updateSpacingControls();
    void resetToTheme(ThemeManager::Theme theme);
    
    // UI Components
    QTabWidget* m_tabWidget;
    QScrollArea* m_scrollArea;
    
    // Color section
    QGroupBox* m_colorGroup;
    struct ColorPickers {
        QPushButton* background;
        QPushButton* foreground;
        QPushButton* accent;
        QPushButton* border;
        QPushButton* hover;
        QPushButton* disabled;
        QPushButton* success;
        QPushButton* warning;
        QPushButton* error;
        QPushButton* info;
    } m_colorPickers;
    
    // Typography section
    QGroupBox* m_typographyGroup;
    QComboBox* m_fontFamilyCombo;
    QSpinBox* m_baseFontSizeSpin;
    QSpinBox* m_titleFontSizeSpin;
    QSpinBox* m_smallFontSizeSpin;
    QCheckBox* m_boldTitlesCheck;
    
    // Spacing section
    QGroupBox* m_spacingGroup;
    QSpinBox* m_paddingSpin;
    QSpinBox* m_marginSpin;
    QSpinBox* m_borderRadiusSpin;
    QSpinBox* m_borderWidthSpin;
    
    // Preview section
    QWidget* m_previewArea;
    QVBoxLayout* m_previewLayout;
    
    // Theme info
    QLineEdit* m_themeNameEdit;
    QTextEdit* m_themeDescriptionEdit;
    
    // Buttons
    QPushButton* m_previewButton;
    QPushButton* m_saveButton;
    QPushButton* m_loadButton;
    QPushButton* m_resetButton;
    QPushButton* m_accessibilityButton;
    QPushButton* m_cancelButton;
    
    // Data
    ThemeData m_currentTheme;
    ThemeManager::Theme m_baseTheme;
    bool m_previewMode;
    bool m_isModified;
    
    // Helper methods
    void setColorButtonColor(QPushButton* button, const QColor& color);
    QColor getColorFromButton(QPushButton* button);
    void connectColorButton(QPushButton* button);
    void createPreviewWidgets();
    void applyThemeToPreview();
    QString generatePreviewStyleSheet() const;
};

#endif // THEME_EDITOR_H