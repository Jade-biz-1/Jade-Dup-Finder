#include "theme_editor.h"
#include "theme_manager.h"
#include "theme_persistence.h"
#include "core/logger.h"
#include <QColorDialog>
#include <QFontDatabase>
#include <QMessageBox>
#include <QFileDialog>
#include <QInputDialog>
#include <QStandardPaths>

ThemeEditor::ThemeEditor(QWidget* parent)
    : QDialog(parent)
    , m_baseTheme(ThemeManager::Light)
    , m_previewMode(false)
    , m_isModified(false)
{
    setWindowTitle("Theme Editor");
    setModal(true);
    resize(800, 600);
    
    setupUI();
    resetToTheme(ThemeManager::Light);
    
    LOG_INFO(LogCategories::UI, "ThemeEditor initialized");
}

void ThemeEditor::setBaseTheme(ThemeManager::Theme baseTheme)
{
    m_baseTheme = baseTheme;
    resetToTheme(baseTheme);
}

void ThemeEditor::loadCustomTheme(const QString& themeName)
{
    if (themeName.isEmpty()) {
        return;
    }
    
    ThemeData theme = ThemeManager::instance()->getThemeData(themeName);
    if (!theme.isValid()) {
        QMessageBox::warning(this, "Load Theme", 
                           QString("Failed to load theme '%1'").arg(themeName));
        return;
    }
    
    m_currentTheme = theme;
    m_themeNameEdit->setText(theme.name);
    m_themeDescriptionEdit->setPlainText(theme.description);
    
    updateColorPickers();
    updateTypographyControls();
    updateSpacingControls();
    updatePreview();
    
    m_isModified = false;
    
    LOG_INFO(LogCategories::UI, QString("Loaded custom theme '%1' in editor").arg(themeName));
}

void ThemeEditor::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    
    // Create tab widget
    m_tabWidget = new QTabWidget;
    
    // Create sections
    createColorSection();
    createTypographySection();
    createSpacingSection();
    createPreviewSection();
    
    // Add tabs
    m_tabWidget->addTab(m_colorGroup, "Colors");
    m_tabWidget->addTab(m_typographyGroup, "Typography");
    m_tabWidget->addTab(m_spacingGroup, "Spacing");
    m_tabWidget->addTab(m_previewArea, "Preview");
    
    mainLayout->addWidget(m_tabWidget);
    
    // Theme info section
    QGroupBox* infoGroup = new QGroupBox("Theme Information");
    QVBoxLayout* infoLayout = new QVBoxLayout(infoGroup);
    
    QHBoxLayout* nameLayout = new QHBoxLayout;
    nameLayout->addWidget(new QLabel("Name:"));
    m_themeNameEdit = new QLineEdit;
    connect(m_themeNameEdit, &QLineEdit::textChanged, this, &ThemeEditor::onNameChanged);
    nameLayout->addWidget(m_themeNameEdit);
    infoLayout->addLayout(nameLayout);
    
    infoLayout->addWidget(new QLabel("Description:"));
    m_themeDescriptionEdit = new QTextEdit;
    m_themeDescriptionEdit->setMaximumHeight(80);
    infoLayout->addWidget(m_themeDescriptionEdit);
    
    mainLayout->addWidget(infoGroup);
    
    // Button section
    createButtonSection();
    mainLayout->addWidget(createButtonSection());
}

void ThemeEditor::createColorSection()
{
    m_colorGroup = new QGroupBox("Color Scheme");
    QGridLayout* colorLayout = new QGridLayout(m_colorGroup);
    
    // Create color picker buttons
    struct ColorInfo {
        QString name;
        QString label;
        QPushButton** button;
    };
    
    QList<ColorInfo> colors = {
        {"background", "Background:", &m_colorPickers.background},
        {"foreground", "Foreground:", &m_colorPickers.foreground},
        {"accent", "Accent:", &m_colorPickers.accent},
        {"border", "Border:", &m_colorPickers.border},
        {"hover", "Hover:", &m_colorPickers.hover},
        {"disabled", "Disabled:", &m_colorPickers.disabled},
        {"success", "Success:", &m_colorPickers.success},
        {"warning", "Warning:", &m_colorPickers.warning},
        {"error", "Error:", &m_colorPickers.error},
        {"info", "Info:", &m_colorPickers.info}
    };
    
    int row = 0;
    for (const ColorInfo& colorInfo : colors) {
        QLabel* label = new QLabel(colorInfo.label);
        colorLayout->addWidget(label, row, 0);
        
        QPushButton* button = new QPushButton;
        button->setMinimumSize(100, 30);
        button->setObjectName(colorInfo.name);
        *colorInfo.button = button;
        
        connectColorButton(button);
        colorLayout->addWidget(button, row, 1);
        
        row++;
    }
}

void ThemeEditor::createTypographySection()
{
    m_typographyGroup = new QGroupBox("Typography");
    QGridLayout* typographyLayout = new QGridLayout(m_typographyGroup);
    
    // Font family
    typographyLayout->addWidget(new QLabel("Font Family:"), 0, 0);
    m_fontFamilyCombo = new QComboBox;
    QFontDatabase fontDb;
    m_fontFamilyCombo->addItems(fontDb.families());
    connect(m_fontFamilyCombo, QOverload<const QString&>::of(&QComboBox::currentTextChanged),
            this, &ThemeEditor::onTypographyChanged);
    typographyLayout->addWidget(m_fontFamilyCombo, 0, 1);
    
    // Base font size
    typographyLayout->addWidget(new QLabel("Base Font Size:"), 1, 0);
    m_baseFontSizeSpin = new QSpinBox;
    m_baseFontSizeSpin->setRange(6, 24);
    m_baseFontSizeSpin->setSuffix(" pt");
    connect(m_baseFontSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ThemeEditor::onTypographyChanged);
    typographyLayout->addWidget(m_baseFontSizeSpin, 1, 1);
    
    // Title font size
    typographyLayout->addWidget(new QLabel("Title Font Size:"), 2, 0);
    m_titleFontSizeSpin = new QSpinBox;
    m_titleFontSizeSpin->setRange(8, 32);
    m_titleFontSizeSpin->setSuffix(" pt");
    connect(m_titleFontSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ThemeEditor::onTypographyChanged);
    typographyLayout->addWidget(m_titleFontSizeSpin, 2, 1);
    
    // Small font size
    typographyLayout->addWidget(new QLabel("Small Font Size:"), 3, 0);
    m_smallFontSizeSpin = new QSpinBox;
    m_smallFontSizeSpin->setRange(4, 16);
    m_smallFontSizeSpin->setSuffix(" pt");
    connect(m_smallFontSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ThemeEditor::onTypographyChanged);
    typographyLayout->addWidget(m_smallFontSizeSpin, 3, 1);
    
    // Bold titles
    m_boldTitlesCheck = new QCheckBox("Bold Titles");
    connect(m_boldTitlesCheck, &QCheckBox::toggled, this, &ThemeEditor::onTypographyChanged);
    typographyLayout->addWidget(m_boldTitlesCheck, 4, 0, 1, 2);
}

void ThemeEditor::createSpacingSection()
{
    m_spacingGroup = new QGroupBox("Spacing");
    QGridLayout* spacingLayout = new QGridLayout(m_spacingGroup);
    
    // Padding
    spacingLayout->addWidget(new QLabel("Padding:"), 0, 0);
    m_paddingSpin = new QSpinBox;
    m_paddingSpin->setRange(0, 32);
    m_paddingSpin->setSuffix(" px");
    connect(m_paddingSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ThemeEditor::onSpacingChanged);
    spacingLayout->addWidget(m_paddingSpin, 0, 1);
    
    // Margin
    spacingLayout->addWidget(new QLabel("Margin:"), 1, 0);
    m_marginSpin = new QSpinBox;
    m_marginSpin->setRange(0, 32);
    m_marginSpin->setSuffix(" px");
    connect(m_marginSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ThemeEditor::onSpacingChanged);
    spacingLayout->addWidget(m_marginSpin, 1, 1);
    
    // Border radius
    spacingLayout->addWidget(new QLabel("Border Radius:"), 2, 0);
    m_borderRadiusSpin = new QSpinBox;
    m_borderRadiusSpin->setRange(0, 20);
    m_borderRadiusSpin->setSuffix(" px");
    connect(m_borderRadiusSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ThemeEditor::onSpacingChanged);
    spacingLayout->addWidget(m_borderRadiusSpin, 2, 1);
    
    // Border width
    spacingLayout->addWidget(new QLabel("Border Width:"), 3, 0);
    m_borderWidthSpin = new QSpinBox;
    m_borderWidthSpin->setRange(0, 10);
    m_borderWidthSpin->setSuffix(" px");
    connect(m_borderWidthSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ThemeEditor::onSpacingChanged);
    spacingLayout->addWidget(m_borderWidthSpin, 3, 1);
}

void ThemeEditor::createPreviewSection()
{
    m_previewArea = new QWidget;
    m_previewLayout = new QVBoxLayout(m_previewArea);
    
    QLabel* previewLabel = new QLabel("Theme Preview");
    previewLabel->setAlignment(Qt::AlignCenter);
    QFont font = previewLabel->font();
    font.setPointSize(14);
    font.setBold(true);
    previewLabel->setFont(font);
    m_previewLayout->addWidget(previewLabel);
    
    createPreviewWidgets();
}

QWidget* ThemeEditor::createButtonSection()
{
    QWidget* buttonWidget = new QWidget;
    QHBoxLayout* buttonLayout = new QHBoxLayout(buttonWidget);
    
    m_previewButton = new QPushButton("Preview");
    connect(m_previewButton, &QPushButton::clicked, this, &ThemeEditor::onPreviewRequested);
    buttonLayout->addWidget(m_previewButton);
    
    m_accessibilityButton = new QPushButton("Check Accessibility");
    connect(m_accessibilityButton, &QPushButton::clicked, this, &ThemeEditor::onAccessibilityCheck);
    buttonLayout->addWidget(m_accessibilityButton);
    
    buttonLayout->addStretch();
    
    m_loadButton = new QPushButton("Load Theme...");
    connect(m_loadButton, &QPushButton::clicked, this, &ThemeEditor::onLoadTheme);
    buttonLayout->addWidget(m_loadButton);
    
    QPushButton* exportButton = new QPushButton("Export Theme...");
    connect(exportButton, &QPushButton::clicked, this, &ThemeEditor::onExportTheme);
    buttonLayout->addWidget(exportButton);
    
    m_resetButton = new QPushButton("Reset");
    connect(m_resetButton, &QPushButton::clicked, this, &ThemeEditor::onResetToDefaults);
    buttonLayout->addWidget(m_resetButton);
    
    m_saveButton = new QPushButton("Save Theme");
    connect(m_saveButton, &QPushButton::clicked, this, &ThemeEditor::onSaveTheme);
    buttonLayout->addWidget(m_saveButton);
    
    m_cancelButton = new QPushButton("Cancel");
    connect(m_cancelButton, &QPushButton::clicked, this, &QDialog::reject);
    buttonLayout->addWidget(m_cancelButton);
    
    return buttonWidget;
}

void ThemeEditor::connectColorButton(QPushButton* button)
{
    connect(button, &QPushButton::clicked, this, [this, button]() {
        QColor currentColor = getColorFromButton(button);
        QColor newColor = QColorDialog::getColor(currentColor, this, 
                                                QString("Choose %1 Color").arg(button->objectName()));
        
        if (newColor.isValid() && newColor != currentColor) {
            setColorButtonColor(button, newColor);
            onColorChanged();
        }
    });
}

void ThemeEditor::setColorButtonColor(QPushButton* button, const QColor& color)
{
    button->setStyleSheet(QString("QPushButton { background-color: %1; color: %2; }")
                         .arg(color.name())
                         .arg(color.lightness() > 128 ? "black" : "white"));
    button->setText(color.name().toUpper());
}

QColor ThemeEditor::getColorFromButton(QPushButton* button)
{
    QString text = button->text();
    if (text.startsWith("#")) {
        return QColor(text);
    }
    return QColor();
}

void ThemeEditor::createPreviewWidgets()
{
    // Create sample widgets to preview the theme
    QGroupBox* sampleGroup = new QGroupBox("Sample Controls");
    QVBoxLayout* sampleLayout = new QVBoxLayout(sampleGroup);
    
    // Buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout;
    buttonLayout->addWidget(new QPushButton("Normal Button"));
    QPushButton* disabledButton = new QPushButton("Disabled Button");
    disabledButton->setEnabled(false);
    buttonLayout->addWidget(disabledButton);
    sampleLayout->addLayout(buttonLayout);
    
    // Input controls
    QHBoxLayout* inputLayout = new QHBoxLayout;
    QLineEdit* lineEdit = new QLineEdit("Sample text input");
    inputLayout->addWidget(lineEdit);
    QComboBox* comboBox = new QComboBox;
    comboBox->addItems({"Option 1", "Option 2", "Option 3"});
    inputLayout->addWidget(comboBox);
    sampleLayout->addLayout(inputLayout);
    
    // Checkboxes and radio buttons
    QHBoxLayout* checkLayout = new QHBoxLayout;
    QCheckBox* checkBox = new QCheckBox("Sample checkbox");
    checkBox->setChecked(true);
    checkLayout->addWidget(checkBox);
    QRadioButton* radioButton = new QRadioButton("Sample radio button");
    radioButton->setChecked(true);
    checkLayout->addWidget(radioButton);
    sampleLayout->addLayout(checkLayout);
    
    // Progress bar
    QProgressBar* progressBar = new QProgressBar;
    progressBar->setValue(65);
    sampleLayout->addWidget(progressBar);
    
    m_previewLayout->addWidget(sampleGroup);
    m_previewLayout->addStretch();
}

void ThemeEditor::onColorChanged()
{
    // Update theme data from color pickers
    m_currentTheme.colors.background = getColorFromButton(m_colorPickers.background);
    m_currentTheme.colors.foreground = getColorFromButton(m_colorPickers.foreground);
    m_currentTheme.colors.accent = getColorFromButton(m_colorPickers.accent);
    m_currentTheme.colors.border = getColorFromButton(m_colorPickers.border);
    m_currentTheme.colors.hover = getColorFromButton(m_colorPickers.hover);
    m_currentTheme.colors.disabled = getColorFromButton(m_colorPickers.disabled);
    m_currentTheme.colors.success = getColorFromButton(m_colorPickers.success);
    m_currentTheme.colors.warning = getColorFromButton(m_colorPickers.warning);
    m_currentTheme.colors.error = getColorFromButton(m_colorPickers.error);
    m_currentTheme.colors.info = getColorFromButton(m_colorPickers.info);
    
    m_currentTheme.modified = QDateTime::currentDateTime();
    m_isModified = true;
    
    if (m_previewMode) {
        updatePreview();
    }
}

void ThemeEditor::onTypographyChanged()
{
    m_currentTheme.typography.fontFamily = m_fontFamilyCombo->currentText();
    m_currentTheme.typography.baseFontSize = m_baseFontSizeSpin->value();
    m_currentTheme.typography.titleFontSize = m_titleFontSizeSpin->value();
    m_currentTheme.typography.smallFontSize = m_smallFontSizeSpin->value();
    m_currentTheme.typography.boldTitles = m_boldTitlesCheck->isChecked();
    
    m_currentTheme.modified = QDateTime::currentDateTime();
    m_isModified = true;
    
    if (m_previewMode) {
        updatePreview();
    }
}

void ThemeEditor::onSpacingChanged()
{
    m_currentTheme.spacing.padding = m_paddingSpin->value();
    m_currentTheme.spacing.margin = m_marginSpin->value();
    m_currentTheme.spacing.borderRadius = m_borderRadiusSpin->value();
    m_currentTheme.spacing.borderWidth = m_borderWidthSpin->value();
    
    m_currentTheme.modified = QDateTime::currentDateTime();
    m_isModified = true;
    
    if (m_previewMode) {
        updatePreview();
    }
}

void ThemeEditor::onPreviewRequested()
{
    m_previewMode = !m_previewMode;
    
    if (m_previewMode) {
        m_previewButton->setText("Stop Preview");
        updatePreview();
        
        // Apply preview theme to the entire dialog for real-time preview
        QString previewStyleSheet = generatePreviewStyleSheet();
        this->setStyleSheet(previewStyleSheet);
        
        LOG_INFO(LogCategories::UI, "Theme preview mode enabled");
    } else {
        m_previewButton->setText("Preview");
        // Restore original theme
        ThemeManager::instance()->applyToWidget(this);
        
        LOG_INFO(LogCategories::UI, "Theme preview mode disabled");
    }
}

void ThemeEditor::onSaveTheme()
{
    QString name = m_themeNameEdit->text().trimmed();
    if (name.isEmpty()) {
        QMessageBox::warning(this, "Save Theme", "Please enter a theme name.");
        return;
    }
    
    // Validate theme before saving
    if (!validateAccessibility()) {
        int ret = QMessageBox::question(this, "Accessibility Warning", 
                                      "This theme does not meet accessibility standards. "
                                      "Do you want to save it anyway?",
                                      QMessageBox::Yes | QMessageBox::No);
        if (ret != QMessageBox::Yes) {
            return;
        }
    }
    
    // Check if theme name already exists
    QStringList existingThemes = ThemeManager::instance()->getCustomThemeNames();
    if (existingThemes.contains(name)) {
        int ret = QMessageBox::question(this, "Theme Exists", 
                                      QString("A theme named '%1' already exists. "
                                              "Do you want to overwrite it?").arg(name),
                                      QMessageBox::Yes | QMessageBox::No);
        if (ret != QMessageBox::Yes) {
            return;
        }
    }
    
    m_currentTheme.name = name;
    m_currentTheme.description = m_themeDescriptionEdit->toPlainText();
    m_currentTheme.modified = QDateTime::currentDateTime();
    
    if (m_currentTheme.created.isNull()) {
        m_currentTheme.created = m_currentTheme.modified;
    }
    
    bool success = ThemeManager::instance()->saveCustomTheme(name, m_currentTheme);
    if (success) {
        QMessageBox::information(this, "Save Theme", 
                               QString("Theme '%1' saved successfully.").arg(name));
        m_isModified = false;
        
        // Ask if user wants to apply the theme immediately
        int ret = QMessageBox::question(this, "Apply Theme", 
                                      QString("Do you want to apply theme '%1' now?").arg(name),
                                      QMessageBox::Yes | QMessageBox::No);
        if (ret == QMessageBox::Yes) {
            ThemeManager::instance()->setTheme(ThemeManager::Custom, name);
        }
        
        accept();
    } else {
        QMessageBox::critical(this, "Save Theme", 
                            QString("Failed to save theme '%1'.").arg(name));
    }
}

void ThemeEditor::onLoadTheme()
{
    QStringList customThemes = ThemeManager::instance()->getCustomThemeNames();
    
    // Add option to load from file
    QStringList options = customThemes;
    options.prepend("--- Load from file ---");
    
    if (customThemes.isEmpty()) {
        // Only show file option if no custom themes exist
        options = QStringList() << "--- Load from file ---";
    }
    
    bool ok;
    QString selection = QInputDialog::getItem(this, "Load Theme", 
                                            "Select a theme to load:", 
                                            options, 0, false, &ok);
    
    if (!ok || selection.isEmpty()) {
        return;
    }
    
    if (selection == "--- Load from file ---") {
        // Load theme from file
        QString fileName = QFileDialog::getOpenFileName(this, 
                                                      "Load Theme File",
                                                      QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),
                                                      "Theme Files (*.json);;All Files (*)");
        
        if (!fileName.isEmpty()) {
            QFile file(fileName);
            if (file.open(QIODevice::ReadOnly)) {
                QByteArray data = file.readAll();
                file.close();
                
                QJsonParseError error;
                QJsonDocument doc = QJsonDocument::fromJson(data, &error);
                
                if (error.error == QJsonParseError::NoError && doc.isObject()) {
                    QJsonObject json = doc.object();
                    if (ThemePersistence::validateThemeJson(json)) {
                        ThemeData theme = ThemePersistence::themeFromJson(json);
                        
                        m_currentTheme = theme;
                        m_themeNameEdit->setText(theme.name);
                        m_themeDescriptionEdit->setPlainText(theme.description);
                        
                        updateColorPickers();
                        updateTypographyControls();
                        updateSpacingControls();
                        updatePreview();
                        
                        m_isModified = true;
                        
                        QMessageBox::information(this, "Load Theme", 
                                               QString("Theme '%1' loaded from file successfully.").arg(theme.name));
                    } else {
                        QMessageBox::critical(this, "Load Theme", "Invalid theme file format.");
                    }
                } else {
                    QMessageBox::critical(this, "Load Theme", 
                                        QString("Failed to parse theme file: %1").arg(error.errorString()));
                }
            } else {
                QMessageBox::critical(this, "Load Theme", "Failed to open theme file.");
            }
        }
    } else {
        // Load existing custom theme
        loadCustomTheme(selection);
    }
}

void ThemeEditor::onResetToDefaults()
{
    if (m_isModified) {
        int ret = QMessageBox::question(this, "Reset Theme", 
                                      "This will discard all changes. Are you sure?",
                                      QMessageBox::Yes | QMessageBox::No);
        if (ret != QMessageBox::Yes) {
            return;
        }
    }
    
    resetToTheme(m_baseTheme);
}

void ThemeEditor::onAccessibilityCheck()
{
    bool isAccessible = validateAccessibility();
    
    QString message;
    if (isAccessible) {
        message = "✅ Theme meets accessibility standards (WCAG 2.1 AA)";
    } else {
        message = "❌ Theme does not meet accessibility standards.\n\n";
        message += "Issues found:\n";
        
        // Check specific contrast ratios
        double bgFgRatio = m_currentTheme.getContrastRatio(m_currentTheme.colors.foreground, 
                                                          m_currentTheme.colors.background);
        if (bgFgRatio < 4.5) {
            message += QString("• Text contrast ratio: %1:1 (minimum 4.5:1)\n").arg(bgFgRatio, 0, 'f', 2);
        }
        
        double accentBgRatio = m_currentTheme.getContrastRatio(m_currentTheme.colors.accent, 
                                                              m_currentTheme.colors.background);
        if (accentBgRatio < 3.0) {
            message += QString("• Accent contrast ratio: %1:1 (minimum 3.0:1)\n").arg(accentBgRatio, 0, 'f', 2);
        }
    }
    
    QMessageBox::information(this, "Accessibility Check", message);
}

void ThemeEditor::onExportTheme()
{
    QString name = m_themeNameEdit->text().trimmed();
    if (name.isEmpty()) {
        name = "Custom Theme";
    }
    
    // Update current theme data
    m_currentTheme.name = name;
    m_currentTheme.description = m_themeDescriptionEdit->toPlainText();
    m_currentTheme.modified = QDateTime::currentDateTime();
    
    if (m_currentTheme.created.isNull()) {
        m_currentTheme.created = m_currentTheme.modified;
    }
    
    QString fileName = QFileDialog::getSaveFileName(this, 
                                                  "Export Theme",
                                                  QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) + "/" + name + ".json",
                                                  "Theme Files (*.json);;All Files (*)");
    
    if (!fileName.isEmpty()) {
        QJsonObject json = ThemePersistence::themeToJson(m_currentTheme);
        QJsonDocument doc(json);
        
        QFile file(fileName);
        if (file.open(QIODevice::WriteOnly)) {
            qint64 bytesWritten = file.write(doc.toJson());
            file.close();
            
            if (bytesWritten > 0) {
                QMessageBox::information(this, "Export Theme", 
                                       QString("Theme '%1' exported successfully to:\n%2").arg(name).arg(fileName));
            } else {
                QMessageBox::critical(this, "Export Theme", "Failed to write theme file.");
            }
        } else {
            QMessageBox::critical(this, "Export Theme", "Failed to create theme file.");
        }
    }
}

void ThemeEditor::onNameChanged()
{
    m_isModified = true;
}

void ThemeEditor::updatePreview()
{
    if (!m_previewMode) {
        return;
    }
    
    applyThemeToPreview();
}

bool ThemeEditor::validateAccessibility()
{
    return m_currentTheme.meetsAccessibilityStandards();
}

void ThemeEditor::updateColorPickers()
{
    setColorButtonColor(m_colorPickers.background, m_currentTheme.colors.background);
    setColorButtonColor(m_colorPickers.foreground, m_currentTheme.colors.foreground);
    setColorButtonColor(m_colorPickers.accent, m_currentTheme.colors.accent);
    setColorButtonColor(m_colorPickers.border, m_currentTheme.colors.border);
    setColorButtonColor(m_colorPickers.hover, m_currentTheme.colors.hover);
    setColorButtonColor(m_colorPickers.disabled, m_currentTheme.colors.disabled);
    setColorButtonColor(m_colorPickers.success, m_currentTheme.colors.success);
    setColorButtonColor(m_colorPickers.warning, m_currentTheme.colors.warning);
    setColorButtonColor(m_colorPickers.error, m_currentTheme.colors.error);
    setColorButtonColor(m_colorPickers.info, m_currentTheme.colors.info);
}

void ThemeEditor::updateTypographyControls()
{
    m_fontFamilyCombo->setCurrentText(m_currentTheme.typography.fontFamily);
    m_baseFontSizeSpin->setValue(m_currentTheme.typography.baseFontSize);
    m_titleFontSizeSpin->setValue(m_currentTheme.typography.titleFontSize);
    m_smallFontSizeSpin->setValue(m_currentTheme.typography.smallFontSize);
    m_boldTitlesCheck->setChecked(m_currentTheme.typography.boldTitles);
}

void ThemeEditor::updateSpacingControls()
{
    m_paddingSpin->setValue(m_currentTheme.spacing.padding);
    m_marginSpin->setValue(m_currentTheme.spacing.margin);
    m_borderRadiusSpin->setValue(m_currentTheme.spacing.borderRadius);
    m_borderWidthSpin->setValue(m_currentTheme.spacing.borderWidth);
}

void ThemeEditor::resetToTheme(ThemeManager::Theme theme)
{
    m_currentTheme = ThemeManager::instance()->getThemeData("");
    
    // Set default name based on theme
    switch (theme) {
        case ThemeManager::Light:
            m_currentTheme.name = "Custom Light Theme";
            break;
        case ThemeManager::Dark:
            m_currentTheme.name = "Custom Dark Theme";
            break;
        default:
            m_currentTheme.name = "Custom Theme";
            break;
    }
    
    m_currentTheme.description = "Custom theme created with Theme Editor";
    m_currentTheme.created = QDateTime::currentDateTime();
    m_currentTheme.modified = m_currentTheme.created;
    
    m_themeNameEdit->setText(m_currentTheme.name);
    m_themeDescriptionEdit->setPlainText(m_currentTheme.description);
    
    updateColorPickers();
    updateTypographyControls();
    updateSpacingControls();
    
    m_isModified = false;
}

void ThemeEditor::applyThemeToPreview()
{
    QString styleSheet = generatePreviewStyleSheet();
    m_previewArea->setStyleSheet(styleSheet);
    m_previewArea->update();
}

QString ThemeEditor::generatePreviewStyleSheet() const
{
    return QString(R"(
        QWidget {
            background-color: %1;
            color: %2;
            font-family: %3;
            font-size: %4pt;
        }
        
        QPushButton {
            background-color: %5;
            color: %1;
            border: %6px solid %7;
            border-radius: %8px;
            padding: %9px;
            min-height: 24px;
        }
        
        QPushButton:hover {
            background-color: %10;
        }
        
        QPushButton:disabled {
            background-color: %11;
            color: %12;
        }
        
        QLineEdit, QComboBox {
            background-color: %1;
            color: %2;
            border: %6px solid %7;
            border-radius: %8px;
            padding: %9px;
        }
        
        QCheckBox, QRadioButton {
            color: %2;
        }
        
        QProgressBar {
            background-color: %1;
            color: %2;
            border: %6px solid %7;
            border-radius: %8px;
            text-align: center;
        }
        
        QProgressBar::chunk {
            background-color: %5;
            border-radius: %13px;
        }
        
        QGroupBox {
            color: %2;
            font-weight: %14;
            border: %6px solid %7;
            border-radius: %8px;
            margin-top: 12px;
            padding-top: 8px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 8px 0 8px;
        }
    )").arg(m_currentTheme.colors.background.name())      // %1
       .arg(m_currentTheme.colors.foreground.name())      // %2
       .arg(m_currentTheme.typography.fontFamily)         // %3
       .arg(m_currentTheme.typography.baseFontSize)       // %4
       .arg(m_currentTheme.colors.accent.name())          // %5
       .arg(m_currentTheme.spacing.borderWidth)           // %6
       .arg(m_currentTheme.colors.border.name())          // %7
       .arg(m_currentTheme.spacing.borderRadius)          // %8
       .arg(m_currentTheme.spacing.padding)               // %9
       .arg(m_currentTheme.colors.hover.name())           // %10
       .arg(m_currentTheme.colors.disabled.name())        // %11
       .arg(m_currentTheme.colors.disabled.darker(150).name()) // %12
       .arg(qMax(1, m_currentTheme.spacing.borderRadius - 1))  // %13
       .arg(m_currentTheme.typography.boldTitles ? "bold" : "normal"); // %14
}