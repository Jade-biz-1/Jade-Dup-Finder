#include "theme_recovery_dialog.h"
#include "theme_manager.h"
#include "core/logger.h"
#include <QApplication>
#include <QDesktopServices>
#include <QUrl>
#include <QMessageBox>
#include <QStyle>
#include <QSpacerItem>
#include <QScrollArea>

ThemeRecoveryDialog::ThemeRecoveryDialog(const QString& errorMessage, 
                                       ThemeErrorHandler::ErrorType errorType,
                                       QWidget* parent)
    : QDialog(parent)
    , m_errorMessage(errorMessage)
    , m_errorType(errorType)
    , m_selectedOption(RecoveryOption::ResetToDefault)
    , m_selectedTheme("system")
    , m_advancedOptionsVisible(false)
    , m_progressTimer(new QTimer(this))
{
    setWindowTitle("Theme System Recovery");
    setWindowIcon(style()->standardIcon(QStyle::SP_MessageBoxWarning));
    setModal(true);
    setMinimumSize(500, 400);
    resize(600, 500);
    
    setupUI();
    updateUIForErrorType();
    
    // Connect progress timer
    connect(m_progressTimer, &QTimer::timeout, this, [this]() {
        static int progress = 0;
        progress += 10;
        if (progress <= 100) {
            updateRecoveryProgress(progress, "Applying recovery...");
        } else {
            m_progressTimer->stop();
            showRecoveryResult(true, "Recovery completed successfully");
        }
    });
    
    LOG_INFO(LogCategories::UI, QString("Theme recovery dialog opened for error: %1").arg(errorMessage));
}

void ThemeRecoveryDialog::setupUI()
{
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setSpacing(16);
    m_mainLayout->setContentsMargins(20, 20, 20, 20);
    
    // Error information section
    QHBoxLayout* errorLayout = new QHBoxLayout();
    
    m_errorIcon = new QLabel();
    m_errorIcon->setPixmap(style()->standardIcon(QStyle::SP_MessageBoxWarning).pixmap(48, 48));
    m_errorIcon->setAlignment(Qt::AlignTop);
    errorLayout->addWidget(m_errorIcon);
    
    QVBoxLayout* errorTextLayout = new QVBoxLayout();
    
    m_errorTitle = new QLabel("Theme System Error Detected");
    // Apply theme-aware error styling
    ThemeManager::instance()->applyToWidget(m_errorTitle);
    QFont errorFont = m_errorTitle->font();
    errorFont.setPointSize(14);
    errorFont.setBold(true);
    m_errorTitle->setFont(errorFont);
    // Use theme error color
    QPalette errorPalette = m_errorTitle->palette();
    errorPalette.setColor(QPalette::WindowText, ThemeManager::instance()->getCurrentThemeData().colors.error);
    m_errorTitle->setPalette(errorPalette);
    errorTextLayout->addWidget(m_errorTitle);
    
    m_errorDescription = new QLabel();
    m_errorDescription->setWordWrap(true);
    // Apply theme-aware styling
    ThemeManager::instance()->applyToWidget(m_errorDescription);
    m_errorDescription->setMargin(8);
    errorTextLayout->addWidget(m_errorDescription);
    
    errorLayout->addLayout(errorTextLayout, 1);
    m_mainLayout->addLayout(errorLayout);
    
    // Error details (collapsible)
    m_errorDetails = new QTextEdit();
    m_errorDetails->setMaximumHeight(100);
    m_errorDetails->setReadOnly(true);
    // Apply theme-aware styling
    ThemeManager::instance()->applyToWidget(m_errorDetails);
    QFont monoFont("monospace");
    m_errorDetails->setFont(monoFont);
    m_errorDetails->setPlainText(m_errorMessage);
    m_mainLayout->addWidget(m_errorDetails);
    
    // Recovery options
    setupRecoveryOptions();
    
    // Advanced options (initially hidden)
    setupAdvancedOptions();
    
    // Progress area (initially hidden)
    setupProgressArea();
    
    // Buttons
    m_buttonLayout = new QHBoxLayout();
    
    m_showAdvancedButton = new QPushButton("Advanced Options");
    m_showAdvancedButton->setCheckable(true);
    connect(m_showAdvancedButton, &QPushButton::toggled, this, &ThemeRecoveryDialog::onAdvancedOptionsToggled);
    m_buttonLayout->addWidget(m_showAdvancedButton);
    
    m_buttonLayout->addStretch();
    
    m_cancelButton = new QPushButton("Cancel");
    connect(m_cancelButton, &QPushButton::clicked, this, &ThemeRecoveryDialog::onCancelClicked);
    m_buttonLayout->addWidget(m_cancelButton);
    
    m_tryRecoveryButton = new QPushButton("Apply Recovery");
    m_tryRecoveryButton->setDefault(true);
    // Apply theme-aware styling
    ThemeManager::instance()->applyToWidget(m_tryRecoveryButton);
    QFont buttonFont = m_tryRecoveryButton->font();
    buttonFont.setBold(true);
    m_tryRecoveryButton->setFont(buttonFont);
    m_tryRecoveryButton->setMinimumSize(ThemeManager::instance()->getMinimumControlSize(ThemeManager::ControlType::Button));
    connect(m_tryRecoveryButton, &QPushButton::clicked, this, &ThemeRecoveryDialog::onTryRecoveryClicked);
    m_buttonLayout->addWidget(m_tryRecoveryButton);
    
    m_mainLayout->addLayout(m_buttonLayout);
}

void ThemeRecoveryDialog::setupRecoveryOptions()
{
    m_recoveryOptionsGroup = new QGroupBox("Recovery Options");
    QVBoxLayout* optionsLayout = new QVBoxLayout(m_recoveryOptionsGroup);
    
    m_recoveryButtonGroup = new QButtonGroup(this);
    
    m_resetToDefaultRadio = new QRadioButton("Reset to system default theme (Recommended)");
    m_resetToDefaultRadio->setToolTip("This will reset the theme to the system default and clear any corrupted settings.");
    m_resetToDefaultRadio->setChecked(true);
    m_recoveryButtonGroup->addButton(m_resetToDefaultRadio, static_cast<int>(RecoveryOption::ResetToDefault));
    optionsLayout->addWidget(m_resetToDefaultRadio);
    
    m_retryCurrentRadio = new QRadioButton("Retry applying the current theme");
    m_retryCurrentRadio->setToolTip("Attempt to reapply the current theme settings.");
    m_recoveryButtonGroup->addButton(m_retryCurrentRadio, static_cast<int>(RecoveryOption::RetryCurrentTheme));
    optionsLayout->addWidget(m_retryCurrentRadio);
    
    m_selectDifferentRadio = new QRadioButton("Switch to a different theme");
    m_selectDifferentRadio->setToolTip("Choose a different theme from the available options.");
    m_recoveryButtonGroup->addButton(m_selectDifferentRadio, static_cast<int>(RecoveryOption::SelectDifferentTheme));
    optionsLayout->addWidget(m_selectDifferentRadio);
    
    m_disableThemeRadio = new QRadioButton("Temporarily disable theme system");
    m_disableThemeRadio->setToolTip("Use basic styling without the theme system until the issue is resolved.");
    m_recoveryButtonGroup->addButton(m_disableThemeRadio, static_cast<int>(RecoveryOption::DisableThemeSystem));
    optionsLayout->addWidget(m_disableThemeRadio);
    
    m_contactSupportRadio = new QRadioButton("Get help and report this issue");
    m_contactSupportRadio->setToolTip("Open support resources and report this error for assistance.");
    m_recoveryButtonGroup->addButton(m_contactSupportRadio, static_cast<int>(RecoveryOption::ContactSupport));
    optionsLayout->addWidget(m_contactSupportRadio);
    
    connect(m_recoveryButtonGroup, &QButtonGroup::idClicked,
            this, &ThemeRecoveryDialog::onRecoveryOptionChanged);
    
    m_mainLayout->addWidget(m_recoveryOptionsGroup);
    
    // Theme selection group (initially hidden)
    m_themeSelectionGroup = new QGroupBox("Select Theme");
    QVBoxLayout* themeLayout = new QVBoxLayout(m_themeSelectionGroup);
    
    m_themeButtonGroup = new QButtonGroup(this);
    
    m_systemThemeRadio = new QRadioButton("System Default");
    m_systemThemeRadio->setChecked(true);
    m_themeButtonGroup->addButton(m_systemThemeRadio);
    themeLayout->addWidget(m_systemThemeRadio);
    
    m_lightThemeRadio = new QRadioButton("Light Theme");
    m_themeButtonGroup->addButton(m_lightThemeRadio);
    themeLayout->addWidget(m_lightThemeRadio);
    
    m_darkThemeRadio = new QRadioButton("Dark Theme");
    m_themeButtonGroup->addButton(m_darkThemeRadio);
    themeLayout->addWidget(m_darkThemeRadio);
    
    m_themeSelectionGroup->setVisible(false);
    m_mainLayout->addWidget(m_themeSelectionGroup);
}

void ThemeRecoveryDialog::setupAdvancedOptions()
{
    m_advancedOptionsGroup = new QGroupBox("Advanced Options");
    QVBoxLayout* advancedLayout = new QVBoxLayout(m_advancedOptionsGroup);
    
    m_rememberChoiceCheckBox = new QCheckBox("Remember this choice for future errors");
    m_rememberChoiceCheckBox->setToolTip("Automatically apply this recovery option for similar errors.");
    advancedLayout->addWidget(m_rememberChoiceCheckBox);
    
    m_enableAutoRecoveryCheckBox = new QCheckBox("Enable automatic error recovery");
    m_enableAutoRecoveryCheckBox->setToolTip("Allow the system to automatically attempt recovery without showing this dialog.");
    m_enableAutoRecoveryCheckBox->setChecked(true);
    advancedLayout->addWidget(m_enableAutoRecoveryCheckBox);
    
    m_enableDetailedLoggingCheckBox = new QCheckBox("Enable detailed error logging");
    m_enableDetailedLoggingCheckBox->setToolTip("Log detailed information about theme errors for troubleshooting.");
    advancedLayout->addWidget(m_enableDetailedLoggingCheckBox);
    
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    
    m_testThemeButton = new QPushButton("Test Theme");
    m_testThemeButton->setToolTip("Test the selected theme before applying it.");
    connect(m_testThemeButton, &QPushButton::clicked, this, &ThemeRecoveryDialog::onTestThemeClicked);
    buttonLayout->addWidget(m_testThemeButton);
    
    m_resetSettingsButton = new QPushButton("Reset All Theme Settings");
    m_resetSettingsButton->setToolTip("Reset all theme-related settings to factory defaults.");
    connect(m_resetSettingsButton, &QPushButton::clicked, this, &ThemeRecoveryDialog::onResetSettingsClicked);
    buttonLayout->addWidget(m_resetSettingsButton);
    
    buttonLayout->addStretch();
    advancedLayout->addLayout(buttonLayout);
    
    m_advancedOptionsGroup->setVisible(false);
    m_mainLayout->addWidget(m_advancedOptionsGroup);
}

void ThemeRecoveryDialog::setupProgressArea()
{
    m_progressGroup = new QGroupBox("Recovery Progress");
    QVBoxLayout* progressLayout = new QVBoxLayout(m_progressGroup);
    
    m_progressStatus = new QLabel("Ready to begin recovery...");
    progressLayout->addWidget(m_progressStatus);
    
    m_progressBar = new QProgressBar();
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);
    progressLayout->addWidget(m_progressBar);
    
    m_progressGroup->setVisible(false);
    m_mainLayout->addWidget(m_progressGroup);
}

void ThemeRecoveryDialog::updateUIForErrorType()
{
    QString description = getErrorDescription();
    QString recommendation = getRecoveryRecommendation();
    
    m_errorDescription->setText(QString("%1\n\n%2").arg(description, recommendation));
    
    // Update default selection based on error type
    switch (m_errorType) {
        case ThemeErrorHandler::ErrorType::CustomThemeCorruption:
            m_resetToDefaultRadio->setChecked(true);
            m_retryCurrentRadio->setEnabled(false);
            break;
        case ThemeErrorHandler::ErrorType::StyleApplicationFailure:
            m_retryCurrentRadio->setChecked(true);
            break;
        case ThemeErrorHandler::ErrorType::ThemeLoadFailure:
            m_selectDifferentRadio->setChecked(true);
            break;
        default:
            m_resetToDefaultRadio->setChecked(true);
            break;
    }
}

QString ThemeRecoveryDialog::getErrorDescription() const
{
    switch (m_errorType) {
        case ThemeErrorHandler::ErrorType::ThemeLoadFailure:
            return "The application failed to load the selected theme. This may be due to corrupted theme files or missing resources.";
        case ThemeErrorHandler::ErrorType::StyleApplicationFailure:
            return "The theme was loaded but could not be applied to one or more interface components. Some elements may appear incorrectly styled.";
        case ThemeErrorHandler::ErrorType::CustomThemeCorruption:
            return "A custom theme appears to be corrupted or contains invalid data. The theme system cannot process this theme safely.";
        case ThemeErrorHandler::ErrorType::SystemThemeDetectionFailure:
            return "The application cannot detect the system theme preferences. This may affect automatic theme switching.";
        case ThemeErrorHandler::ErrorType::ComponentRegistrationFailure:
            return "Some interface components failed to register with the theme system. These components may not update when themes change.";
        case ThemeErrorHandler::ErrorType::ValidationFailure:
            return "Theme validation detected issues that may affect accessibility or visual consistency.";
        case ThemeErrorHandler::ErrorType::PersistenceFailure:
            return "The application cannot save or load theme preferences. Your theme selection may not be remembered between sessions.";
        case ThemeErrorHandler::ErrorType::DialogRegistrationFailure:
            return "Dialog windows failed to register with the theme system. Some dialogs may not follow the current theme.";
        default:
            return "An unexpected error occurred in the theme system.";
    }
}

QString ThemeRecoveryDialog::getRecoveryRecommendation() const
{
    switch (m_errorType) {
        case ThemeErrorHandler::ErrorType::ThemeLoadFailure:
        case ThemeErrorHandler::ErrorType::CustomThemeCorruption:
            return "Recommendation: Reset to system default theme to ensure stability.";
        case ThemeErrorHandler::ErrorType::StyleApplicationFailure:
            return "Recommendation: Retry applying the current theme, or switch to a different theme if the problem persists.";
        case ThemeErrorHandler::ErrorType::SystemThemeDetectionFailure:
            return "Recommendation: Manually select a theme instead of using system default.";
        case ThemeErrorHandler::ErrorType::ComponentRegistrationFailure:
        case ThemeErrorHandler::ErrorType::DialogRegistrationFailure:
            return "Recommendation: Restart the application to re-register all components.";
        case ThemeErrorHandler::ErrorType::ValidationFailure:
            return "Recommendation: Switch to a validated theme or reset to default.";
        case ThemeErrorHandler::ErrorType::PersistenceFailure:
            return "Recommendation: Check file permissions and available disk space.";
        default:
            return "Recommendation: Reset to system default theme for best compatibility.";
    }
}

void ThemeRecoveryDialog::onRecoveryOptionChanged()
{
    int selectedId = m_recoveryButtonGroup->checkedId();
    m_selectedOption = static_cast<RecoveryOption>(selectedId);
    
    // Show/hide theme selection based on option
    bool showThemeSelection = (m_selectedOption == RecoveryOption::SelectDifferentTheme);
    m_themeSelectionGroup->setVisible(showThemeSelection);
    
    // Update button text based on selection
    switch (m_selectedOption) {
        case RecoveryOption::ResetToDefault:
            m_tryRecoveryButton->setText("Reset to Default");
            break;
        case RecoveryOption::RetryCurrentTheme:
            m_tryRecoveryButton->setText("Retry Current Theme");
            break;
        case RecoveryOption::SelectDifferentTheme:
            m_tryRecoveryButton->setText("Apply Selected Theme");
            break;
        case RecoveryOption::DisableThemeSystem:
            m_tryRecoveryButton->setText("Disable Theme System");
            break;
        case RecoveryOption::ContactSupport:
            m_tryRecoveryButton->setText("Open Support");
            break;
    }
    
    // Adjust dialog size
    adjustSize();
}

void ThemeRecoveryDialog::onTryRecoveryClicked()
{
    LOG_INFO(LogCategories::UI, QString("User selected recovery option: %1").arg(static_cast<int>(m_selectedOption)));
    
    if (m_selectedOption == RecoveryOption::ContactSupport) {
        // Open support resources
        QMessageBox::information(this, "Support Resources", 
            "Support resources would be opened here.\n\n"
            "This would typically include:\n"
            "• Online documentation\n"
            "• Bug report form\n"
            "• Community forums\n"
            "• Contact information");
        return;
    }
    
    // Update selected theme if applicable
    if (m_selectedOption == RecoveryOption::SelectDifferentTheme) {
        if (m_systemThemeRadio->isChecked()) {
            m_selectedTheme = "system";
        } else if (m_lightThemeRadio->isChecked()) {
            m_selectedTheme = "light";
        } else if (m_darkThemeRadio->isChecked()) {
            m_selectedTheme = "dark";
        }
    }
    
    // Show progress and start recovery
    showRecoveryProgress();
    
    // Apply recovery settings to ThemeErrorHandler
    ThemeErrorHandler::setAutoRecoveryEnabled(m_enableAutoRecoveryCheckBox->isChecked());
    ThemeErrorHandler::setUserNotificationEnabled(!m_rememberChoiceCheckBox->isChecked());
    
    // Start simulated recovery process
    m_progressTimer->start(500);
}

void ThemeRecoveryDialog::onCancelClicked()
{
    LOG_INFO(LogCategories::UI, "User cancelled theme recovery dialog");
    reject();
}

void ThemeRecoveryDialog::onAdvancedOptionsToggled(bool show)
{
    m_advancedOptionsVisible = show;
    m_advancedOptionsGroup->setVisible(show);
    m_showAdvancedButton->setText(show ? "Hide Advanced Options" : "Show Advanced Options");
    adjustSize();
}

void ThemeRecoveryDialog::onTestThemeClicked()
{
    QString testTheme = "system";
    if (m_lightThemeRadio->isChecked()) {
        testTheme = "light";
    } else if (m_darkThemeRadio->isChecked()) {
        testTheme = "dark";
    }
    
    QMessageBox::information(this, "Theme Test", 
        QString("Testing %1 theme...\n\n"
                "In a real implementation, this would:\n"
                "• Temporarily apply the selected theme\n"
                "• Show a preview window\n"
                "• Allow you to confirm or cancel the change").arg(testTheme));
}

void ThemeRecoveryDialog::onResetSettingsClicked()
{
    int result = QMessageBox::warning(this, "Reset Theme Settings",
        "This will reset all theme-related settings to their factory defaults.\n\n"
        "This action cannot be undone. Are you sure you want to continue?",
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    
    if (result == QMessageBox::Yes) {
        LOG_INFO(LogCategories::UI, "User requested theme settings reset");
        
        // Reset ThemeErrorHandler settings
        ThemeErrorHandler::resetErrorCounters();
        ThemeErrorHandler::setAutoRecoveryEnabled(true);
        ThemeErrorHandler::setUserNotificationEnabled(true);
        ThemeErrorHandler::setMaxRetryAttempts(3);
        
        QMessageBox::information(this, "Settings Reset", 
            "Theme settings have been reset to factory defaults.");
    }
}

void ThemeRecoveryDialog::showRecoveryProgress()
{
    // Hide other sections and show progress
    m_recoveryOptionsGroup->setVisible(false);
    m_themeSelectionGroup->setVisible(false);
    m_advancedOptionsGroup->setVisible(false);
    m_progressGroup->setVisible(true);
    
    // Disable buttons during recovery
    m_tryRecoveryButton->setEnabled(false);
    m_cancelButton->setText("Close");
    
    adjustSize();
}

void ThemeRecoveryDialog::updateRecoveryProgress(int percentage, const QString& status)
{
    m_progressBar->setValue(percentage);
    m_progressStatus->setText(status);
}

void ThemeRecoveryDialog::showRecoveryResult(bool success, const QString& message)
{
    m_progressStatus->setText(message);
    
    if (success) {
        // Apply theme-aware success styling
        m_progressStatus->setStyleSheet(ThemeManager::instance()->getStatusIndicatorStyle(ThemeManager::StatusType::Success));
        m_tryRecoveryButton->setText("Close");
        m_tryRecoveryButton->setEnabled(true);
        
        // Auto-close after successful recovery
        QTimer::singleShot(2000, this, &QDialog::accept);
    } else {
        // Apply theme-aware error styling
        m_progressStatus->setStyleSheet(ThemeManager::instance()->getStatusIndicatorStyle(ThemeManager::StatusType::Error));
        m_cancelButton->setEnabled(true);
        
        // Show recovery options again on failure
        QTimer::singleShot(3000, [this]() {
            m_recoveryOptionsGroup->setVisible(true);
            m_progressGroup->setVisible(false);
            m_tryRecoveryButton->setEnabled(true);
            m_tryRecoveryButton->setText("Retry Recovery");
            adjustSize();
        });
    }
}

ThemeRecoveryDialog::RecoveryOption ThemeRecoveryDialog::getSelectedRecoveryOption() const
{
    return m_selectedOption;
}

QString ThemeRecoveryDialog::getSelectedTheme() const
{
    return m_selectedTheme;
}

bool ThemeRecoveryDialog::shouldRememberChoice() const
{
    return m_rememberChoiceCheckBox->isChecked();
}

bool ThemeRecoveryDialog::shouldEnableAutoRecovery() const
{
    return m_enableAutoRecoveryCheckBox->isChecked();
}