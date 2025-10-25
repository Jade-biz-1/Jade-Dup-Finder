#ifndef THEME_RECOVERY_DIALOG_H
#define THEME_RECOVERY_DIALOG_H

#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include <QCheckBox>
#include <QProgressBar>
#include <QTimer>
#include <QGroupBox>
#include <QRadioButton>
#include <QButtonGroup>
#include "theme_error_handler.h"

class ThemeRecoveryDialog : public QDialog
{
    Q_OBJECT

public:
    enum class RecoveryOption {
        ResetToDefault,
        RetryCurrentTheme,
        SelectDifferentTheme,
        DisableThemeSystem,
        ContactSupport
    };

    explicit ThemeRecoveryDialog(const QString& errorMessage, 
                                ThemeErrorHandler::ErrorType errorType,
                                QWidget* parent = nullptr);
    ~ThemeRecoveryDialog() = default;

    RecoveryOption getSelectedRecoveryOption() const;
    QString getSelectedTheme() const;
    bool shouldRememberChoice() const;
    bool shouldEnableAutoRecovery() const;

public slots:
    void showRecoveryProgress();
    void updateRecoveryProgress(int percentage, const QString& status);
    void showRecoveryResult(bool success, const QString& message);

private slots:
    void onRecoveryOptionChanged();
    void onTryRecoveryClicked();
    void onCancelClicked();
    void onAdvancedOptionsToggled(bool show);
    void onTestThemeClicked();
    void onResetSettingsClicked();

private:
    void setupUI();
    void setupRecoveryOptions();
    void setupAdvancedOptions();
    void setupProgressArea();
    void updateUIForErrorType();
    void populateThemeList();
    QString getErrorDescription() const;
    QString getRecoveryRecommendation() const;

    // UI Components
    QVBoxLayout* m_mainLayout;
    QLabel* m_errorIcon;
    QLabel* m_errorTitle;
    QLabel* m_errorDescription;
    QTextEdit* m_errorDetails;
    
    QGroupBox* m_recoveryOptionsGroup;
    QButtonGroup* m_recoveryButtonGroup;
    QRadioButton* m_resetToDefaultRadio;
    QRadioButton* m_retryCurrentRadio;
    QRadioButton* m_selectDifferentRadio;
    QRadioButton* m_disableThemeRadio;
    QRadioButton* m_contactSupportRadio;
    
    QGroupBox* m_themeSelectionGroup;
    QButtonGroup* m_themeButtonGroup;
    QRadioButton* m_lightThemeRadio;
    QRadioButton* m_darkThemeRadio;
    QRadioButton* m_systemThemeRadio;
    
    QGroupBox* m_advancedOptionsGroup;
    QCheckBox* m_rememberChoiceCheckBox;
    QCheckBox* m_enableAutoRecoveryCheckBox;
    QCheckBox* m_enableDetailedLoggingCheckBox;
    QPushButton* m_testThemeButton;
    QPushButton* m_resetSettingsButton;
    QPushButton* m_showAdvancedButton;
    
    QGroupBox* m_progressGroup;
    QProgressBar* m_progressBar;
    QLabel* m_progressStatus;
    
    QHBoxLayout* m_buttonLayout;
    QPushButton* m_tryRecoveryButton;
    QPushButton* m_cancelButton;
    
    // Data
    QString m_errorMessage;
    ThemeErrorHandler::ErrorType m_errorType;
    RecoveryOption m_selectedOption;
    QString m_selectedTheme;
    bool m_advancedOptionsVisible;
    QTimer* m_progressTimer;
};

#endif // THEME_RECOVERY_DIALOG_H