#ifndef SETTINGS_DIALOG_H
#define SETTINGS_DIALOG_H

#include <QDialog>
#include <QTabWidget>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QListWidget>
#include <QDialogButtonBox>

class SettingsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SettingsDialog(QWidget* parent = nullptr);
    ~SettingsDialog();

    // Load/Save settings
    void loadSettings();
    void saveSettings();

signals:
    void settingsChanged();

private slots:
    void onApplyClicked();
    void onOkClicked();
    void onCancelClicked();
    void onRestoreDefaultsClicked();
    void onBrowseLogDirectory();
    void onBrowseBackupDirectory();
    void onAddProtectedPath();
    void onRemoveProtectedPath();
    void onOpenLogDirectory();

private:
    void setupUI();
    void createGeneralTab();
    void createScanningTab();
    void createSafetyTab();
    void createLoggingTab();
    void createAdvancedTab();
    void applyTheme();
    
    // UI Components
    QTabWidget* m_tabWidget;
    QDialogButtonBox* m_buttonBox;
    QPushButton* m_applyButton;
    QPushButton* m_restoreDefaultsButton;
    
    // General Tab
    QWidget* m_generalTab;
    QComboBox* m_languageCombo;
    QComboBox* m_themeCombo;
    QCheckBox* m_startupCheck;
    QCheckBox* m_updateCheck;
    
    // Scanning Tab
    QWidget* m_scanningTab;
    QSpinBox* m_minFileSizeSpin;
    QCheckBox* m_includeHiddenCheck;
    QCheckBox* m_followSymlinksCheck;
    QSpinBox* m_threadCountSpin;
    QSpinBox* m_cacheSizeSpin;
    
    // Safety Tab
    QWidget* m_safetyTab;
    QLineEdit* m_backupLocationEdit;
    QPushButton* m_browseBackupButton;
    QSpinBox* m_backupRetentionSpin;
    QListWidget* m_protectedPathsList;
    QPushButton* m_addPathButton;
    QPushButton* m_removePathButton;
    QCheckBox* m_confirmDeleteCheck;
    QCheckBox* m_confirmMoveCheck;
    
    // Logging Tab
    QWidget* m_loggingTab;
    QComboBox* m_logLevelCombo;
    QCheckBox* m_logToFileCheck;
    QCheckBox* m_logToConsoleCheck;
    QLineEdit* m_logDirectoryEdit;
    QPushButton* m_browseLogButton;
    QPushButton* m_openLogButton;
    QSpinBox* m_maxLogFilesSpin;
    QSpinBox* m_maxLogSizeSpin;
    
    // Advanced Tab
    QWidget* m_advancedTab;
    QLineEdit* m_databaseLocationEdit;
    QLineEdit* m_cacheDirectoryEdit;
    QComboBox* m_exportFormatCombo;
    QCheckBox* m_enablePerformanceCheck;
};

#endif // SETTINGS_DIALOG_H
