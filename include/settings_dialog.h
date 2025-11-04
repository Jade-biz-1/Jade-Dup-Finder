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
    void onThemeChanged();
    void onGPUTestClicked();  // Task 27: GPU test functionality

private:
    void setupUI();
    void createGeneralTab();
    void createScanningTab();
    void createSafetyTab();
    void createLoggingTab();
    void createAdvancedTab();
    void createUIFeaturesTab();  // Task 32: New UI features settings
    void createGPUTab();  // Task 27: GPU acceleration settings
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
    
    // UI Features Tab (Task 32)
    QWidget* m_uiFeaturesTab;
    QSpinBox* m_thumbnailSizeSpin;
    QSpinBox* m_thumbnailCacheSizeSpin;
    QCheckBox* m_enableThumbnailsCheck;
    QSpinBox* m_operationQueueSizeSpin;
    QSpinBox* m_selectionHistorySizeSpin;
    QCheckBox* m_enableAdvancedFiltersCheck;
    QCheckBox* m_enableSmartSelectionCheck;
    QCheckBox* m_enableOperationQueueCheck;
    QCheckBox* m_showDetailedProgressCheck;
    QLineEdit* m_cacheDirectoryEdit;
    QComboBox* m_exportFormatCombo;
    QCheckBox* m_enablePerformanceCheck;
    
    // GPU Tab (Task 27)
    QWidget* m_gpuTab;
    QCheckBox* m_enableGPUCheck;
    QCheckBox* m_preferCUDACheck;
    QCheckBox* m_gpuFallbackCheck;
    QSpinBox* m_gpuMemoryLimitSpin;
    QSpinBox* m_gpuBatchSizeSpin;
    QCheckBox* m_gpuCachingCheck;
    QLabel* m_gpuStatusLabel;
    QPushButton* m_gpuTestButton;
};

#endif // SETTINGS_DIALOG_H
