#ifndef SCAN_DIALOG_H
#define SCAN_DIALOG_H

#include <QtWidgets/QDialog>
#include <QtWidgets/QWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QTreeWidgetItem>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QLabel>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QFrame>
#include <QtCore/QTimer>

// Forward declarations
class FileScanner;

class ScanSetupDialog : public QDialog
{
    Q_OBJECT

public:
    enum class DetectionMode {
        Quick,      // Size + filename matching
        Smart,      // Adaptive based on file types (default)
        Deep,       // Size + hash comparison  
        Media       // Deep + metadata comparison for media files
    };
    
    enum class FileTypeFilter {
        All = 0,
        Images = 1,
        Documents = 2,
        Videos = 4,
        Audio = 8,
        Archives = 16
    };

    struct ScanConfiguration {
        QStringList targetPaths;
        DetectionMode detectionMode;
        qint64 minimumFileSize;
        int maximumDepth;
        QStringList excludePatterns;
        QStringList excludeFolders;  // New: folders to exclude from scanning
        bool includeHidden;
        bool includeSystem;
        bool followSymlinks;
        bool scanArchives;
        FileTypeFilter fileTypeFilter;
        
        // Validation
        bool isValid() const;
        QString validationError() const;
    };

    struct EstimatedStats {
        int estimatedFiles;
        qint64 estimatedSize;
        bool withinLimits;
        QString warningMessage;
    };

    explicit ScanSetupDialog(QWidget* parent = nullptr);
    ~ScanSetupDialog();

    // Configuration management
    void setConfiguration(const ScanConfiguration& config);
    ScanConfiguration getCurrentConfiguration() const;
    
    // Preset management
    void loadPreset(const QString& presetName);
    void saveCurrentAsPreset(const QString& presetName);
    QStringList getAvailablePresets() const;

public slots:
    void updateEstimates();
    void resetToDefaults();

signals:
    void scanConfigured(const ScanConfiguration& config);
    void presetSaved(const QString& name, const ScanConfiguration& config);
    void estimatesUpdated(const EstimatedStats& stats);

protected:
    void showEvent(QShowEvent* event) override;

private slots:
    void onDirectoryItemChanged(QTreeWidgetItem* item, int column);
    void onOptionsChanged();
    void addFolder();
    void removeSelectedFolder();
    void startScan();
    void savePreset();
    void showUpgradeDialog();
    void performEstimation();
    
    // Preset slots
    void applyDownloadsPreset();
    void applyPhotosPreset();
    void applyDocumentsPreset();
    void applyMediaPreset();
    void applyCustomPreset();
    void applyFullSystemPreset();
    
    // File type slots
    void onAllTypesToggled(bool checked);
    void onFileTypeChanged();
    
    // Folder exclusion slots
    void addExcludeFolder();
    void removeSelectedExcludeFolder();
    void onExcludeFolderItemChanged(QTreeWidgetItem* item, int column);

private:
    void setupUI();
    void createLocationsPanel();
    void createOptionsPanel();
    void createPreviewPanel();
    void createButtonBar();
    void setupConnections();
    void applyTheme();
    
    // Utility methods
    void populateDirectoryTree();
    void updatePreviewPanel();
    void showEstimationProgress(bool show);
    EstimatedStats calculateEstimates() const;
    ScanConfiguration getBuiltInPreset(const QString& presetName) const;
    void clearAllSelections();
    void selectPath(const QString& path);
    QString formatFileSize(qint64 bytes) const;
    
    // UI Components
    QHBoxLayout* m_mainLayout;
    
    // Locations Panel
    QGroupBox* m_locationsGroup;
    QVBoxLayout* m_locationsLayout;
    QTreeWidget* m_directoryTree;
    QHBoxLayout* m_directoryButtonsLayout;
    QPushButton* m_addFolderButton;
    QPushButton* m_removeFolderButton;
    
    // Quick Presets
    QWidget* m_presetsWidget;
    QGridLayout* m_presetsLayout;
    QPushButton* m_downloadsButton;
    QPushButton* m_photosButton;
    QPushButton* m_documentsButton;
    QPushButton* m_mediaButton;
    QPushButton* m_customButton;
    QPushButton* m_fullSystemButton;
    
    // Options Panel
    QGroupBox* m_optionsGroup;
    QVBoxLayout* m_optionsLayout;
    QComboBox* m_detectionMode;
    QSpinBox* m_minimumSize;
    QComboBox* m_maxDepth;
    QCheckBox* m_includeHidden;
    QCheckBox* m_includeSystem;
    QCheckBox* m_followSymlinks;
    QCheckBox* m_scanArchives;
    QLineEdit* m_excludePatterns;
    
    // Exclude folders
    QTreeWidget* m_excludeFoldersTree;
    QPushButton* m_addExcludeFolderButton;
    QPushButton* m_removeExcludeFolderButton;
    
    // File type filters
    QWidget* m_fileTypesWidget;
    QCheckBox* m_allTypesCheck;
    QCheckBox* m_imagesCheck;
    QCheckBox* m_documentsCheck;
    QCheckBox* m_videosCheck;
    QCheckBox* m_audioCheck;
    QCheckBox* m_archivesCheck;
    
    // Preview Panel
    QGroupBox* m_previewGroup;
    QVBoxLayout* m_previewLayout;
    QLabel* m_estimateLabel;
    QProgressBar* m_estimationProgress;
    QLabel* m_limitWarning;
    QPushButton* m_upgradeButton;
    
    // Button Bar
    QDialogButtonBox* m_buttonBox;
    QPushButton* m_startScanButton;
    QPushButton* m_savePresetButton;
    QPushButton* m_cancelButton;
    
    // Utilities
    QTimer* m_estimationTimer;
    
    // Data
    ScanConfiguration m_currentConfig;
    EstimatedStats m_currentStats;
    bool m_estimationInProgress;
    
    // Constants
    static const qint64 FREE_TIER_FILE_LIMIT = 10000;
    static const qint64 FREE_TIER_SIZE_LIMIT = 100LL * 1024 * 1024 * 1024; // 100GB
};

Q_DECLARE_METATYPE(ScanSetupDialog::ScanConfiguration)
Q_DECLARE_METATYPE(ScanSetupDialog::EstimatedStats)

#endif // SCAN_DIALOG_H