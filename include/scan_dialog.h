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
#include <QtWidgets/QSlider>
#include <QtCore/QTimer>
#include <QtCore/QThread>  // T11: For thread count

// Forward declarations
class FileScanner;
class ExcludePatternWidget;
class ScanScopePreviewWidget;

class ScanSetupDialog : public QDialog
{
    Q_OBJECT

public:
    enum class DetectionMode {
        ExactHash,      // SHA-256 hash-based exact matching (most accurate)
        QuickScan,      // Size + filename matching (fastest)
        PerceptualHash, // Image similarity detection (for photos)
        DocumentSimilarity, // Document content similarity (for documents)
        Smart           // Auto-select best algorithm based on file types (default)
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
        qint64 maximumFileSize;         // T11: Maximum file size limit
        int maximumDepth;
        QStringList excludePatterns;
        QStringList excludeFolders;  // New: folders to exclude from scanning
        bool includeHidden;
        bool includeSystem;
        bool followSymlinks;
        bool scanArchives;
        FileTypeFilter fileTypeFilter;
        
        // T11: Advanced options
        int threadCount;                // Number of threads for scanning
        bool enableCaching;             // Enable hash caching
        bool skipEmptyFiles;            // Skip zero-byte files
        bool skipDuplicateNames;        // Skip files with identical names
        int hashAlgorithm;              // 0=MD5, 1=SHA1, 2=SHA256
        bool enablePrefiltering;        // Enable size-based prefiltering
        
        // T11: Performance options
        int bufferSize;                 // I/O buffer size in KB
        bool useMemoryMapping;          // Use memory-mapped files
        bool enableParallelHashing;     // Parallel hash calculation
        
        // Advanced Detection Algorithm Options (Phase 2)
        double similarityThreshold;     // Similarity threshold for perceptual/document algorithms (0.0-1.0)
        bool enableAutoAlgorithmSelection; // Auto-select best algorithm for each file type
        QString algorithmPreset;        // Algorithm preset: "Fast", "Balanced", "Thorough"
        
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
    void openPresetManager();

public slots:
    void updateEstimates();
    void resetToDefaults();
    void validateConfiguration();

signals:
    void scanConfigured(const ScanConfiguration& config);
    void presetSaved(const QString& name, const ScanConfiguration& config);
    void estimatesUpdated(const EstimatedStats& stats);
    void validationChanged(bool isValid, const QString& error);

protected:
    void showEvent(QShowEvent* event) override;

private slots:
    void onDirectoryItemChanged(QTreeWidgetItem* item, int column);
    void onOptionsChanged();
    void addFolder();
    void removeSelectedFolder();
    void startScan();
    void savePreset();
    void managePresets();
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
    
    // Algorithm configuration slots (Phase 2)
    void updateAlgorithmDescription();
    void onSimilarityThresholdChanged(int value);
    void onAlgorithmPresetChanged(int index);
    void showAlgorithmHelp();
    
    // Folder exclusion slots
    void addExcludeFolder();
    void removeSelectedExcludeFolder();
    void onExcludeFolderItemChanged(QTreeWidgetItem* item, int column);

private:
    void setupUI();
    void createLocationsPanel();
    void createOptionsPanel();
    void createAdvancedOptionsPanel();     // T11: Advanced options
    void createPerformanceOptionsPanel();  // T11: Performance options
    void createAlgorithmConfigPanel();      // Phase 2: Algorithm configuration
    void createPreviewPanel();
    void createButtonBar();
    void setupConnections();
    void applyTheme();
    void updateValidationDisplay();        // T11: Real-time validation
    
    // Utility methods
    void populateDirectoryTree();
    void updatePreviewPanel();
    void updateScopePreview();
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
    QSpinBox* m_maximumSize;            // T11: Maximum file size
    QComboBox* m_maxDepth;
    QCheckBox* m_includeHidden;
    QCheckBox* m_includeSystem;
    QCheckBox* m_followSymlinks;
    QCheckBox* m_scanArchives;
    QLineEdit* m_excludePatterns;
    ExcludePatternWidget* m_excludePatternWidget;
    
    // T11: Advanced Options Panel
    QGroupBox* m_advancedGroup;
    QVBoxLayout* m_advancedLayout;
    QSpinBox* m_threadCount;
    QCheckBox* m_enableCaching;
    QCheckBox* m_skipEmptyFiles;
    QCheckBox* m_skipDuplicateNames;
    QComboBox* m_hashAlgorithm;
    QCheckBox* m_enablePrefiltering;
    
    // T11: Performance Options Panel
    QGroupBox* m_performanceGroup;
    QVBoxLayout* m_performanceLayout;
    QSpinBox* m_bufferSize;
    QCheckBox* m_useMemoryMapping;
    QCheckBox* m_enableParallelHashing;
    
    // Algorithm Configuration Panel (Phase 2)
    QGroupBox* m_algorithmGroup;
    QVBoxLayout* m_algorithmLayout;
    QLabel* m_algorithmDescription;
    QSlider* m_similarityThreshold;
    QLabel* m_similarityLabel;
    QCheckBox* m_autoAlgorithmSelection;
    QComboBox* m_algorithmPreset;
    QPushButton* m_algorithmHelp;
    
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
    QLabel* m_validationLabel;
    QPushButton* m_upgradeButton;
    ScanScopePreviewWidget* m_scopePreviewWidget;
    
    // Button Bar
    QDialogButtonBox* m_buttonBox;
    QPushButton* m_startScanButton;
    QPushButton* m_savePresetButton;
    QPushButton* m_managePresetsButton;
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