#ifndef RESULTS_WINDOW_H
#define RESULTS_WINDOW_H

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QTreeWidgetItem>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTableWidget>
#include <QtCore/QTimer>
#include <QtCore/QDateTime>
#include <QtCore/QSize>
#include <QtGui/QPixmap>
#include <QtGui/QCloseEvent>
#include <QtGui/QShowEvent>

class ResultsWindow : public QMainWindow
{
    Q_OBJECT

public:
    // Window size constants
    static const QSize MIN_WINDOW_SIZE;
    static const QSize DEFAULT_WINDOW_SIZE;

    // Data structures
    struct DuplicateFile {
        QString filePath;
        QString fileName;
        QString directory;
        qint64 fileSize = 0;
        QDateTime lastModified;
        QDateTime created;
        QString hash;
        QPixmap thumbnail;
        bool isSelected = false;
        bool isMarkedForDeletion = false;
        QString fileType;
        
        // Comparison operator for QList::contains
        bool operator==(const DuplicateFile& other) const {
            return filePath == other.filePath;
        }
    };

    struct DuplicateGroup {
        QString groupId;
        QList<DuplicateFile> files;
        int fileCount = 0;
        qint64 totalSize = 0;
        QString primaryFile;  // Recommended file to keep
        bool isExpanded = false;
        bool hasSelection = false;
        
        // Helper methods
        inline void calculateSize() {
            totalSize = 0;
            fileCount = static_cast<int>(files.size());
            for (const auto& file : files) {
                totalSize += file.fileSize;
            }
        }
        
        inline qint64 getWastedSpace() const {
            return (fileCount > 0) ? totalSize - (totalSize / fileCount) : 0;
        }
    };
    
    struct ScanResults {
        QString scanPath;
        QDateTime scanTime;
        QString scanDuration;
        int totalFilesScanned = 0;
        int totalDuplicatesFound = 0;
        qint64 potentialSavings = 0;
        QList<DuplicateGroup> duplicateGroups;
        
        // Helper methods
        inline void calculateTotals() {
            totalDuplicatesFound = 0;
            potentialSavings = 0;
            for (const auto& group : duplicateGroups) {
                totalDuplicatesFound += group.fileCount;
                potentialSavings += group.getWastedSpace();
            }
        }
        
        // Constructor
        ScanResults() = default;
    };

public:
    explicit ResultsWindow(QWidget* parent = nullptr);
    virtual ~ResultsWindow();

    // Public interface
    void displayResults(const ScanResults& results);
    void clearResults();
    void updateProgress(const QString& operation, int percentage);

public slots:
    void refreshResults();
    void exportResults();

signals:
    void windowClosed();
    void fileOperationRequested(const QString& operation, const QStringList& files);
    void resultsUpdated(const ScanResults& results);

protected:
    void closeEvent(QCloseEvent* event) override;
    void showEvent(QShowEvent* event) override;

private slots:
    // UI update slots
    void updateStatusBar();
    void onFileSelectionChanged();
    void updateSelectionSummary();
    
    // Selection actions
    void selectAllDuplicates();
    void selectNoneFiles();
    void selectRecommended();
    void selectBySize(qint64 minSize = 1024 * 1024); // Default 1MB
    void selectByType(const QString& fileType);
    
    // File operations
    void deleteSelectedFiles();
    void moveSelectedFiles();
    void ignoreSelectedFiles();
    void previewSelectedFile();
    void openFileLocation();
    void copyFilePath();
    
    // Bulk operations
    void performBulkDelete();
    void performBulkMove();
    void confirmBulkOperation(const QString& operation, int fileCount, qint64 totalSize);
    
    // Filter and sort
    void onFilterChanged();
    void onSortChanged();
    
    // Tree widget events
    void onGroupExpanded(QTreeWidgetItem* item);
    void onGroupCollapsed(QTreeWidgetItem* item);
    void onGroupSelectionChanged();

private:
    // Initialization methods
    void initializeUI();
    void createHeaderPanel();
    void createMainContent();
    void createResultsTree();
    void createDetailsPanel();
    void createActionsPanel();
    void createStatusBar();
    void createToolBar();
    void setupConnections();
    void applyTheme();

    // Data population methods
    void loadSampleData();
    void populateResultsTree();
    void updateGroupItem(QTreeWidgetItem* groupItem, const DuplicateGroup& group);
    void updateFileItem(QTreeWidgetItem* fileItem, const DuplicateFile& file);

    // Utility methods
    QString formatFileSize(qint64 bytes) const;
    QString getRecommendedFileToKeep(const DuplicateGroup& group) const;
    bool isImageFile(const QString& filePath) const;
    bool isVideoFile(const QString& filePath) const;
    QList<DuplicateFile> getSelectedFiles() const;
    qint64 getSelectedFilesSize() const;
    int getSelectedFilesCount() const;

private:
    // Current state
    ScanResults m_currentResults;

    // Main layout widgets
    QWidget* m_centralWidget;
    QVBoxLayout* m_mainLayout;
    
    // Header panel
    QWidget* m_headerPanel;
    QHBoxLayout* m_headerLayout;
    QLabel* m_titleLabel;
    QLabel* m_summaryLabel;
    QPushButton* m_refreshButton;
    QPushButton* m_exportButton;
    QPushButton* m_settingsButton;
    
    // Main content splitter
    QSplitter* m_mainSplitter;
    
    // Results panel (left)
    QWidget* m_resultsPanel;
    QVBoxLayout* m_resultsPanelLayout;
    
    // Filter controls
    QWidget* m_filterPanel;
    QHBoxLayout* m_filterLayout;
    QLabel* m_filterLabel;
    QLineEdit* m_searchFilter;
    QComboBox* m_sizeFilter;
    QComboBox* m_typeFilter;
    QComboBox* m_sortCombo;
    QPushButton* m_clearFiltersButton;
    
    // Selection controls
    QWidget* m_selectionPanel;
    QHBoxLayout* m_selectionLayout;
    QCheckBox* m_selectAllCheckbox;
    QPushButton* m_selectRecommendedButton;
    QPushButton* m_selectByTypeButton;
    QPushButton* m_clearSelectionButton;
    QLabel* m_selectionSummaryLabel;
    
    // Results tree
    QTreeWidget* m_resultsTree;
    
    // Details panel (middle)
    QWidget* m_detailsPanel;
    QVBoxLayout* m_detailsPanelLayout;
    QTabWidget* m_detailsTabs;
    
    // File info tab
    QWidget* m_fileInfoTab;
    QVBoxLayout* m_fileInfoLayout;
    QScrollArea* m_previewScrollArea;
    QLabel* m_previewLabel;
    QLabel* m_fileNameLabel;
    QLabel* m_fileSizeLabel;
    QLabel* m_filePathLabel;
    QLabel* m_fileDateLabel;
    QLabel* m_fileTypeLabel;
    QLabel* m_fileHashLabel;
    
    // Group info tab
    QWidget* m_groupInfoTab;
    QVBoxLayout* m_groupInfoLayout;
    QLabel* m_groupSummaryLabel;
    QTableWidget* m_groupFilesTable;
    
    // Actions panel (right)
    QWidget* m_actionsPanel;
    QVBoxLayout* m_actionsPanelLayout;
    
    // File actions group
    QGroupBox* m_fileActionsGroup;
    QVBoxLayout* m_fileActionsLayout;
    QPushButton* m_deleteButton;
    QPushButton* m_moveButton;
    QPushButton* m_ignoreButton;
    QPushButton* m_previewButton;
    QPushButton* m_openLocationButton;
    QPushButton* m_copyPathButton;
    
    // Bulk actions group
    QGroupBox* m_bulkActionsGroup;
    QVBoxLayout* m_bulkActionsLayout;
    QPushButton* m_bulkDeleteButton;
    QPushButton* m_bulkMoveButton;
    QPushButton* m_bulkIgnoreButton;
    
    // Status bar
    QLabel* m_statusLabel;
    QProgressBar* m_progressBar;
    QLabel* m_statisticsLabel;
    
    // Utilities
    QTimer* m_thumbnailTimer;
    bool m_isProcessingBulkOperation;
};

#endif // RESULTS_WINDOW_H