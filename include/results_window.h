#ifndef RESULTS_WINDOW_H
#define RESULTS_WINDOW_H

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QTreeWidgetItem>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTableWidgetItem>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QFrame>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QTabWidget>
#include <QtCore/QTimer>
#include <QtCore/QFileInfo>
#include <QtCore/QDateTime>
#include <QtGui/QPixmap>
#include <QtGui/QIcon>


// Include headers for types used in method signatures
#include "duplicate_detector.h"
#include "duplicate_relationship_widget.h"
#include "smart_selection_dialog.h"
#include "settings_dialog.h"
#include "grouping_options_dialog.h"

// Forward declarations
class ScanSetupDialog;
class FileManager;
class ThumbnailCache;
class ThumbnailDelegate;
class DuplicateRelationshipWidget;
class SmartSelectionDialog;

class ResultsWindow : public QMainWindow
{
    Q_OBJECT

public:
    // Duplicate file information structure
    struct DuplicateFile {
        QString filePath;
        QString fileName;
        QString directory;
        qint64 fileSize;
        QDateTime lastModified;
        QDateTime created;
        QString hash;
        QPixmap thumbnail;      // For image files
        bool isSelected;
        bool isMarkedForDeletion;
        QString fileType;       // Extension/MIME type
        
        bool operator==(const DuplicateFile& other) const {
            return filePath == other.filePath;
        }
    };
    
    // Group of duplicate files
    struct DuplicateGroup {
        QString groupId;
        QList<DuplicateFile> files;
        qint64 totalSize;
        qint64 wastedSpace;     // Size that could be saved
        int fileCount;
        QString primaryFile;    // Recommended file to keep
        bool isExpanded;
        bool hasSelection;
        
        qint64 getWastedSpace() const {
            return fileCount > 1 ? totalSize * (fileCount - 1) : 0;
        }
    };
    
    // Scan results summary
    struct ScanResults {
        QList<DuplicateGroup> duplicateGroups;
        qint64 totalFilesScanned;
        qint64 totalDuplicatesFound;
        qint64 totalSpaceWasted;
        qint64 potentialSavings;
        QString scanPath;
        QDateTime scanTime;
        QString scanDuration;
        
        void calculateTotals() {
            totalDuplicatesFound = 0;
            totalSpaceWasted = 0;
            for (const auto& group : duplicateGroups) {
                totalDuplicatesFound += group.fileCount;
                totalSpaceWasted += group.getWastedSpace();
            }
            potentialSavings = totalSpaceWasted;
        }
    };

    explicit ResultsWindow(QWidget* parent = nullptr);
    ~ResultsWindow();

    // Main interface
    void displayResults(const ScanResults& results);
    void displayDuplicateGroups(const QList<DuplicateDetector::DuplicateGroup>& groups);
    void clearResults();
    void updateProgress(const QString& operation, int percentage);
    void setFileManager(FileManager* fileManager);
    
    // Thumbnail support
    void enableThumbnails(bool enable);
    void setThumbnailSize(int size);
    void preloadVisibleThumbnails();
    
    // Relationship visualization
    void showRelationshipVisualization(bool show);
    void updateRelationshipVisualization();
    void highlightFileInVisualization(const QString& filePath);
    
    // Smart selection
    void showSmartSelectionDialog();
    void applySmartSelection(const SmartSelectionDialog::SelectionCriteria& criteria);
    QStringList selectFilesByCriteria(const SmartSelectionDialog::SelectionCriteria& criteria);
    
    // Smart selection helper methods
    QStringList selectOldestFiles(const QList<QPair<QString, QDateTime>>& fileDatePairs, int maxFiles);
    QStringList selectNewestFiles(const QList<QPair<QString, QDateTime>>& fileDatePairs, int maxFiles);
    QStringList selectLargestFiles(const QList<QPair<QString, qint64>>& fileSizePairs, int maxFiles);
    QStringList selectSmallestFiles(const QList<QPair<QString, qint64>>& fileSizePairs, int maxFiles);
    QStringList selectByPathPattern(const QStringList& files, const QString& pattern);
    QStringList selectByFileType(const QStringList& files, const QStringList& fileTypes);
    QStringList selectByLocationPattern(const QStringList& files, const QStringList& patterns);
    QStringList selectByCombinedCriteria(const QStringList& files, const SmartSelectionDialog::SelectionCriteria& criteria);
    QStringList applyAdditionalFilters(const QStringList& files, const SmartSelectionDialog::SelectionCriteria& criteria);
    bool matchesDateRange(const QString& filePath, const SmartSelectionDialog::SelectionCriteria& criteria);
    bool matchesSizeRange(const QString& filePath, const SmartSelectionDialog::SelectionCriteria& criteria);
    bool matchesFileTypes(const QString& filePath, const SmartSelectionDialog::SelectionCriteria& criteria);
    bool matchesLocationPatterns(const QString& filePath, const SmartSelectionDialog::SelectionCriteria& criteria);
    
    // Selection and actions
    int getSelectedFilesCount() const;
    qint64 getSelectedFilesSize() const;
    QList<DuplicateFile> getSelectedFiles() const;

public slots:
    void refreshResults();
    void exportResults();
    void selectAllDuplicates();
    void selectNoneFiles();
    void selectBySize(qint64 minSize);
    void selectByType(const QString& fileType);
    void selectRecommended();
    void showAdvancedFilterDialog();
    void showSmartSelectionDialog();

signals:
    void filesDeleted(const QStringList& filePaths);
    void filesMoved(const QStringList& filePaths, const QString& destination);
    void resultsExported(const QString& filePath);
    void windowClosed();

protected:
    void closeEvent(QCloseEvent* event) override;
    void showEvent(QShowEvent* event) override;


private slots:
    void initializeUI();
    void setupConnections();
    void applyTheme();
    
    // Results display
    void onGroupExpanded(QTreeWidgetItem* item);
    void onGroupCollapsed(QTreeWidgetItem* item);
    void onFileSelectionChanged();
    void onGroupSelectionChanged();
    void updateSelectionSummary();
    
    // Undo/Redo operations
    void onUndoRequested();
    void onRedoRequested();
    void onInvertSelection();
    
    // Grouping operations
    void showGroupingOptions();
    void applyGrouping(const GroupingOptionsDialog::GroupingOptions& options);
    
    // Selection operations
    void recordSelectionState(const QString& operation);
    
    // Operation queue
    void setupOperationQueue();
    
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
    
    // Filtering and sorting
    void filterResults();
    void sortResults();
    void onFilterChanged();
    void onSortChanged();
    
    // Progress and status
    void updateStatusBar();
    void showProgressDialog(const QString& title);
    void hideProgressDialog();

private:
    // UI Creation methods
    void createHeaderPanel();
    void createMainContent();
    void createResultsTree();
    void createDetailsPanel();
    void createActionsPanel();
    void createStatusBar();
    void createToolBar();
    
    // Utility methods
    void populateResultsTree();
    void updateGroupItem(QTreeWidgetItem* groupItem, const DuplicateGroup& group);
    void updateFileItem(QTreeWidgetItem* fileItem, const DuplicateFile& file);
    void convertDetectorGroupToDisplayGroup(const DuplicateDetector::DuplicateGroup& source, 
                                            DuplicateGroup& target);
    void updateStatisticsDisplay();
    void removeFilesFromDisplay(const QStringList& filePaths);
    void generateFileThumbnail(DuplicateFile& file);
    QString formatFileSize(qint64 bytes) const;
    QString getFileIcon(const QString& filePath) const;
    QPixmap createThumbnail(const QString& filePath, const QSize& size = QSize(64, 64));
    void selectGroupFiles(const DuplicateGroup& group, bool select);


    
    // Data management
    void loadSampleData();  // For testing purposes
    bool isImageFile(const QString& filePath) const;
    bool isVideoFile(const QString& filePath) const;
    bool isTextFile(const QString& filePath) const;
    QString getRecommendedFileToKeep(const DuplicateGroup& group) const;
    
    // Preview helper methods
    void previewImageFile(const QString& filePath);
    void previewTextFile(const QString& filePath);
    void showFileInfo(const QString& filePath);
    
    // Export helper methods
    bool exportToCSV(QTextStream& out);
    bool exportToJSON(QTextStream& out);
    bool exportToText(QTextStream& out);
    bool exportToHTML(QTextStream& out, const QString& fileName);
    QString generateThumbnailForExport(const QString& filePath, const QString& thumbnailDir, const QString& baseName);

    // UI Components
    QWidget* m_centralWidget;
    QVBoxLayout* m_mainLayout;
    
    // Header Panel
    QWidget* m_headerPanel;
    QHBoxLayout* m_headerLayout;
    QLabel* m_titleLabel;
    QLabel* m_summaryLabel;
    QPushButton* m_refreshButton;
    QPushButton* m_exportButton;
    QPushButton* m_settingsButton;
    
    // Main Content (Splitter)
    QSplitter* m_mainSplitter;
    
    // Results Tree (Left side)
    QWidget* m_resultsPanel;
    QVBoxLayout* m_resultsPanelLayout;
    QTreeWidget* m_resultsTree;
    
    // Filter Panel
    QWidget* m_filterPanel;
    QHBoxLayout* m_filterLayout;
    QLabel* m_filterLabel;
    QLineEdit* m_searchFilter;
    QComboBox* m_sizeFilter;
    QComboBox* m_typeFilter;
    QComboBox* m_sortCombo;
    QPushButton* m_clearFiltersButton;
    
    // Selection Panel
    QWidget* m_selectionPanel;
    QHBoxLayout* m_selectionLayout;
    QCheckBox* m_selectAllCheckbox;
    QPushButton* m_selectRecommendedButton;
    QPushButton* m_selectByTypeButton;
    QPushButton* m_clearSelectionButton;
    QLabel* m_selectionSummaryLabel;
    
    // Details Panel (Right side)
    QWidget* m_detailsPanel;
    QVBoxLayout* m_detailsPanelLayout;
    QTabWidget* m_detailsTabs;
    
    // File Info Tab
    QWidget* m_fileInfoTab;
    QVBoxLayout* m_fileInfoLayout;
    QLabel* m_previewLabel;
    QScrollArea* m_previewScrollArea;
    QLabel* m_fileNameLabel;
    QLabel* m_fileSizeLabel;
    QLabel* m_filePathLabel;
    QLabel* m_fileDateLabel;
    QLabel* m_fileTypeLabel;
    QLabel* m_fileHashLabel;
    
    // Group Info Tab
    QWidget* m_groupInfoTab;
    QVBoxLayout* m_groupInfoLayout;
    QLabel* m_groupSummaryLabel;
    QTableWidget* m_groupFilesTable;
    
    // Actions Panel
    QWidget* m_actionsPanel;
    QVBoxLayout* m_actionsPanelLayout;
    QGroupBox* m_fileActionsGroup;
    QVBoxLayout* m_fileActionsLayout;
    QPushButton* m_deleteButton;
    QPushButton* m_moveButton;
    QPushButton* m_ignoreButton;
    QPushButton* m_previewButton;
    QPushButton* m_openLocationButton;
    QPushButton* m_copyPathButton;
    
    // Bulk Operations Group
    QGroupBox* m_bulkActionsGroup;
    QVBoxLayout* m_bulkActionsLayout;
    QPushButton* m_bulkDeleteButton;
    QPushButton* m_bulkMoveButton;
    QPushButton* m_bulkIgnoreButton;
    
    // Progress and Status
    QLabel* m_statusLabel;
    QProgressBar* m_progressBar;
    QLabel* m_statisticsLabel;
    
    // Data
    ScanResults m_currentResults;
    QList<DuplicateFile> m_selectedFiles;
    QTimer* m_thumbnailTimer;
    FileManager* m_fileManager;
    ThumbnailCache* m_thumbnailCache;
    ThumbnailDelegate* m_thumbnailDelegate;
    DuplicateRelationshipWidget* m_relationshipWidget;
    SmartSelectionDialog* m_smartSelectionDialog;

    
    // State
    bool m_isProcessingBulkOperation;
    QString m_lastExportPath;
    
    // Constants
    static const int THUMBNAIL_SIZE = 64;
    static const int MAX_THUMBNAILS_PER_BATCH = 10;
    static const QSize MIN_WINDOW_SIZE;
    static const QSize DEFAULT_WINDOW_SIZE;
};

// Helper widget for custom duplicate group display
class DuplicateGroupWidget : public QWidget
{
    Q_OBJECT
    
public:
    explicit DuplicateGroupWidget(const ResultsWindow::DuplicateGroup& group, QWidget* parent = nullptr);
    
    void updateGroup(const ResultsWindow::DuplicateGroup& group);
    void setExpanded(bool expanded);
    bool isExpanded() const { return m_isExpanded; }
    
    const ResultsWindow::DuplicateGroup& getGroup() const { return m_group; }
    
signals:
    void expansionToggled(bool expanded);
    void fileSelectionChanged(const QString& filePath, bool selected);
    void groupSelectionChanged(bool selected);
    
private slots:
    void onExpandButtonClicked();
    void onFileCheckboxToggled(bool checked);
    void updateDisplay();
    
private:
    void createGroupHeader();
    void createFilesList();
    void updateGroupHeader();
    void updateFilesList();
    
    ResultsWindow::DuplicateGroup m_group;
    bool m_isExpanded;
    
    // UI Components
    QVBoxLayout* m_layout;
    QWidget* m_headerWidget;
    QHBoxLayout* m_headerLayout;
    QPushButton* m_expandButton;
    QCheckBox* m_groupCheckbox;
    QLabel* m_groupIcon;
    QLabel* m_groupTitle;
    QLabel* m_groupStats;
    
    QWidget* m_filesWidget;
    QVBoxLayout* m_filesLayout;
    QList<QCheckBox*> m_fileCheckboxes;
};

#endif // RESULTS_WINDOW_H