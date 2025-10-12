#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QFrame>
#include <QtWidgets/QScrollArea>
#include <QtCore/QTimer>

// Include headers for types used in method signatures
#include "file_scanner.h"
#include "duplicate_detector.h"

// Forward declarations
class HashCalculator;
class SafetyManager;
class FileManager;
class ScanSetupDialog;
class ResultsWindow;

class QuickActionsWidget;
class ScanHistoryWidget;
class SystemOverviewWidget;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    // Integration with core components
    void setFileScanner(FileScanner* scanner);
    void setHashCalculator(HashCalculator* calculator);
    void setDuplicateDetector(DuplicateDetector* detector);
    void setSafetyManager(SafetyManager* manager);
    void setFileManager(FileManager* manager);

    // Status management
    void updateScanProgress(int percentage, const QString& status);
    void showScanResults();
    void showError(const QString& title, const QString& message);
    void showSuccess(const QString& title, const QString& message);

public slots:
    void onNewScanRequested();
    void onPresetSelected(const QString& preset);
    void onSettingsRequested();
    void onHelpRequested();
    void updateSystemInfo();
    void onScanHistoryItemClicked(int index);
    void onViewAllHistoryClicked();

signals:
    void scanRequested(const QString& preset);
    void settingsRequested();
    void helpRequested();
    void applicationExit();

protected:
    void closeEvent(QCloseEvent* event) override;
    void changeEvent(QEvent* event) override;

private slots:
    void initializeUI();
    void setupConnections();
    void updatePlanIndicator();
    void refreshSystemStats();
    void handleScanConfiguration();
    
    // Duplicate detection handlers
    void onScanCompleted();
    void onDuplicateDetectionStarted(int totalFiles);
    void onDuplicateDetectionProgress(const DuplicateDetector::DetectionProgress& progress);
    void onDuplicateDetectionCompleted(int totalGroups);
    void onDuplicateDetectionError(const QString& error);

private:
    // UI Components
    QWidget* m_centralWidget;
    QVBoxLayout* m_mainLayout;
    
    // Header
    QWidget* m_headerWidget;
    QHBoxLayout* m_headerLayout;
    QPushButton* m_newScanButton;
    QPushButton* m_settingsButton;
    QPushButton* m_helpButton;
    QLabel* m_notificationIndicator;
    QLabel* m_planIndicator;
    
    // Content Areas
    QuickActionsWidget* m_quickActions;
    ScanHistoryWidget* m_scanHistory;
    SystemOverviewWidget* m_systemOverview;
    
    // Status
    QStatusBar* m_statusBar;
    QProgressBar* m_progressBar;
    QLabel* m_statusLabel;
    QLabel* m_fileCountLabel;
    QLabel* m_groupCountLabel;
    QLabel* m_savingsLabel;
    
    // Core Engine References
    FileScanner* m_fileScanner;
    DuplicateDetector* m_duplicateDetector;
    HashCalculator* m_hashCalculator;
    SafetyManager* m_safetyManager;
    FileManager* m_fileManager;
    
    // Child Windows
    ScanSetupDialog* m_scanSetupDialog;
    ResultsWindow* m_resultsWindow;
    
    // Utilities
    QTimer* m_systemUpdateTimer;
    
    // Scan results cache
    QList<FileScanner::FileInfo> m_lastScanResults;
    
    void createHeaderWidget();
    void createContentWidgets();
    void createStatusBar();
    void applyTheme();
    void loadSettings();
    void saveSettings();
    QString formatFileSize(qint64 bytes) const;
};

// Custom Widgets for Main Window Content Areas

class QuickActionsWidget : public QGroupBox
{
    Q_OBJECT
    
public:
    explicit QuickActionsWidget(QWidget* parent = nullptr);
    
public slots:
    void setEnabled(bool enabled);
    
signals:
    void presetSelected(const QString& preset);
    
private slots:
    void onQuickScanClicked();
    void onDownloadsCleanupClicked();
    void onPhotoCleanupClicked();
    void onDocumentsClicked();
    void onFullSystemClicked();
    void onCustomPresetClicked();
    
private:
    QGridLayout* m_layout;
    QPushButton* m_quickScanButton;
    QPushButton* m_downloadsButton;
    QPushButton* m_photoButton;
    QPushButton* m_documentsButton;
    QPushButton* m_fullSystemButton;
    QPushButton* m_customButton;
    
    void createButtons();
    void setupButtonStyles();
};

class ScanHistoryWidget : public QGroupBox
{
    Q_OBJECT
    
public:
    struct ScanHistoryItem {
        QString date;
        QString type;
        int duplicateCount;
        qint64 spaceSaved;
        QString scanId;
    };
    
    explicit ScanHistoryWidget(QWidget* parent = nullptr);
    
    void addScanResult(const ScanHistoryItem& item);
    void clearHistory();
    QList<ScanHistoryItem> getHistory() const;
    
public slots:
    void refreshHistory();
    
signals:
    void historyItemClicked(int index);
    void viewAllRequested();
    
private slots:
    void onItemDoubleClicked(int row);
    void onViewAllClicked();
    
private:
    QVBoxLayout* m_layout;
    QListWidget* m_historyList;
    QPushButton* m_viewAllButton;
    QList<ScanHistoryItem> m_historyItems;
    
    void updateHistoryDisplay();
    QString formatHistoryItem(const ScanHistoryItem& item) const;
    QString formatBytes(qint64 bytes) const;
    void addSampleHistory();
};

class SystemOverviewWidget : public QGroupBox
{
    Q_OBJECT
    
public:
    struct SystemStats {
        qint64 totalDiskSpace;
        qint64 availableDiskSpace;
        qint64 potentialSavings;
        int filesScanned;
        double usagePercentage;
    };
    
    explicit SystemOverviewWidget(QWidget* parent = nullptr);
    
    void updateStats(const SystemStats& stats);
    SystemStats getCurrentStats() const;
    
public slots:
    void refreshStats();
    
private:
    QVBoxLayout* m_layout;
    QLabel* m_diskSpaceLabel;
    QProgressBar* m_diskUsageBar;
    QLabel* m_availableSpaceLabel;
    QLabel* m_potentialSavingsLabel;
    QLabel* m_filesScannedLabel;
    
    SystemStats m_currentStats;
    
    void createStatsDisplay();
    void updateDisplay();
    QString formatBytes(qint64 bytes) const;
    QColor getUsageColor(double percentage) const;
};

#endif // MAIN_WINDOW_H