#pragma once

#include <QDialog>
#include <QProgressBar>
#include <QLabel>
#include <QPushButton>
#include <QElapsedTimer>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QDateTime>
#include <QListWidget>

/**
 * @brief Dialog for displaying detailed scan progress information
 * 
 * This dialog shows:
 * - Overall progress bar
 * - Files scanned count
 * - Data scanned (MB/GB)
 * - Current folder and file being scanned
 * - Scan rate (files/second)
 * - Estimated time remaining (ETA)
 * - Pause/Resume and Cancel buttons
 * 
 * Requirements: 2.1, 2.2, 2.3, 2.4, 2.7
 */
class ScanProgressDialog : public QDialog {
    Q_OBJECT

public:
    /**
     * @brief Operation status enumeration for enhanced visual indication
     */
    enum class OperationStatus {
        Initializing,
        Running,
        Paused,
        Completed,
        Error,
        Cancelled
    };

    /**
     * @brief Queue operation information for multiple operations support
     */
    struct QueuedOperation {
        QString operationId;
        QString operationType;          // "scan", "hash", "delete", etc.
        QString description;
        OperationStatus status = OperationStatus::Initializing;
        int priority = 0;              // Higher number = higher priority
        QDateTime queuedTime;
        QDateTime startTime;
        QDateTime completedTime;
        int filesTotal = 0;
        int filesProcessed = 0;
        qint64 bytesTotal = 0;
        qint64 bytesProcessed = 0;
        int errorsEncountered = 0;
    };

    /**
     * @brief Enhanced progress information structure with queue support
     */
    struct ProgressInfo {
        // Current operation details
        QString operationId;            // Unique identifier for current operation
        QString operationType;          // Type of operation (scan, hash, delete, etc.)
        OperationStatus status = OperationStatus::Initializing;
        
        // File and data progress
        int filesScanned = 0;           // Files scanned so far
        int totalFiles = 0;             // Estimated total files (0 if unknown)
        qint64 bytesScanned = 0;        // Bytes scanned so far
        qint64 totalBytes = 0;          // Estimated total bytes (0 if unknown)
        
        // Current activity
        QString currentFolder;          // Current folder being scanned
        QString currentFile;            // Current file being processed
        
        // Performance metrics
        double filesPerSecond = 0.0;    // Current scan rate
        double bytesPerSecond = 0.0;    // Current data processing rate
        double averageFileSize = 0.0;   // Average file size processed
        
        // Timing information
        int secondsRemaining = -1;      // ETA in seconds (-1 if unknown)
        int secondsElapsed = 0;         // Time elapsed since start
        bool isPaused = false;          // Whether scan is paused
        
        // Error handling
        int errorsEncountered = 0;      // Number of errors encountered
        QString lastError;              // Description of last error
        
        // Queue information for multiple operations
        QList<QueuedOperation> operationQueue;
        int currentOperationIndex = 0;  // Index of current operation in queue
        int totalOperationsInQueue = 0; // Total operations in queue
    };

    explicit ScanProgressDialog(QWidget* parent = nullptr);
    ~ScanProgressDialog() override = default;

    /**
     * @brief Update the progress display
     * @param info Progress information to display
     */
    void updateProgress(const ProgressInfo& info);

    /**
     * @brief Set the paused state
     * @param paused True if scan is paused
     */
    void setPaused(bool paused);

    /**
     * @brief Check if scan is paused
     * @return True if paused
     */
    bool isPaused() const;

    /**
     * @brief Update operation queue display
     * @param operations List of queued operations
     */
    void updateOperationQueue(const QList<QueuedOperation>& operations);

    /**
     * @brief Set the current operation status with visual indication
     * @param status Current operation status
     */
    void setOperationStatus(OperationStatus status);

    /**
     * @brief Set the operation manager for enhanced queue support
     * @param manager The operation manager instance
     */
    void setOperationManager(class OperationManager* manager);

    /**
     * @brief Calculate ETA based on progress
     * @param filesScanned Files scanned so far
     * @param totalFiles Total files to scan
     * @param filesPerSecond Current scan rate
     * @return Estimated seconds remaining (-1 if cannot calculate)
     */
    static int calculateETA(int filesScanned, int totalFiles, double filesPerSecond);

    /**
     * @brief Format time duration in human-readable format
     * @param seconds Time in seconds
     * @return Formatted string (e.g., "2h 15m 30s", "45s", "< 1s")
     */
    static QString formatTime(int seconds);

    /**
     * @brief Format byte size in human-readable format
     * @param bytes Size in bytes
     * @return Formatted string (e.g., "1.5 GB", "234 MB", "5.2 KB")
     */
    static QString formatBytes(qint64 bytes);

signals:
    /**
     * @brief Emitted when user clicks pause button
     */
    void pauseRequested();

    /**
     * @brief Emitted when user clicks resume button
     */
    void resumeRequested();

    /**
     * @brief Emitted when user clicks cancel button
     */
    void cancelRequested();

    /**
     * @brief Emitted when user clicks view errors button (Task 10)
     */
    void viewErrorsRequested();

private:
    // UI Components - Main Progress
    QProgressBar* m_overallProgress;
    QProgressBar* m_throughputProgress;     // T12: Throughput indicator
    QLabel* m_filesLabel;
    QLabel* m_sizeLabel;
    QLabel* m_currentFolderLabel;
    QLabel* m_currentFileLabel;
    QLabel* m_rateLabel;
    QLabel* m_throughputLabel;              // T12: Data throughput
    QLabel* m_etaLabel;
    QLabel* m_elapsedLabel;                 // T12: Elapsed time
    QLabel* m_statusLabel;
    QPushButton* m_pauseButton;
    QPushButton* m_cancelButton;
    QLabel* m_errorsLabel;              // Task 10: Error count display
    QPushButton* m_viewErrorsButton;    // Task 10: View errors button

    // Enhanced Progress Components - Task 7.1 & 7.2
    QLabel* m_operationTypeLabel;           // Current operation type display
    QLabel* m_operationStatusLabel;         // Visual status indicator with icon
    QLabel* m_averageFileSizeLabel;         // Average file size metric
    QLabel* m_bytesPerSecondLabel;          // Enhanced data rate display
    QProgressBar* m_queueProgress;          // Overall queue progress
    QLabel* m_queueStatusLabel;             // Queue status (X of Y operations)
    QListWidget* m_operationQueueList;     // List of queued operations
    QGroupBox* m_queueGroup;                // Queue section group box
    QLabel* m_lastErrorLabel;               // Last error message display

    // State
    bool m_isPaused;
    QElapsedTimer m_scanTimer;
    class OperationManager* m_operationManager;
    
    // T12: Enhanced ETA calculation
    QList<double> m_recentRates;           // Recent scan rates for smoothing
    QList<qint64> m_recentTimestamps;      // Timestamps for rate calculation
    static const int MAX_RATE_SAMPLES = 10; // Maximum samples to keep

    // UI Setup
    void setupUI();
    void createProgressSection(QVBoxLayout* mainLayout);
    void createDetailsSection(QVBoxLayout* mainLayout);
    void createButtonSection(QVBoxLayout* mainLayout);
    void createQueueSection(QVBoxLayout* mainLayout);      // Task 7.2: Queue display
    void createEnhancedMetricsSection(QVBoxLayout* mainLayout); // Task 7.1: Enhanced metrics

    // Helper methods
    void updateETA(const ProgressInfo& info);
    void updateButtonStates();
    void updateVisualFeedback(const ProgressInfo& info);  // T12: Enhanced visual feedback
    void updateOperationStatusDisplay(OperationStatus status); // Task 7.1: Status indication
    void updateQueueDisplay(const QList<QueuedOperation>& operations); // Task 7.2: Queue updates
    void updateEnhancedMetrics(const ProgressInfo& info);  // Task 7.1: Enhanced metrics
    void updateOperationQueueProgress(const ProgressInfo& info); // Task 7.2: Queue progress
    QString getStatusIcon(OperationStatus status) const;   // Task 7.1: Status icons
    QString getStatusText(OperationStatus status) const;   // Task 7.1: Status text
    QColor getStatusColor(OperationStatus status) const;   // Task 7.1: Status colors
};
