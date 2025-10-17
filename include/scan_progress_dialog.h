#pragma once

#include <QDialog>
#include <QProgressBar>
#include <QLabel>
#include <QPushButton>
#include <QElapsedTimer>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>

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
     * @brief Progress information structure
     */
    struct ProgressInfo {
        int filesScanned = 0;           // Files scanned so far
        int totalFiles = 0;             // Estimated total files (0 if unknown)
        qint64 bytesScanned = 0;        // Bytes scanned so far
        qint64 totalBytes = 0;          // Estimated total bytes (0 if unknown)
        QString currentFolder;          // Current folder being scanned
        QString currentFile;            // Current file being processed
        double filesPerSecond = 0.0;    // Current scan rate
        int secondsRemaining = -1;      // ETA in seconds (-1 if unknown)
        bool isPaused = false;          // Whether scan is paused
        int errorsEncountered = 0;      // Number of errors encountered (Task 10)
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
    // UI Components
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

    // State
    bool m_isPaused;
    QElapsedTimer m_scanTimer;
    
    // T12: Enhanced ETA calculation
    QList<double> m_recentRates;           // Recent scan rates for smoothing
    QList<qint64> m_recentTimestamps;      // Timestamps for rate calculation
    static const int MAX_RATE_SAMPLES = 10; // Maximum samples to keep

    // UI Setup
    void setupUI();
    void createProgressSection(QVBoxLayout* mainLayout);
    void createDetailsSection(QVBoxLayout* mainLayout);
    void createButtonSection(QVBoxLayout* mainLayout);

    // Helper methods
    void updateETA(const ProgressInfo& info);
    void updateButtonStates();
    void updateVisualFeedback(const ProgressInfo& info);  // T12: Enhanced visual feedback
};
