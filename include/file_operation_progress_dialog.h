#pragma once

#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QTimer>
#include "file_operation_queue.h"

/**
 * @brief Progress dialog for file operations (Task 24)
 * 
 * This dialog shows detailed progress for file operations including:
 * - Current operation type and status
 * - Files processed vs total files
 * - Bytes processed vs total bytes
 * - Current file being processed
 * - Time elapsed and estimated time remaining
 * - Operation speed (files/sec, bytes/sec)
 * - Cancel button
 */
class FileOperationProgressDialog : public QDialog {
    Q_OBJECT

public:
    explicit FileOperationProgressDialog(QWidget* parent = nullptr);
    ~FileOperationProgressDialog() override = default;

    /**
     * @brief Set the file operation queue to monitor
     */
    void setOperationQueue(FileOperationQueue* queue);

    /**
     * @brief Show dialog for a specific operation
     */
    void showForOperation(const QString& operationId);

signals:
    /**
     * @brief Emitted when user requests to cancel the operation
     */
    void cancelRequested(const QString& operationId);

private slots:
    void updateProgress();
    void onCancelClicked();
    void onOperationStarted(const QString& operationId);
    void onOperationCompleted(const QString& operationId, bool success, const QString& errorMessage);
    void onOperationCancelled(const QString& operationId);

private:
    // UI Components
    QLabel* m_titleLabel;
    QLabel* m_operationTypeLabel;
    QLabel* m_statusLabel;
    
    QProgressBar* m_fileProgressBar;
    QLabel* m_fileProgressLabel;
    
    QProgressBar* m_byteProgressBar;
    QLabel* m_byteProgressLabel;
    
    QLabel* m_currentFileLabel;
    QLabel* m_timeElapsedLabel;
    QLabel* m_timeRemainingLabel;
    QLabel* m_speedLabel;
    
    QPushButton* m_cancelButton;
    QPushButton* m_closeButton;
    
    // Data
    FileOperationQueue* m_operationQueue;
    QString m_currentOperationId;
    QTimer* m_updateTimer;
    
    // Helper methods
    void setupUI();
    void connectSignals();
    void resetDisplay();
    void updateOperationInfo(const FileOperationQueue::OperationProgress& progress);
    QString formatOperationType(FileOperationQueue::OperationType type) const;
    QString formatOperationStatus(FileOperationQueue::OperationStatus status) const;
    QString formatFileSize(qint64 bytes) const;
    QString formatTime(qint64 milliseconds) const;
    QString formatSpeed(double bytesPerSecond) const;
};