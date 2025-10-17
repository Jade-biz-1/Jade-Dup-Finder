#pragma once

#include <QDialog>
#include <QTableWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include "file_scanner.h"

/**
 * @brief Dialog for displaying scan error details (Task 10)
 * 
 * This dialog shows:
 * - List of all errors encountered during scan
 * - Error type, file path, and description
 * - Timestamp of when error occurred
 * - Copy to clipboard functionality
 */
class ScanErrorDialog : public QDialog {
    Q_OBJECT

public:
    explicit ScanErrorDialog(QWidget* parent = nullptr);
    ~ScanErrorDialog() override = default;

    /**
     * @brief Set the errors to display
     * @param errors List of scan errors
     */
    void setErrors(const QList<FileScanner::ScanErrorInfo>& errors);

    /**
     * @brief Clear all errors from the display
     */
    void clearErrors();

private slots:
    void copyToClipboard();
    void copySelectedToClipboard();

private:
    // UI Components
    QTableWidget* m_errorTable;
    QLabel* m_summaryLabel;
    QPushButton* m_copyAllButton;
    QPushButton* m_copySelectedButton;
    QPushButton* m_closeButton;

    // Helper methods
    void setupUI();
    void setupTable();
    QString formatErrorType(FileScanner::ScanError errorType) const;
    QString formatTimestamp(const QDateTime& timestamp) const;
    QString errorsToText(bool selectedOnly = false) const;
};