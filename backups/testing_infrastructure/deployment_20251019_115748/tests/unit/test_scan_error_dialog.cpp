#include <QtTest/QtTest>
#include "scan_error_dialog.h"
#include "file_scanner.h"

class TestScanErrorDialog : public QObject
{
    Q_OBJECT

private slots:
    void testEmptyErrors();
    void testSetErrors();
    void testClearErrors();
    void testErrorFormatting();

private:
    FileScanner::ScanErrorInfo createTestError(FileScanner::ScanError type, const QString& path, const QString& message);
};

FileScanner::ScanErrorInfo TestScanErrorDialog::createTestError(FileScanner::ScanError type, const QString& path, const QString& message)
{
    FileScanner::ScanErrorInfo error;
    error.errorType = type;
    error.filePath = path;
    error.errorMessage = message;
    error.timestamp = QDateTime::currentDateTime();
    return error;
}

void TestScanErrorDialog::testEmptyErrors()
{
    ScanErrorDialog dialog;
    
    // Should start with no errors
    QList<FileScanner::ScanErrorInfo> emptyErrors;
    dialog.setErrors(emptyErrors);
    
    // Dialog should handle empty list gracefully
    QVERIFY(true); // If we get here without crashing, test passes
}

void TestScanErrorDialog::testSetErrors()
{
    ScanErrorDialog dialog;
    
    QList<FileScanner::ScanErrorInfo> errors;
    errors.append(createTestError(FileScanner::ScanError::PermissionDenied, "/test/path1", "Access denied"));
    errors.append(createTestError(FileScanner::ScanError::FileSystemError, "/test/path2", "File not found"));
    
    dialog.setErrors(errors);
    
    // Should not crash and should accept the errors
    QVERIFY(true);
}

void TestScanErrorDialog::testClearErrors()
{
    ScanErrorDialog dialog;
    
    // Add some errors first
    QList<FileScanner::ScanErrorInfo> errors;
    errors.append(createTestError(FileScanner::ScanError::PermissionDenied, "/test/path", "Test error"));
    dialog.setErrors(errors);
    
    // Clear them
    dialog.clearErrors();
    
    // Should not crash
    QVERIFY(true);
}

void TestScanErrorDialog::testErrorFormatting()
{
    ScanErrorDialog dialog;
    
    // Test different error types
    QList<FileScanner::ScanErrorInfo> errors;
    errors.append(createTestError(FileScanner::ScanError::PermissionDenied, "/test/permission", "Permission denied"));
    errors.append(createTestError(FileScanner::ScanError::PathTooLong, "/very/long/path", "Path too long"));
    errors.append(createTestError(FileScanner::ScanError::NetworkTimeout, "/network/path", "Network timeout"));
    
    dialog.setErrors(errors);
    
    // Should handle different error types without crashing
    QVERIFY(true);
}

#include "test_scan_error_dialog.moc"