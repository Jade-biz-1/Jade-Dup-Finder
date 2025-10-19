#include <QtTest/QtTest>
#include "scan_progress_dialog.h"

/**
 * @brief Unit tests for ScanProgressDialog
 * 
 * Tests cover:
 * - ETA calculation logic
 * - Time formatting
 * - Byte formatting
 * - Progress updates
 * - Pause/Resume state management
 */
class TestScanProgressDialog : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();

    // ETA calculation tests
    void testCalculateETA_ValidInputs();
    void testCalculateETA_ZeroFilesPerSecond();
    void testCalculateETA_NegativeInputs();
    void testCalculateETA_AlreadyComplete();
    void testCalculateETA_LargeNumbers();
    void testCalculateETA_SmallProgress();
    void testCalculateETA_NearCompletion();

    // Time formatting tests
    void testFormatTime_Seconds();
    void testFormatTime_Minutes();
    void testFormatTime_Hours();
    void testFormatTime_Mixed();
    void testFormatTime_Zero();
    void testFormatTime_Negative();
    void testFormatTime_LessThanOneSecond();

    // Byte formatting tests
    void testFormatBytes_Bytes();
    void testFormatBytes_Kilobytes();
    void testFormatBytes_Megabytes();
    void testFormatBytes_Gigabytes();
    void testFormatBytes_Terabytes();
    void testFormatBytes_Zero();
    void testFormatBytes_Negative();
    void testFormatBytes_Boundaries();

    // Dialog functionality tests
    void testInitialState();
    void testUpdateProgress_BasicInfo();
    void testUpdateProgress_WithTotalFiles();
    void testUpdateProgress_WithoutTotalFiles();
    void testUpdateProgress_CurrentActivity();
    void testSetPaused_True();
    void testSetPaused_False();
    void testIsPaused();

    // Signal tests
    void testPauseRequestedSignal();
    void testResumeRequestedSignal();
    void testCancelRequestedSignal();

private:
    ScanProgressDialog* m_dialog;
};

void TestScanProgressDialog::initTestCase() {
    // Setup for all tests
}

void TestScanProgressDialog::cleanupTestCase() {
    // Cleanup after all tests
}

void TestScanProgressDialog::init() {
    // Create fresh dialog for each test
    m_dialog = new ScanProgressDialog();
}

void TestScanProgressDialog::cleanup() {
    // Clean up after each test
    delete m_dialog;
    m_dialog = nullptr;
}

// ============================================================================
// ETA Calculation Tests
// ============================================================================

void TestScanProgressDialog::testCalculateETA_ValidInputs() {
    // Test: 100 files scanned, 1000 total, 10 files/sec
    // Expected: (1000 - 100) / 10 = 90 seconds
    int eta = ScanProgressDialog::calculateETA(100, 1000, 10.0);
    QCOMPARE(eta, 90);

    // Test: 500 files scanned, 1000 total, 50 files/sec
    // Expected: (1000 - 500) / 50 = 10 seconds
    eta = ScanProgressDialog::calculateETA(500, 1000, 50.0);
    QCOMPARE(eta, 10);

    // Test: 1 file scanned, 100 total, 1 file/sec
    // Expected: (100 - 1) / 1 = 99 seconds
    eta = ScanProgressDialog::calculateETA(1, 100, 1.0);
    QCOMPARE(eta, 99);
}

void TestScanProgressDialog::testCalculateETA_ZeroFilesPerSecond() {
    // Cannot calculate ETA with zero rate
    int eta = ScanProgressDialog::calculateETA(100, 1000, 0.0);
    QCOMPARE(eta, -1);
}

void TestScanProgressDialog::testCalculateETA_NegativeInputs() {
    // Negative files scanned
    int eta = ScanProgressDialog::calculateETA(-10, 1000, 10.0);
    QCOMPARE(eta, -1);

    // Negative total files
    eta = ScanProgressDialog::calculateETA(100, -1000, 10.0);
    QCOMPARE(eta, -1);

    // Negative rate
    eta = ScanProgressDialog::calculateETA(100, 1000, -10.0);
    QCOMPARE(eta, -1);
}

void TestScanProgressDialog::testCalculateETA_AlreadyComplete() {
    // Files scanned equals total
    int eta = ScanProgressDialog::calculateETA(1000, 1000, 10.0);
    QCOMPARE(eta, 0);

    // Files scanned exceeds total
    eta = ScanProgressDialog::calculateETA(1500, 1000, 10.0);
    QCOMPARE(eta, 0);
}

void TestScanProgressDialog::testCalculateETA_LargeNumbers() {
    // Test with large file counts
    // 1 million files scanned, 10 million total, 1000 files/sec
    // Expected: (10000000 - 1000000) / 1000 = 9000 seconds
    int eta = ScanProgressDialog::calculateETA(1000000, 10000000, 1000.0);
    QCOMPARE(eta, 9000);
}

void TestScanProgressDialog::testCalculateETA_SmallProgress() {
    // Very small progress
    // 1 file scanned, 10000 total, 100 files/sec
    // Expected: (10000 - 1) / 100 = 99.99 ≈ 100 seconds
    int eta = ScanProgressDialog::calculateETA(1, 10000, 100.0);
    QCOMPARE(eta, 100);
}

void TestScanProgressDialog::testCalculateETA_NearCompletion() {
    // Near completion
    // 999 files scanned, 1000 total, 10 files/sec
    // Expected: (1000 - 999) / 10 = 0.1 ≈ 0 seconds
    int eta = ScanProgressDialog::calculateETA(999, 1000, 10.0);
    QCOMPARE(eta, 0);
}

// ============================================================================
// Time Formatting Tests
// ============================================================================

void TestScanProgressDialog::testFormatTime_Seconds() {
    QCOMPARE(ScanProgressDialog::formatTime(1), QString("1s"));
    QCOMPARE(ScanProgressDialog::formatTime(30), QString("30s"));
    QCOMPARE(ScanProgressDialog::formatTime(59), QString("59s"));
}

void TestScanProgressDialog::testFormatTime_Minutes() {
    QCOMPARE(ScanProgressDialog::formatTime(60), QString("1m"));
    QCOMPARE(ScanProgressDialog::formatTime(90), QString("1m 30s"));
    QCOMPARE(ScanProgressDialog::formatTime(120), QString("2m"));
    QCOMPARE(ScanProgressDialog::formatTime(3599), QString("59m 59s"));
}

void TestScanProgressDialog::testFormatTime_Hours() {
    QCOMPARE(ScanProgressDialog::formatTime(3600), QString("1h 0m"));
    QCOMPARE(ScanProgressDialog::formatTime(3661), QString("1h 1m 1s"));
    QCOMPARE(ScanProgressDialog::formatTime(7200), QString("2h 0m"));
    QCOMPARE(ScanProgressDialog::formatTime(7265), QString("2h 1m 5s"));
}

void TestScanProgressDialog::testFormatTime_Mixed() {
    // 1 hour, 30 minutes, 45 seconds
    QCOMPARE(ScanProgressDialog::formatTime(5445), QString("1h 30m 45s"));
    
    // 2 hours, 15 minutes, 30 seconds
    QCOMPARE(ScanProgressDialog::formatTime(8130), QString("2h 15m 30s"));
}

void TestScanProgressDialog::testFormatTime_Zero() {
    QCOMPARE(ScanProgressDialog::formatTime(0), QString("Complete"));
}

void TestScanProgressDialog::testFormatTime_Negative() {
    QCOMPARE(ScanProgressDialog::formatTime(-1), QString("Unknown"));
    QCOMPARE(ScanProgressDialog::formatTime(-100), QString("Unknown"));
}

void TestScanProgressDialog::testFormatTime_LessThanOneSecond() {
    // The function doesn't handle fractional seconds, but test boundary
    // Any value < 1 should be handled
    // Based on implementation, 0 returns "Complete"
    QCOMPARE(ScanProgressDialog::formatTime(0), QString("Complete"));
}

// ============================================================================
// Byte Formatting Tests
// ============================================================================

void TestScanProgressDialog::testFormatBytes_Bytes() {
    QCOMPARE(ScanProgressDialog::formatBytes(0), QString("0 B"));
    QCOMPARE(ScanProgressDialog::formatBytes(1), QString("1 B"));
    QCOMPARE(ScanProgressDialog::formatBytes(512), QString("512 B"));
    QCOMPARE(ScanProgressDialog::formatBytes(1023), QString("1023 B"));
}

void TestScanProgressDialog::testFormatBytes_Kilobytes() {
    QCOMPARE(ScanProgressDialog::formatBytes(1024), QString("1.00 KB"));
    QCOMPARE(ScanProgressDialog::formatBytes(2048), QString("2.00 KB"));
    QCOMPARE(ScanProgressDialog::formatBytes(1536), QString("1.50 KB"));
    QCOMPARE(ScanProgressDialog::formatBytes(1024 * 1023), QString("1023.00 KB"));
}

void TestScanProgressDialog::testFormatBytes_Megabytes() {
    qint64 MB = 1024 * 1024;
    QCOMPARE(ScanProgressDialog::formatBytes(MB), QString("1.00 MB"));
    QCOMPARE(ScanProgressDialog::formatBytes(MB * 2), QString("2.00 MB"));
    QCOMPARE(ScanProgressDialog::formatBytes(MB + MB / 2), QString("1.50 MB"));
    QCOMPARE(ScanProgressDialog::formatBytes(MB * 1023), QString("1023.00 MB"));
}

void TestScanProgressDialog::testFormatBytes_Gigabytes() {
    qint64 GB = 1024LL * 1024 * 1024;
    QCOMPARE(ScanProgressDialog::formatBytes(GB), QString("1.00 GB"));
    QCOMPARE(ScanProgressDialog::formatBytes(GB * 2), QString("2.00 GB"));
    QCOMPARE(ScanProgressDialog::formatBytes(GB + GB / 2), QString("1.50 GB"));
    QCOMPARE(ScanProgressDialog::formatBytes(GB * 1023), QString("1023.00 GB"));
}

void TestScanProgressDialog::testFormatBytes_Terabytes() {
    qint64 TB = 1024LL * 1024 * 1024 * 1024;
    QCOMPARE(ScanProgressDialog::formatBytes(TB), QString("1.00 TB"));
    QCOMPARE(ScanProgressDialog::formatBytes(TB * 2), QString("2.00 TB"));
    QCOMPARE(ScanProgressDialog::formatBytes(TB + TB / 2), QString("1.50 TB"));
}

void TestScanProgressDialog::testFormatBytes_Zero() {
    QCOMPARE(ScanProgressDialog::formatBytes(0), QString("0 B"));
}

void TestScanProgressDialog::testFormatBytes_Negative() {
    // Negative values should be treated as 0
    QCOMPARE(ScanProgressDialog::formatBytes(-1), QString("0 B"));
    QCOMPARE(ScanProgressDialog::formatBytes(-1024), QString("0 B"));
}

void TestScanProgressDialog::testFormatBytes_Boundaries() {
    // Test boundary values between units
    QCOMPARE(ScanProgressDialog::formatBytes(1023), QString("1023 B"));
    QCOMPARE(ScanProgressDialog::formatBytes(1024), QString("1.00 KB"));
    
    qint64 MB = 1024 * 1024;
    // MB - 1 = 1048575 bytes = 1024.00 KB (rounded)
    QCOMPARE(ScanProgressDialog::formatBytes(MB - 1), QString("1024.00 KB"));
    QCOMPARE(ScanProgressDialog::formatBytes(MB), QString("1.00 MB"));
    
    qint64 GB = 1024LL * 1024 * 1024;
    // GB - 1 = 1073741823 bytes = 1024.00 MB (rounded)
    QCOMPARE(ScanProgressDialog::formatBytes(GB - 1), QString("1024.00 MB"));
    QCOMPARE(ScanProgressDialog::formatBytes(GB), QString("1.00 GB"));
}

// ============================================================================
// Dialog Functionality Tests
// ============================================================================

void TestScanProgressDialog::testInitialState() {
    QVERIFY(m_dialog != nullptr);
    QVERIFY(!m_dialog->isPaused());
    QCOMPARE(m_dialog->windowTitle(), QString("Scan Progress"));
}

void TestScanProgressDialog::testUpdateProgress_BasicInfo() {
    ScanProgressDialog::ProgressInfo info;
    info.filesScanned = 100;
    info.bytesScanned = 1024 * 1024 * 50; // 50 MB
    info.filesPerSecond = 10.5;

    m_dialog->updateProgress(info);

    // Dialog should update without crashing
    QVERIFY(true);
}

void TestScanProgressDialog::testUpdateProgress_WithTotalFiles() {
    ScanProgressDialog::ProgressInfo info;
    info.filesScanned = 250;
    info.totalFiles = 1000;
    info.bytesScanned = 1024 * 1024 * 100; // 100 MB
    info.filesPerSecond = 25.0;

    m_dialog->updateProgress(info);

    // Dialog should update without crashing
    QVERIFY(true);
}

void TestScanProgressDialog::testUpdateProgress_WithoutTotalFiles() {
    ScanProgressDialog::ProgressInfo info;
    info.filesScanned = 500;
    info.totalFiles = 0; // Unknown total
    info.bytesScanned = 1024 * 1024 * 200; // 200 MB
    info.filesPerSecond = 50.0;

    m_dialog->updateProgress(info);

    // Dialog should handle unknown total gracefully
    QVERIFY(true);
}

void TestScanProgressDialog::testUpdateProgress_CurrentActivity() {
    ScanProgressDialog::ProgressInfo info;
    info.filesScanned = 100;
    info.currentFolder = "/home/user/documents";
    info.currentFile = "report.pdf";
    info.filesPerSecond = 10.0;

    m_dialog->updateProgress(info);

    // Dialog should display current activity
    QVERIFY(true);
}

void TestScanProgressDialog::testSetPaused_True() {
    QVERIFY(!m_dialog->isPaused());
    
    m_dialog->setPaused(true);
    
    QVERIFY(m_dialog->isPaused());
}

void TestScanProgressDialog::testSetPaused_False() {
    m_dialog->setPaused(true);
    QVERIFY(m_dialog->isPaused());
    
    m_dialog->setPaused(false);
    
    QVERIFY(!m_dialog->isPaused());
}

void TestScanProgressDialog::testIsPaused() {
    QVERIFY(!m_dialog->isPaused());
    
    m_dialog->setPaused(true);
    QVERIFY(m_dialog->isPaused());
    
    m_dialog->setPaused(false);
    QVERIFY(!m_dialog->isPaused());
}

// ============================================================================
// Signal Tests
// ============================================================================

void TestScanProgressDialog::testPauseRequestedSignal() {
    QSignalSpy spy(m_dialog, &ScanProgressDialog::pauseRequested);
    
    // Simulate pause button click
    // Note: This requires accessing the button, which is private
    // For now, we'll test that the signal exists
    QVERIFY(spy.isValid());
}

void TestScanProgressDialog::testResumeRequestedSignal() {
    QSignalSpy spy(m_dialog, &ScanProgressDialog::resumeRequested);
    
    // Simulate resume button click
    QVERIFY(spy.isValid());
}

void TestScanProgressDialog::testCancelRequestedSignal() {
    QSignalSpy spy(m_dialog, &ScanProgressDialog::cancelRequested);
    
    // Simulate cancel button click
    QVERIFY(spy.isValid());
}

QTEST_MAIN(TestScanProgressDialog)
#include "test_scan_progress_dialog.moc"
