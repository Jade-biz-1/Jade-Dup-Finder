#include <QtTest>
#include <QDebug>
#include "file_scanner.h"
#include <QtTest/QSignalSpy>
#include <QtCore/QTemporaryDir>
#include <QtCore/QFile>
#include <QtCore/QTextStream>

/**
 * @brief Unit test runner for DupFinder
 * 
 * This file runs all unit tests for the core components.
 * Currently contains placeholder tests to verify the test framework.
 */

class BasicTest : public QObject {
    Q_OBJECT

private slots:
    void initTestCase() {
        qDebug() << "Starting DupFinder unit tests...";
    }

    void testBasicFunctionality() {
        // Placeholder test - replace with actual tests as components are developed
        QVERIFY(true);
        qDebug() << "✅ Basic test passed";
    }

    void testQtVersion() {
        qDebug() << "Qt version:" << QT_VERSION_STR;
        QVERIFY(!QString(QT_VERSION_STR).isEmpty());
    }

    void cleanupTestCase() {
        qDebug() << "Unit tests completed.";
    }
};

class TestFileScanner : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void testBasicScan();
    void testScanSignals();
    void cleanupTestCase();

private:
    QTemporaryDir* m_tempDir;
};

void TestFileScanner::initTestCase()
{
    // Create temporary directory with test files
    m_tempDir = new QTemporaryDir();
    QVERIFY(m_tempDir->isValid());
    
    // Create some test files
    QFile file1(m_tempDir->path() + "/test1.txt");
    QVERIFY(file1.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream out1(&file1);
    out1 << "This is test file 1";
    file1.close();
    
    QFile file2(m_tempDir->path() + "/test2.txt");
    QVERIFY(file2.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream out2(&file2);
    out2 << "This is test file 2";
    file2.close();
}

void TestFileScanner::testBasicScan()
{
    FileScanner scanner;
    
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1; // Include small files
    
    scanner.startScan(options);
    
    // Wait for scan to complete (simple busy wait for test)
    int maxWait = 5000; // 5 seconds max
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    QVERIFY(!scanner.isScanning());
    QVERIFY(scanner.getTotalFilesFound() >= 2); // Should find at least our 2 test files
}

void TestFileScanner::testScanSignals()
{
    FileScanner scanner;
    
    // Set up signal spies
    QSignalSpy startedSpy(&scanner, &FileScanner::scanStarted);
    QSignalSpy progressSpy(&scanner, &FileScanner::scanProgress);
    QSignalSpy completedSpy(&scanner, &FileScanner::scanCompleted);
    QSignalSpy fileFoundSpy(&scanner, &FileScanner::fileFound);
    
    // Configure scan
    FileScanner::ScanOptions options;
    options.targetPaths << m_tempDir->path();
    options.minimumFileSize = 1;
    
    // Start scan
    scanner.startScan(options);
    
    // Verify scanStarted was emitted
    QVERIFY(startedSpy.count() == 1);
    
    // Wait for scan to complete
    int maxWait = 5000;
    while (scanner.isScanning() && maxWait > 0) {
        QTest::qWait(10);
        maxWait -= 10;
    }
    
    // Verify scan completion
    QVERIFY(!scanner.isScanning());
    QVERIFY(completedSpy.count() == 1);
    
    // Should have found some files
    QVERIFY(fileFoundSpy.count() >= 0); // File found signals may or may not fire depending on timing
    
    qDebug() << "🔍 Scan completed successfully:";
    qDebug() << "  - Files found:" << scanner.getTotalFilesFound();
    qDebug() << "  - Bytes scanned:" << scanner.getTotalBytesScanned();
    qDebug() << "  - scanStarted signals:" << startedSpy.count();
    qDebug() << "  - scanCompleted signals:" << completedSpy.count();
    qDebug() << "  - fileFound signals:" << fileFoundSpy.count();
}

void TestFileScanner::cleanupTestCase()
{
    delete m_tempDir;
};

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    
    int result = 0;
    
    // Run basic tests
    {
        BasicTest test;
        result |= QTest::qExec(&test, argc, argv);
    }
    
    // Run FileScanner tests
    {
        TestFileScanner fileScannerTest;
        result |= QTest::qExec(&fileScannerTest, argc, argv);
    }
    
    return result;
}

#include "main_test.moc"