#include <QtTest/QtTest>
#include <QtCore/QTemporaryDir>
#include <QtCore/QDir>
#include <QtCore/QFile>
#include "scan_dialog.h"

class TestScanConfigurationValidation : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();
    
    // Basic validation tests
    void testEmptyTargetPaths();
    void testValidConfiguration();
    void testInvalidMinimumFileSize();
    void testInvalidMaximumDepth();
    
    // Path validation tests
    void testNonExistentPath();
    void testMultipleNonExistentPaths();
    void testAllPathsNonExistent();
    void testInaccessiblePath();
    void testMixedValidInvalidPaths();
    
    // Exclude pattern validation tests
    void testValidExcludePatterns();
    void testInvalidRegexPattern();
    void testEmptyExcludePatterns();
    
    // Exclude folder validation tests
    void testValidExcludeFolders();
    void testExcludeFolderContainsTargetPath();
    void testTargetPathInsideExcludeFolder();
    void testEmptyExcludeFolders();
    
    // Integration tests
    void testValidationErrorMessages();
    void testIsValidMethod();

private:
    QTemporaryDir* m_tempDir;
    QString m_validPath1;
    QString m_validPath2;
    QString m_invalidPath;
};

void TestScanConfigurationValidation::initTestCase()
{
    // Create temporary directory for testing
    m_tempDir = new QTemporaryDir();
    QVERIFY(m_tempDir->isValid());
    
    // Create test directories
    m_validPath1 = m_tempDir->path() + "/valid1";
    m_validPath2 = m_tempDir->path() + "/valid2";
    m_invalidPath = m_tempDir->path() + "/nonexistent";
    
    QDir().mkpath(m_validPath1);
    QDir().mkpath(m_validPath2);
    
    // Create some test files
    QFile file1(m_validPath1 + "/test1.txt");
    file1.open(QIODevice::WriteOnly);
    file1.write("test content");
    file1.close();
    
    QFile file2(m_validPath2 + "/test2.txt");
    file2.open(QIODevice::WriteOnly);
    file2.write("test content");
    file2.close();
}

void TestScanConfigurationValidation::cleanupTestCase()
{
    delete m_tempDir;
}

void TestScanConfigurationValidation::init()
{
    // Setup before each test
}

void TestScanConfigurationValidation::cleanup()
{
    // Cleanup after each test
}

// Basic validation tests

void TestScanConfigurationValidation::testEmptyTargetPaths()
{
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths.clear();
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    
    QVERIFY(!config.isValid());
    QString error = config.validationError();
    QVERIFY(!error.isEmpty());
    QVERIFY(error.contains("No scan locations"));
}

void TestScanConfigurationValidation::testValidConfiguration()
{
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << m_validPath1;
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    config.includeHidden = false;
    config.includeSystem = false;
    config.followSymlinks = true;
    config.scanArchives = false;
    
    QVERIFY(config.isValid());
    QString error = config.validationError();
    QVERIFY(error.isEmpty());
}

void TestScanConfigurationValidation::testInvalidMinimumFileSize()
{
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << m_validPath1;
    config.minimumFileSize = -100;
    config.maximumDepth = -1;
    
    QVERIFY(!config.isValid());
    QString error = config.validationError();
    QVERIFY(!error.isEmpty());
    QVERIFY(error.contains("Invalid minimum file size"));
}

void TestScanConfigurationValidation::testInvalidMaximumDepth()
{
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << m_validPath1;
    config.minimumFileSize = 0;
    config.maximumDepth = -5;
    
    QVERIFY(!config.isValid());
    QString error = config.validationError();
    QVERIFY(!error.isEmpty());
    QVERIFY(error.contains("Invalid maximum depth"));
}

// Path validation tests

void TestScanConfigurationValidation::testNonExistentPath()
{
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << m_invalidPath;
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    
    QVERIFY(!config.isValid());
    QString error = config.validationError();
    QVERIFY(!error.isEmpty());
    QVERIFY(error.contains("None of the selected paths exist") || 
            error.contains("Path does not exist"));
}

void TestScanConfigurationValidation::testMultipleNonExistentPaths()
{
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << m_invalidPath << (m_tempDir->path() + "/another_invalid");
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    
    QVERIFY(!config.isValid());
    QString error = config.validationError();
    QVERIFY(!error.isEmpty());
    QVERIFY(error.contains("paths do not exist") || 
            error.contains("None of the selected paths exist"));
}

void TestScanConfigurationValidation::testAllPathsNonExistent()
{
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << m_invalidPath << (m_tempDir->path() + "/invalid2");
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    
    QVERIFY(!config.isValid());
    QString error = config.validationError();
    QVERIFY(!error.isEmpty());
    QVERIFY(error.contains("None of the selected paths exist"));
}

void TestScanConfigurationValidation::testInaccessiblePath()
{
    // This test is platform-specific and may not work on all systems
    // Skip if we can't create an inaccessible directory
    QString inaccessiblePath = m_tempDir->path() + "/inaccessible";
    QDir().mkpath(inaccessiblePath);
    
#ifndef Q_OS_WIN
    // Try to make directory unreadable (Unix-like systems)
    QFile::setPermissions(inaccessiblePath, QFile::WriteOwner);
    
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << inaccessiblePath;
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    
    // Check if we actually made it inaccessible
    if (!QFileInfo(inaccessiblePath).isReadable()) {
        QVERIFY(!config.isValid());
        QString error = config.validationError();
        QVERIFY(!error.isEmpty());
        QVERIFY(error.contains("not readable") || error.contains("permission"));
    }
    
    // Restore permissions for cleanup
    QFile::setPermissions(inaccessiblePath, QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner);
#else
    QSKIP("Inaccessible path test not supported on Windows");
#endif
}

void TestScanConfigurationValidation::testMixedValidInvalidPaths()
{
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << m_validPath1 << m_invalidPath;
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    
    QVERIFY(!config.isValid());
    QString error = config.validationError();
    QVERIFY(!error.isEmpty());
    QVERIFY(error.contains("do not exist") || error.contains("does not exist"));
}

// Exclude pattern validation tests

void TestScanConfigurationValidation::testValidExcludePatterns()
{
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << m_validPath1;
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    config.excludePatterns << "*.tmp" << "*.log" << "Thumbs.db";
    
    QVERIFY(config.isValid());
    QString error = config.validationError();
    QVERIFY(error.isEmpty());
}

void TestScanConfigurationValidation::testInvalidRegexPattern()
{
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << m_validPath1;
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    config.excludePatterns << "[invalid[regex";
    
    QVERIFY(!config.isValid());
    QString error = config.validationError();
    QVERIFY(!error.isEmpty());
    QVERIFY(error.contains("Invalid exclude pattern"));
}

void TestScanConfigurationValidation::testEmptyExcludePatterns()
{
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << m_validPath1;
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    config.excludePatterns << "" << "   " << "*.tmp";
    
    QVERIFY(config.isValid());
    QString error = config.validationError();
    QVERIFY(error.isEmpty());
}

// Exclude folder validation tests

void TestScanConfigurationValidation::testValidExcludeFolders()
{
    QString subFolder = m_validPath1 + "/subfolder";
    QDir().mkpath(subFolder);
    
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << m_validPath1;
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    config.excludeFolders << subFolder;
    
    QVERIFY(config.isValid());
    QString error = config.validationError();
    QVERIFY(error.isEmpty());
}

void TestScanConfigurationValidation::testExcludeFolderContainsTargetPath()
{
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << m_validPath1;
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    config.excludeFolders << m_tempDir->path(); // Parent of target path
    
    QVERIFY(!config.isValid());
    QString error = config.validationError();
    QVERIFY(!error.isEmpty());
    QVERIFY(error.contains("contains target path") || 
            error.contains("exclude the entire scan location"));
}

void TestScanConfigurationValidation::testTargetPathInsideExcludeFolder()
{
    QString subFolder = m_validPath1 + "/subfolder";
    QDir().mkpath(subFolder);
    
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << m_validPath1;
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    config.excludeFolders << subFolder; // Subfolder of target - this is OK
    
    QVERIFY(config.isValid());
    QString error = config.validationError();
    QVERIFY(error.isEmpty());
}

void TestScanConfigurationValidation::testEmptyExcludeFolders()
{
    ScanSetupDialog::ScanConfiguration config;
    config.targetPaths << m_validPath1;
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    config.excludeFolders << "" << "   ";
    
    QVERIFY(config.isValid());
    QString error = config.validationError();
    QVERIFY(error.isEmpty());
}

// Integration tests

void TestScanConfigurationValidation::testValidationErrorMessages()
{
    // Test that error messages are descriptive
    ScanSetupDialog::ScanConfiguration config;
    
    // Empty paths
    config.targetPaths.clear();
    QString error = config.validationError();
    QVERIFY(error.length() > 20); // Should be descriptive
    QVERIFY(error.contains("scan") || error.contains("location"));
    
    // Invalid size
    config.targetPaths << m_validPath1;
    config.minimumFileSize = -1;
    error = config.validationError();
    QVERIFY(error.length() > 20);
    QVERIFY(error.contains("size") || error.contains("minimum"));
}

void TestScanConfigurationValidation::testIsValidMethod()
{
    ScanSetupDialog::ScanConfiguration config;
    
    // Invalid config
    config.targetPaths.clear();
    QVERIFY(!config.isValid());
    QVERIFY(!config.validationError().isEmpty());
    
    // Valid config
    config.targetPaths << m_validPath1;
    config.minimumFileSize = 0;
    config.maximumDepth = -1;
    QVERIFY(config.isValid());
    QVERIFY(config.validationError().isEmpty());
    
    // isValid() should match validationError().isEmpty()
    config.minimumFileSize = -1;
    QCOMPARE(config.isValid(), config.validationError().isEmpty());
}

QTEST_MAIN(TestScanConfigurationValidation)
#include "test_scan_configuration_validation.moc"
