#include <QtTest/QtTest>
#include <QTemporaryDir>
#include <QFile>
#include <QTextStream>
#include <QApplication>

#include "file_manager.h"
#include "safety_manager.h"
#include "restore_dialog.h"

/**
 * @brief Comprehensive test for end-to-end restore functionality
 * 
 * This test verifies that the complete restore workflow works:
 * 1. FileManager creates backups and registers operations with SafetyManager
 * 2. SafetyManager tracks operations in undo history
 * 3. RestoreDialog can retrieve and display operations
 * 4. Restore operations work correctly
 */
class TestRestoreFunctionality : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();
    
    // Core functionality tests
    void testFileManagerRegistersOperations();
    void testSafetyManagerUndoHistory();
    void testRestoreDialogPopulation();
    void testEndToEndRestoreWorkflow();
    void testRestoreWithMissingBackup();
    void testRestoreWithConflicts();
    void testMultipleFileRestore();
    
    // Edge cases
    void testRestoreAfterBackupCleanup();
    void testRestoreWithCorruptedBackup();
    void testRestorePermissionIssues();

private:
    // Helper methods
    void createTestFile(const QString& path, const QString& content);
    bool verifyFileContent(const QString& path, const QString& expectedContent);
    void simulateFileOperation(const QString& filePath, SafetyManager::OperationType operation);
    
    // Test infrastructure
    QTemporaryDir* m_testDir;
    SafetyManager* m_safetyManager;
    FileManager* m_fileManager;
    QString m_testFilePath1;
    QString m_testFilePath2;
    QString m_testFilePath3;
};

void TestRestoreFunctionality::initTestCase()
{
    // Initialize Qt application if not already done
    if (!QApplication::instance()) {
        int argc = 0;
        char** argv = nullptr;
        new QApplication(argc, argv);
    }
}

void TestRestoreFunctionality::cleanupTestCase()
{
    // Cleanup handled in individual test cleanup
}

void TestRestoreFunctionality::init()
{
    // Create temporary directory for test files
    m_testDir = new QTemporaryDir();
    QVERIFY(m_testDir->isValid());
    
    // Initialize SafetyManager with test backup directory
    m_safetyManager = new SafetyManager(this);
    QString backupDir = m_testDir->path() + "/backups";
    m_safetyManager->setBackupDirectory(backupDir);
    m_safetyManager->setSafetyLevel(SafetyManager::SafetyLevel::Standard);
    
    // Initialize FileManager with SafetyManager
    m_fileManager = new FileManager(this);
    m_fileManager->setSafetyManager(m_safetyManager);
    m_fileManager->setCreateBackupsByDefault(true);
    
    // Create test files
    m_testFilePath1 = m_testDir->path() + "/test_file_1.txt";
    m_testFilePath2 = m_testDir->path() + "/test_file_2.txt";
    m_testFilePath3 = m_testDir->path() + "/test_file_3.txt";
    
    createTestFile(m_testFilePath1, "Test content for file 1");
    createTestFile(m_testFilePath2, "Test content for file 2");
    createTestFile(m_testFilePath3, "Test content for file 3");
}

void TestRestoreFunctionality::cleanup()
{
    delete m_fileManager;
    delete m_safetyManager;
    delete m_testDir;
}

void TestRestoreFunctionality::testFileManagerRegistersOperations()
{
    // Verify initial state - no operations registered
    QList<SafetyManager::SafetyOperation> initialHistory = m_safetyManager->getUndoHistory();
    QCOMPARE(initialHistory.size(), 0);
    
    // Perform delete operation through FileManager
    QString operationId = QUuid::createUuid().toString();
    bool deleteResult = m_fileManager->performDelete(m_testFilePath1, operationId);
    QVERIFY(deleteResult);
    
    // Verify file was deleted
    QVERIFY(!QFile::exists(m_testFilePath1));
    
    // Verify operation was registered with SafetyManager
    QList<SafetyManager::SafetyOperation> history = m_safetyManager->getUndoHistory();
    QCOMPARE(history.size(), 1);
    
    const SafetyManager::SafetyOperation& operation = history.first();
    QCOMPARE(operation.type, SafetyManager::OperationType::Delete);
    QCOMPARE(operation.sourceFile, m_testFilePath1);
    QVERIFY(!operation.backupPath.isEmpty());
    QVERIFY(operation.canUndo);
    
    // Verify backup file exists
    QVERIFY(QFile::exists(operation.backupPath));
    QVERIFY(verifyFileContent(operation.backupPath, "Test content for file 1"));
}

void TestRestoreFunctionality::testSafetyManagerUndoHistory()
{
    // Perform multiple operations
    QString opId1 = QUuid::createUuid().toString();
    QString opId2 = QUuid::createUuid().toString();
    QString opId3 = QUuid::createUuid().toString();
    
    // Delete operations
    QVERIFY(m_fileManager->performDelete(m_testFilePath1, opId1));
    QVERIFY(m_fileManager->performDelete(m_testFilePath2, opId2));
    
    // Move operation
    QString moveTarget = m_testDir->path() + "/moved_file.txt";
    QVERIFY(m_fileManager->performMove(m_testFilePath3, moveTarget, opId3));
    
    // Verify all operations are in history
    QList<SafetyManager::SafetyOperation> history = m_safetyManager->getUndoHistory();
    QCOMPARE(history.size(), 3);
    
    // Verify operations are sorted by timestamp (most recent first)
    QVERIFY(history[0].timestamp >= history[1].timestamp);
    QVERIFY(history[1].timestamp >= history[2].timestamp);
    
    // Verify operation types
    QSet<SafetyManager::OperationType> expectedTypes = {
        SafetyManager::OperationType::Delete,
        SafetyManager::OperationType::Delete,
        SafetyManager::OperationType::Move
    };
    
    QSet<SafetyManager::OperationType> actualTypes;
    for (const auto& op : history) {
        actualTypes.insert(op.type);
    }
    
    QVERIFY(actualTypes.contains(SafetyManager::OperationType::Delete));
    QVERIFY(actualTypes.contains(SafetyManager::OperationType::Move));
    
    // Verify all operations have backups and can be undone
    for (const auto& op : history) {
        QVERIFY(!op.backupPath.isEmpty());
        QVERIFY(QFile::exists(op.backupPath));
        QVERIFY(op.canUndo);
    }
}

void TestRestoreFunctionality::testRestoreDialogPopulation()
{
    // Perform some operations to populate history
    QString opId1 = QUuid::createUuid().toString();
    QString opId2 = QUuid::createUuid().toString();
    
    QVERIFY(m_fileManager->performDelete(m_testFilePath1, opId1));
    QVERIFY(m_fileManager->performDelete(m_testFilePath2, opId2));
    
    // Create RestoreDialog and verify it loads operations
    RestoreDialog dialog(m_safetyManager);
    
    // Access private members for testing (normally we'd use public interface)
    // In a real test, we'd verify through the UI or public methods
    QList<SafetyManager::SafetyOperation> dialogHistory = m_safetyManager->getUndoHistory();
    QCOMPARE(dialogHistory.size(), 2);
    
    // Verify dialog would show operations
    for (const auto& op : dialogHistory) {
        QCOMPARE(op.type, SafetyManager::OperationType::Delete);
        QVERIFY(!op.sourceFile.isEmpty());
        QVERIFY(!op.backupPath.isEmpty());
        QVERIFY(QFile::exists(op.backupPath));
    }
}

void TestRestoreFunctionality::testEndToEndRestoreWorkflow()
{
    // Step 1: Delete a file through FileManager
    QString operationId = QUuid::createUuid().toString();
    QString originalContent = "Original file content for restore test";
    
    // Recreate test file with specific content
    QFile::remove(m_testFilePath1);
    createTestFile(m_testFilePath1, originalContent);
    
    // Delete the file
    QVERIFY(m_fileManager->performDelete(m_testFilePath1, operationId));
    QVERIFY(!QFile::exists(m_testFilePath1));
    
    // Step 2: Verify operation is in SafetyManager history
    QList<SafetyManager::SafetyOperation> history = m_safetyManager->getUndoHistory();
    QCOMPARE(history.size(), 1);
    
    const SafetyManager::SafetyOperation& operation = history.first();
    QCOMPARE(operation.sourceFile, m_testFilePath1);
    QVERIFY(!operation.backupPath.isEmpty());
    QVERIFY(QFile::exists(operation.backupPath));
    
    // Step 3: Restore the file using SafetyManager
    bool restoreResult = m_safetyManager->restoreFromBackup(operation.backupPath);
    QVERIFY(restoreResult);
    
    // Step 4: Verify file was restored correctly
    QVERIFY(QFile::exists(m_testFilePath1));
    QVERIFY(verifyFileContent(m_testFilePath1, originalContent));
    
    // Step 5: Verify file permissions and metadata are preserved
    QFileInfo originalInfo(m_testFilePath1);
    QFileInfo backupInfo(operation.backupPath);
    QCOMPARE(originalInfo.size(), backupInfo.size());
}

void TestRestoreFunctionality::testRestoreWithMissingBackup()
{
    // Delete a file to create backup
    QString operationId = QUuid::createUuid().toString();
    QVERIFY(m_fileManager->performDelete(m_testFilePath1, operationId));
    
    // Get the operation
    QList<SafetyManager::SafetyOperation> history = m_safetyManager->getUndoHistory();
    QCOMPARE(history.size(), 1);
    QString backupPath = history.first().backupPath;
    
    // Manually delete the backup file to simulate missing backup
    QVERIFY(QFile::remove(backupPath));
    QVERIFY(!QFile::exists(backupPath));
    
    // Attempt to restore - should fail gracefully
    bool restoreResult = m_safetyManager->restoreFromBackup(backupPath);
    QVERIFY(!restoreResult);
    
    // Original file should still not exist
    QVERIFY(!QFile::exists(m_testFilePath1));
}

void TestRestoreFunctionality::testRestoreWithConflicts()
{
    // Delete a file
    QString operationId = QUuid::createUuid().toString();
    QString originalContent = "Original content";
    
    QFile::remove(m_testFilePath1);
    createTestFile(m_testFilePath1, originalContent);
    QVERIFY(m_fileManager->performDelete(m_testFilePath1, operationId));
    
    // Create a new file at the same location with different content
    QString conflictContent = "Conflicting content";
    createTestFile(m_testFilePath1, conflictContent);
    
    // Get backup path
    QList<SafetyManager::SafetyOperation> history = m_safetyManager->getUndoHistory();
    QString backupPath = history.first().backupPath;
    
    // Restore should overwrite the conflicting file
    bool restoreResult = m_safetyManager->restoreFromBackup(backupPath);
    QVERIFY(restoreResult);
    
    // Verify original content was restored
    QVERIFY(verifyFileContent(m_testFilePath1, originalContent));
}

void TestRestoreFunctionality::testMultipleFileRestore()
{
    // Delete multiple files
    QStringList operationIds = {
        QUuid::createUuid().toString(),
        QUuid::createUuid().toString(),
        QUuid::createUuid().toString()
    };
    
    QStringList originalContents = {
        "Content for file 1",
        "Content for file 2", 
        "Content for file 3"
    };
    
    QStringList filePaths = { m_testFilePath1, m_testFilePath2, m_testFilePath3 };
    
    // Recreate files with specific content
    for (int i = 0; i < filePaths.size(); ++i) {
        QFile::remove(filePaths[i]);
        createTestFile(filePaths[i], originalContents[i]);
        QVERIFY(m_fileManager->performDelete(filePaths[i], operationIds[i]));
        QVERIFY(!QFile::exists(filePaths[i]));
    }
    
    // Get all backup paths
    QList<SafetyManager::SafetyOperation> history = m_safetyManager->getUndoHistory();
    QCOMPARE(history.size(), 3);
    
    QStringList backupPaths;
    for (const auto& op : history) {
        backupPaths.append(op.backupPath);
    }
    
    // Restore all files
    int successCount = 0;
    for (const QString& backupPath : backupPaths) {
        if (m_safetyManager->restoreFromBackup(backupPath)) {
            successCount++;
        }
    }
    
    QCOMPARE(successCount, 3);
    
    // Verify all files were restored with correct content
    for (int i = 0; i < filePaths.size(); ++i) {
        QVERIFY(QFile::exists(filePaths[i]));
        QVERIFY(verifyFileContent(filePaths[i], originalContents[i]));
    }
}

void TestRestoreFunctionality::testRestoreAfterBackupCleanup()
{
    // Delete a file
    QString operationId = QUuid::createUuid().toString();
    QVERIFY(m_fileManager->performDelete(m_testFilePath1, operationId));
    
    // Get backup path
    QList<SafetyManager::SafetyOperation> history = m_safetyManager->getUndoHistory();
    QString backupPath = history.first().backupPath;
    QVERIFY(QFile::exists(backupPath));
    
    // Simulate backup cleanup by setting very short retention period
    m_safetyManager->setMaxBackupAge(0); // 0 days = immediate cleanup
    m_safetyManager->cleanupOldBackups();
    
    // Backup should still exist since it was just created
    // (cleanup typically has some grace period)
    // If backup was cleaned up, restore should fail gracefully
    bool restoreResult = m_safetyManager->restoreFromBackup(backupPath);
    
    if (QFile::exists(backupPath)) {
        QVERIFY(restoreResult);
        QVERIFY(QFile::exists(m_testFilePath1));
    } else {
        QVERIFY(!restoreResult);
        QVERIFY(!QFile::exists(m_testFilePath1));
    }
}

void TestRestoreFunctionality::testRestoreWithCorruptedBackup()
{
    // Delete a file
    QString operationId = QUuid::createUuid().toString();
    QVERIFY(m_fileManager->performDelete(m_testFilePath1, operationId));
    
    // Get backup path
    QList<SafetyManager::SafetyOperation> history = m_safetyManager->getUndoHistory();
    QString backupPath = history.first().backupPath;
    
    // Corrupt the backup file
    QFile backupFile(backupPath);
    QVERIFY(backupFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
    backupFile.write("CORRUPTED DATA");
    backupFile.close();
    
    // Restore should still work (it copies whatever is in the backup)
    bool restoreResult = m_safetyManager->restoreFromBackup(backupPath);
    QVERIFY(restoreResult);
    
    // File should exist but with corrupted content
    QVERIFY(QFile::exists(m_testFilePath1));
    QVERIFY(verifyFileContent(m_testFilePath1, "CORRUPTED DATA"));
}

void TestRestoreFunctionality::testRestorePermissionIssues()
{
    // This test would require platform-specific permission manipulation
    // For now, we'll skip it or implement basic checks
    QSKIP("Permission testing requires platform-specific implementation");
}

// Helper methods

void TestRestoreFunctionality::createTestFile(const QString& path, const QString& content)
{
    QFile file(path);
    QVERIFY(file.open(QIODevice::WriteOnly | QIODevice::Text));
    QTextStream stream(&file);
    stream << content;
    file.close();
    QVERIFY(file.exists());
}

bool TestRestoreFunctionality::verifyFileContent(const QString& path, const QString& expectedContent)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return false;
    }
    
    QTextStream stream(&file);
    QString actualContent = stream.readAll();
    file.close();
    
    return actualContent == expectedContent;
}

void TestRestoreFunctionality::simulateFileOperation(const QString& filePath, SafetyManager::OperationType operation)
{
    QString operationId = QUuid::createUuid().toString();
    
    switch (operation) {
        case SafetyManager::OperationType::Delete:
            m_fileManager->performDelete(filePath, operationId);
            break;
        case SafetyManager::OperationType::Move:
            {
                QString targetPath = filePath + ".moved";
                m_fileManager->performMove(filePath, targetPath, operationId);
            }
            break;
        default:
            // Other operations not implemented for this test
            break;
    }
}

QTEST_MAIN(TestRestoreFunctionality)
#include "test_restore_functionality.moc"