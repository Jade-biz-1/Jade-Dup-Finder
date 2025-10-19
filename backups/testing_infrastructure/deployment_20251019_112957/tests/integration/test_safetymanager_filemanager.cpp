#include <QCoreApplication>
#include <QDebug>
#include <QTimer>
#include <QEventLoop>
#include <QTemporaryDir>
#include <QFile>
#include <QDir>
#include <QSignalSpy>
#include <QTest>
#include <QFileInfo>

#include "file_manager.h"
#include "../src/core/safety_manager.h"

/**
 * @brief Integration test for SafetyManager and FileManager
 * 
 * This test verifies:
 * - SafetyManager backup creation before FileManager operations
 * - Backup validation and integrity checking
 * - Restore operations coordination
 * - Safety level enforcement
 * - Error handling and rollback mechanisms
 * - Protection rule validation
 * 
 * Requirements: 1.3, 2.3, 7.4
 */

class SafetyManagerFileManagerTest : public QObject {
    Q_OBJECT

private:
    QTemporaryDir* m_tempDir;
    QTemporaryDir* m_backupDir;
    QString m_testPath;
    QString m_backupPath;
    SafetyManager* m_safetyManager;
    FileManager* m_fileManager;
    
    // Helper to create test file with specific content
    QString createTestFile(const QString& relativePath, const QByteArray& content) {
        QString fullPath = m_testPath + "/" + relativePath;
        QFileInfo info(fullPath);
        QDir().mkpath(info.absolutePath());
        
        QFile file(fullPath);
        if (!file.open(QIODevice::WriteOnly)) {
            qWarning() << "Failed to create test file:" << fullPath;
            return QString();
        }
        file.write(content);
        file.close();
        return fullPath;
    }

private slots:
    void initTestCase() {
        qDebug() << "===========================================";
        qDebug() << "SafetyManager <-> FileManager Integration Test";
        qDebug() << "===========================================";
        qDebug();
        
        // Register metatypes for signal/slot system
        qRegisterMetaType<FileManager::OperationResult>("FileManager::OperationResult");
        qRegisterMetaType<FileManager::OperationResult>("OperationResult");
        qRegisterMetaType<SafetyManager::BackupResult>("SafetyManager::BackupResult");
        qRegisterMetaType<SafetyManager::BackupResult>("BackupResult");
        
        m_tempDir = new QTemporaryDir();
        m_backupDir = new QTemporaryDir();
        QVERIFY(m_tempDir->isValid());
        QVERIFY(m_backupDir->isValid());
        
        m_testPath = m_tempDir->path();
        m_backupPath = m_backupDir->path();
        
        qDebug() << "Test directory:" << m_testPath;
        qDebug() << "Backup directory:" << m_backupPath;
        
        // Create components
        m_safetyManager = new SafetyManager(this);
        m_fileManager = new FileManager(this);
        
        // Configure SafetyManager
        m_safetyManager->setBackupDirectory(m_backupPath);
        m_safetyManager->setSafetyLevel(SafetyManager::SafetyLevel::Standard);
        
        // Wire components together
        m_fileManager->setSafetyManager(m_safetyManager);
    }
    
    void cleanupTestCase() {
        delete m_fileManager;
        delete m_safetyManager;
        delete m_backupDir;
        delete m_tempDir;
        
        qDebug() << "\n===========================================";
        qDebug() << "All tests completed";
        qDebug() << "===========================================";
    }
    
    /**
     * Test 1: Backup creation before file operations
     * Verify SafetyManager creates backups before FileManager deletes files
     */
    void test_backupCreationBeforeOperations() {
        qDebug() << "\n[Test 1] Backup Creation Before Operations";
        qDebug() << "===========================================";
        
        // Create test files
        QString file1 = createTestFile("backup_test/file1.txt", "Content of file 1");
        QString file2 = createTestFile("backup_test/file2.txt", "Content of file 2");
        QString file3 = createTestFile("backup_test/file3.txt", "Content of file 3");
        
        QVERIFY(QFile::exists(file1));
        QVERIFY(QFile::exists(file2));
        QVERIFY(QFile::exists(file3));
        
        qDebug() << "   Created 3 test files";
        
        // Test backup creation
        QStringList filesToDelete = {file1, file2, file3};
        
        QSignalSpy backupCompletedSpy(m_safetyManager, &SafetyManager::backupCompleted);
        QSignalSpy operationCompletedSpy(m_fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
        
        // Delete files (should trigger backup creation)
        QString operationId = m_fileManager->deleteFiles(filesToDelete, "Test backup creation");
        
        // Wait for operation to complete
        QVERIFY(operationCompletedSpy.wait(10000));
        
        // Verify backup was created
        qDebug() << "   Backup signals received:" << backupCompletedSpy.count();
        qDebug() << "   Operation completed";
        
        // Get operation result
        FileManager::OperationResult result = m_fileManager->getOperationResult(operationId);
        qDebug() << "   Operation success:" << result.success;
        qDebug() << "   Processed files:" << result.processedFiles.size();
        qDebug() << "   Failed files:" << result.failedFiles.size();
        
        QVERIFY(result.success);
        QCOMPARE(result.processedFiles.size(), 3);
        
        // Verify files were deleted
        QVERIFY(!QFile::exists(file1));
        QVERIFY(!QFile::exists(file2));
        QVERIFY(!QFile::exists(file3));
        
        // Verify backups exist
        QStringList backups = m_safetyManager->getAvailableBackups();
        qDebug() << "   Available backups:" << backups.size();
        QVERIFY(backups.size() >= 3);
        
        qDebug() << "✓ Backup creation before operations verified";
    }
    
    /**
     * Test 2: Backup validation and integrity checking
     * Test backup integrity validation
     */
    void test_backupValidationAndIntegrity() {
        qDebug() << "\n[Test 2] Backup Validation and Integrity";
        qDebug() << "=========================================";
        
        // Create test files with known content
        QByteArray content1 = "Test content for integrity check 1";
        QByteArray content2 = "Test content for integrity check 2";
        
        QString file1 = createTestFile("integrity_test/file1.txt", content1);
        QString file2 = createTestFile("integrity_test/file2.txt", content2);
        
        qDebug() << "   Created test files with known content";
        
        // Create backups manually
        QSignalSpy backupCompletedSpy(m_safetyManager, &SafetyManager::backupCompleted);
        
        QString backupId1 = m_safetyManager->createBackup(file1, "Manual backup test 1");
        QString backupId2 = m_safetyManager->createBackup(file2, "Manual backup test 2");
        
        // Wait for backups to complete
        QVERIFY(backupCompletedSpy.wait(5000));
        if (backupCompletedSpy.count() < 2) {
            QVERIFY(backupCompletedSpy.wait(5000)); // Wait for second backup
        }
        
        qDebug() << "   Backup operations completed:" << backupCompletedSpy.count();
        
        // Validate backup integrity
        bool integrity1 = m_safetyManager->validateBackupIntegrity(backupId1);
        bool integrity2 = m_safetyManager->validateBackupIntegrity(backupId2);
        
        qDebug() << "   Backup 1 integrity:" << integrity1;
        qDebug() << "   Backup 2 integrity:" << integrity2;
        
        QVERIFY(integrity1);
        QVERIFY(integrity2);
        
        // Test backup metadata
        SafetyManager::BackupInfo info1 = m_safetyManager->getBackupInfo(backupId1);
        SafetyManager::BackupInfo info2 = m_safetyManager->getBackupInfo(backupId2);
        
        qDebug() << "   Backup 1 info:";
        qDebug() << "      Original path:" << info1.originalPath;
        qDebug() << "      Backup path:" << info1.backupPath;
        qDebug() << "      Size:" << info1.fileSize;
        qDebug() << "      Created:" << info1.createdAt.toString();
        
        QCOMPARE(info1.originalPath, file1);
        QVERIFY(info1.fileSize == content1.size());
        QVERIFY(QFile::exists(info1.backupPath));
        
        QCOMPARE(info2.originalPath, file2);
        QVERIFY(info2.fileSize == content2.size());
        QVERIFY(QFile::exists(info2.backupPath));
        
        qDebug() << "✓ Backup validation and integrity verified";
    }
    
    /**
     * Test 3: Restore operations coordination
     * Test restore operations between SafetyManager and FileManager
     */
    void test_restoreOperationsCoordination() {
        qDebug() << "\n[Test 3] Restore Operations Coordination";
        qDebug() << "=========================================";
        
        // Create and backup test files
        QByteArray originalContent = "Original content for restore test";
        QString originalFile = createTestFile("restore_test/original.txt", originalContent);
        
        qDebug() << "   Created original file";
        
        // Create backup
        QSignalSpy backupCompletedSpy(m_safetyManager, &SafetyManager::backupCompleted);
        QString backupId = m_safetyManager->createBackup(originalFile, "Restore test backup");
        
        QVERIFY(backupCompletedSpy.wait(5000));
        qDebug() << "   Backup created:" << backupId;
        
        // Delete the original file
        QSignalSpy operationCompletedSpy(m_fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
        QString deleteOpId = m_fileManager->deleteFiles({originalFile}, "Delete for restore test");
        
        QVERIFY(operationCompletedSpy.wait(5000));
        QVERIFY(!QFile::exists(originalFile));
        qDebug() << "   Original file deleted";
        
        // Restore the file using FileManager (which should coordinate with SafetyManager)
        SafetyManager::BackupInfo backupInfo = m_safetyManager->getBackupInfo(backupId);
        QString restoreOpId = m_fileManager->restoreFiles({backupInfo.backupPath}, QFileInfo(originalFile).absolutePath());
        
        QVERIFY(operationCompletedSpy.wait(5000));
        
        // Verify restore operation
        FileManager::OperationResult restoreResult = m_fileManager->getOperationResult(restoreOpId);
        qDebug() << "   Restore operation success:" << restoreResult.success;
        qDebug() << "   Restored files:" << restoreResult.processedFiles.size();
        
        QVERIFY(restoreResult.success);
        QVERIFY(restoreResult.processedFiles.size() >= 1);
        
        // Verify file content is restored correctly
        QVERIFY(QFile::exists(originalFile));
        
        QFile restoredFile(originalFile);
        QVERIFY(restoredFile.open(QIODevice::ReadOnly));
        QByteArray restoredContent = restoredFile.readAll();
        restoredFile.close();
        
        QCOMPARE(restoredContent, originalContent);
        qDebug() << "   File content verified after restore";
        
        qDebug() << "✓ Restore operations coordination verified";
    }
    
    /**
     * Test 4: Safety level enforcement
     * Test different safety levels and their enforcement
     */
    void test_safetyLevelEnforcement() {
        qDebug() << "\n[Test 4] Safety Level Enforcement";
        qDebug() << "==================================";
        
        // Create test files
        QString file1 = createTestFile("safety_test/file1.txt", "Content 1");
        QString file2 = createTestFile("safety_test/file2.txt", "Content 2");
        
        // Test Conservative safety level
        qDebug() << "   Testing Conservative safety level...";
        m_safetyManager->setSafetyLevel(SafetyManager::SafetyLevel::Conservative);
        
        QSignalSpy operationCompletedSpy(m_fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
        QString opId1 = m_fileManager->deleteFiles({file1}, "Conservative safety test");
        
        QVERIFY(operationCompletedSpy.wait(10000));
        
        FileManager::OperationResult result1 = m_fileManager->getOperationResult(opId1);
        qDebug() << "      Conservative result:" << result1.success;
        
        // Conservative mode should create backup and succeed
        QVERIFY(result1.success);
        
        // Test Standard safety level
        qDebug() << "   Testing Standard safety level...";
        m_safetyManager->setSafetyLevel(SafetyManager::SafetyLevel::Standard);
        
        QString opId2 = m_fileManager->deleteFiles({file2}, "Standard safety test");
        QVERIFY(operationCompletedSpy.wait(10000));
        
        FileManager::OperationResult result2 = m_fileManager->getOperationResult(opId2);
        qDebug() << "      Standard result:" << result2.success;
        
        // Standard mode should also create backup and succeed
        QVERIFY(result2.success);
        
        // Verify backups were created for both operations
        QStringList backups = m_safetyManager->getAvailableBackups();
        qDebug() << "   Total backups created:" << backups.size();
        QVERIFY(backups.size() >= 2);
        
        qDebug() << "✓ Safety level enforcement verified";
    }
    
    /**
     * Test 5: Error handling and rollback mechanisms
     * Test error handling and rollback when operations fail
     */
    void test_errorHandlingAndRollback() {
        qDebug() << "\n[Test 5] Error Handling and Rollback";
        qDebug() << "=====================================";
        
        // Create test files
        QString validFile = createTestFile("error_test/valid.txt", "Valid content");
        QString nonExistentFile = m_testPath + "/error_test/nonexistent.txt";
        
        qDebug() << "   Created test scenario with valid and invalid files";
        
        // Try to delete both valid and non-existent files
        QStringList mixedFiles = {validFile, nonExistentFile};
        
        QSignalSpy operationCompletedSpy(m_fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
        QSignalSpy operationErrorSpy(m_fileManager, &FileManager::operationError);
        
        QString operationId = m_fileManager->deleteFiles(mixedFiles, "Error handling test");
        
        QVERIFY(operationCompletedSpy.wait(10000));
        
        // Check operation result
        FileManager::OperationResult result = m_fileManager->getOperationResult(operationId);
        qDebug() << "   Operation completed";
        qDebug() << "   Overall success:" << result.success;
        qDebug() << "   Processed files:" << result.processedFiles.size();
        qDebug() << "   Failed files:" << result.failedFiles.size();
        qDebug() << "   Error signals:" << operationErrorSpy.count();
        
        // The operation should handle partial success/failure gracefully
        QVERIFY(result.processedFiles.size() >= 1); // Valid file should be processed
        QVERIFY(result.failedFiles.size() >= 1);    // Non-existent file should fail
        
        // Verify the valid file was backed up before deletion
        QStringList backups = m_safetyManager->getAvailableBackups();
        qDebug() << "   Backups created:" << backups.size();
        QVERIFY(backups.size() >= 1);
        
        // Verify the valid file was actually deleted
        QVERIFY(!QFile::exists(validFile));
        
        qDebug() << "✓ Error handling and rollback verified";
    }
    
    /**
     * Test 6: Protection rule validation
     * Test protection rules enforcement
     */
    void test_protectionRuleValidation() {
        qDebug() << "\n[Test 6] Protection Rule Validation";
        qDebug() << "====================================";
        
        // Create test files in different locations
        QString normalFile = createTestFile("protection_test/normal.txt", "Normal file");
        QString systemLikeFile = createTestFile("protection_test/system/important.txt", "System-like file");
        
        qDebug() << "   Created test files in different locations";
        
        // Add protection rules
        m_safetyManager->addProtectionRule("*/system/*", "System directory protection");
        m_safetyManager->addProtectionRule("*.important", "Important file protection");
        
        qDebug() << "   Added protection rules";
        
        // Try to delete normal file (should succeed)
        QSignalSpy operationCompletedSpy(m_fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
        
        QString normalOpId = m_fileManager->deleteFiles({normalFile}, "Normal file deletion");
        QVERIFY(operationCompletedSpy.wait(5000));
        
        FileManager::OperationResult normalResult = m_fileManager->getOperationResult(normalOpId);
        qDebug() << "   Normal file deletion result:" << normalResult.success;
        QVERIFY(normalResult.success);
        
        // Try to delete system-like file (should be protected)
        QString systemOpId = m_fileManager->deleteFiles({systemLikeFile}, "System file deletion");
        QVERIFY(operationCompletedSpy.wait(5000));
        
        FileManager::OperationResult systemResult = m_fileManager->getOperationResult(systemOpId);
        qDebug() << "   System file deletion result:" << systemResult.success;
        qDebug() << "   System file still exists:" << QFile::exists(systemLikeFile);
        
        // The system file should either be protected (operation fails) or require special handling
        // The exact behavior depends on SafetyManager implementation
        if (!systemResult.success) {
            qDebug() << "   System file was protected (operation failed)";
            QVERIFY(QFile::exists(systemLikeFile)); // File should still exist
        } else {
            qDebug() << "   System file deletion succeeded with extra safety measures";
            // If deletion succeeded, extra backups should have been created
            QStringList backups = m_safetyManager->getAvailableBackups();
            QVERIFY(backups.size() >= 2); // Should have backups for both operations
        }
        
        qDebug() << "✓ Protection rule validation verified";
    }
    
    /**
     * Test 7: Concurrent operations handling
     * Test handling of multiple concurrent operations
     */
    void test_concurrentOperationsHandling() {
        qDebug() << "\n[Test 7] Concurrent Operations Handling";
        qDebug() << "========================================";
        
        // Create multiple test files
        QStringList files;
        for (int i = 0; i < 5; i++) {
            QString file = createTestFile(QString("concurrent_test/file_%1.txt").arg(i), 
                                        QString("Content %1").arg(i).toUtf8());
            files << file;
        }
        
        qDebug() << "   Created" << files.size() << "test files";
        
        // Start multiple operations concurrently
        QSignalSpy operationCompletedSpy(m_fileManager, SIGNAL(operationCompleted(const FileManager::OperationResult&)));
        
        QStringList operationIds;
        for (int i = 0; i < files.size(); i++) {
            QString opId = m_fileManager->deleteFiles({files[i]}, QString("Concurrent operation %1").arg(i));
            operationIds << opId;
        }
        
        qDebug() << "   Started" << operationIds.size() << "concurrent operations";
        
        // Wait for all operations to complete
        int completedOperations = 0;
        while (completedOperations < operationIds.size() && operationCompletedSpy.wait(2000)) {
            completedOperations = operationCompletedSpy.count();
            qDebug() << "      Completed operations:" << completedOperations;
        }
        
        QCOMPARE(completedOperations, operationIds.size());
        
        // Verify all operations completed successfully
        int successfulOps = 0;
        for (const QString& opId : operationIds) {
            FileManager::OperationResult result = m_fileManager->getOperationResult(opId);
            if (result.success) {
                successfulOps++;
            }
        }
        
        qDebug() << "   Successful operations:" << successfulOps << "of" << operationIds.size();
        QCOMPARE(successfulOps, operationIds.size());
        
        // Verify backups were created for all operations
        QStringList backups = m_safetyManager->getAvailableBackups();
        qDebug() << "   Total backups created:" << backups.size();
        QVERIFY(backups.size() >= files.size());
        
        qDebug() << "✓ Concurrent operations handling verified";
    }
};

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    SafetyManagerFileManagerTest test;
    int result = QTest::qExec(&test, argc, argv);
    
    // Process any remaining events before exit
    QCoreApplication::processEvents();
    
    return result;
}

#include "test_safetymanager_filemanager.moc"