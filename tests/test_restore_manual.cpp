#include <QApplication>
#include <QTemporaryDir>
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <QUuid>

#include "file_manager.h"
#include "safety_manager.h"
#include "restore_dialog.h"

/**
 * @brief Manual test to verify restore functionality works end-to-end
 * 
 * This test creates test files, deletes them through FileManager,
 * then verifies they appear in RestoreDialog and can be restored.
 */
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    qDebug() << "=== Manual Restore Functionality Test ===";
    
    // Create temporary directory for test
    QTemporaryDir testDir;
    if (!testDir.isValid()) {
        qDebug() << "Failed to create temporary directory";
        return 1;
    }
    
    qDebug() << "Test directory:" << testDir.path();
    
    // Initialize SafetyManager
    SafetyManager safetyManager;
    QString backupDir = testDir.path() + "/backups";
    safetyManager.setBackupDirectory(backupDir);
    safetyManager.setSafetyLevel(SafetyManager::SafetyLevel::Standard);
    
    // Initialize FileManager
    FileManager fileManager;
    fileManager.setSafetyManager(&safetyManager);
    fileManager.setCreateBackupsByDefault(true);
    
    // Create test files
    QString testFile1 = testDir.path() + "/test_file_1.txt";
    QString testFile2 = testDir.path() + "/test_file_2.txt";
    
    // Create test file 1
    {
        QFile file(testFile1);
        if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream stream(&file);
            stream << "This is test file 1 content";
            file.close();
            qDebug() << "Created test file 1:" << testFile1;
        } else {
            qDebug() << "Failed to create test file 1";
            return 1;
        }
    }
    
    // Create test file 2
    {
        QFile file(testFile2);
        if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream stream(&file);
            stream << "This is test file 2 content";
            file.close();
            qDebug() << "Created test file 2:" << testFile2;
        } else {
            qDebug() << "Failed to create test file 2";
            return 1;
        }
    }
    
    // Verify files exist
    qDebug() << "Test file 1 exists:" << QFile::exists(testFile1);
    qDebug() << "Test file 2 exists:" << QFile::exists(testFile2);
    
    // Delete files through FileManager
    QString opId1 = QUuid::createUuid().toString();
    QString opId2 = QUuid::createUuid().toString();
    
    qDebug() << "\\n=== Deleting Files ===";
    bool delete1 = fileManager.performDelete(testFile1, opId1);
    bool delete2 = fileManager.performDelete(testFile2, opId2);
    
    qDebug() << "Delete file 1 result:" << delete1;
    qDebug() << "Delete file 2 result:" << delete2;
    qDebug() << "Test file 1 exists after delete:" << QFile::exists(testFile1);
    qDebug() << "Test file 2 exists after delete:" << QFile::exists(testFile2);
    
    // Check SafetyManager undo history
    qDebug() << "\\n=== Checking Undo History ===";
    QList<SafetyManager::SafetyOperation> history = safetyManager.getUndoHistory();
    qDebug() << "Operations in history:" << history.size();
    
    for (int i = 0; i < history.size(); ++i) {
        const auto& op = history[i];
        qDebug() << QString("Operation %1:").arg(i + 1);
        qDebug() << "  - Type:" << static_cast<int>(op.type);
        qDebug() << "  - Source file:" << op.sourceFile;
        qDebug() << "  - Backup path:" << op.backupPath;
        qDebug() << "  - Can undo:" << op.canUndo;
        qDebug() << "  - Backup exists:" << QFile::exists(op.backupPath);
    }
    
    if (history.isEmpty()) {
        qDebug() << "ERROR: No operations found in history!";
        qDebug() << "This indicates the FileManager is not properly registering operations with SafetyManager.";
        return 1;
    }
    
    // Test restore functionality
    qDebug() << "\\n=== Testing Restore ===";
    if (!history.isEmpty()) {
        const auto& firstOp = history.first();
        qDebug() << "Attempting to restore:" << firstOp.sourceFile;
        qDebug() << "From backup:" << firstOp.backupPath;
        
        bool restoreResult = safetyManager.restoreFromBackup(firstOp.backupPath);
        qDebug() << "Restore result:" << restoreResult;
        qDebug() << "File exists after restore:" << QFile::exists(firstOp.sourceFile);
        
        if (restoreResult && QFile::exists(firstOp.sourceFile)) {
            // Verify content
            QFile restoredFile(firstOp.sourceFile);
            if (restoredFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
                QTextStream stream(&restoredFile);
                QString content = stream.readAll();
                restoredFile.close();
                qDebug() << "Restored file content:" << content;
            }
        }
    }
    
    // Show RestoreDialog for manual verification
    qDebug() << "\\n=== Showing RestoreDialog ===";
    qDebug() << "Opening RestoreDialog for manual verification...";
    
    RestoreDialog dialog(&safetyManager);
    dialog.setWindowTitle("Manual Test - Restore Dialog");
    dialog.resize(800, 600);
    
    // Show dialog and wait for user interaction
    int result = dialog.exec();
    
    qDebug() << "Dialog closed with result:" << result;
    qDebug() << "\\n=== Test Complete ===";
    
    // Final verification
    qDebug() << "Final state:";
    qDebug() << "Test file 1 exists:" << QFile::exists(testFile1);
    qDebug() << "Test file 2 exists:" << QFile::exists(testFile2);
    
    return 0;
}