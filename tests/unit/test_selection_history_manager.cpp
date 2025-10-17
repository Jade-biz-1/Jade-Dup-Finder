#include <QtTest/QtTest>
#include "selection_history_manager.h"

class TestSelectionHistoryManager : public QObject
{
    Q_OBJECT

private slots:
    void testInitialState();
    void testPushState();
    void testUndoRedo();
    void testHistoryLimit();
    void testClear();
    void testSignals();

private:
    QStringList createFileList(const QStringList& files);
};

QStringList TestSelectionHistoryManager::createFileList(const QStringList& files)
{
    return files;
}

void TestSelectionHistoryManager::testInitialState()
{
    SelectionHistoryManager manager;
    
    // Initially should have no undo/redo available
    QVERIFY(!manager.canUndo());
    QVERIFY(!manager.canRedo());
    QCOMPARE(manager.getUndoStackSize(), 0);
    QCOMPARE(manager.getRedoStackSize(), 0);
    QCOMPARE(manager.getMaxHistorySize(), 50); // Default size
}

void TestSelectionHistoryManager::testPushState()
{
    SelectionHistoryManager manager;
    
    QStringList files1 = createFileList({"file1.txt", "file2.txt"});
    manager.pushState(files1, "Selected files 1 and 2");
    
    QVERIFY(manager.canUndo());
    QVERIFY(!manager.canRedo());
    QCOMPARE(manager.getUndoStackSize(), 1);
    QCOMPARE(manager.getRedoStackSize(), 0);
    
    QStringList files2 = createFileList({"file3.txt"});
    manager.pushState(files2, "Selected file 3");
    
    QCOMPARE(manager.getUndoStackSize(), 2);
    QCOMPARE(manager.getUndoDescription(), QString("Selected file 3"));
}

void TestSelectionHistoryManager::testUndoRedo()
{
    SelectionHistoryManager manager;
    
    QStringList files1 = createFileList({"file1.txt"});
    QStringList files2 = createFileList({"file2.txt"});
    
    manager.pushState(files1, "State 1");
    manager.pushState(files2, "State 2");
    
    // Undo should return to previous state
    SelectionHistoryManager::SelectionState undoState = manager.undo();
    QCOMPARE(undoState.selectedFiles, files1);
    QCOMPARE(undoState.description, QString("State 1"));
    
    QVERIFY(manager.canRedo());
    QCOMPARE(manager.getRedoDescription(), QString("State 2"));
    
    // Redo should restore state 2
    SelectionHistoryManager::SelectionState redoState = manager.redo();
    QCOMPARE(redoState.selectedFiles, files2);
    QCOMPARE(redoState.description, QString("State 2"));
    
    QVERIFY(!manager.canRedo());
}

void TestSelectionHistoryManager::testHistoryLimit()
{
    SelectionHistoryManager manager;
    manager.setMaxHistorySize(3);
    
    // Add more states than the limit
    for (int i = 0; i < 5; ++i) {
        QStringList files = createFileList({QString("file%1.txt").arg(i)});
        manager.pushState(files, QString("State %1").arg(i));
    }
    
    // Should not exceed the limit
    QVERIFY(manager.getUndoStackSize() <= 3);
    QCOMPARE(manager.getMaxHistorySize(), 3);
}

void TestSelectionHistoryManager::testClear()
{
    SelectionHistoryManager manager;
    
    QStringList files = createFileList({"file1.txt"});
    manager.pushState(files, "Test state");
    
    QVERIFY(manager.canUndo());
    
    manager.clear();
    
    QVERIFY(!manager.canUndo());
    QVERIFY(!manager.canRedo());
    QCOMPARE(manager.getUndoStackSize(), 0);
    QCOMPARE(manager.getRedoStackSize(), 0);
}

void TestSelectionHistoryManager::testSignals()
{
    SelectionHistoryManager manager;
    
    QSignalSpy undoSpy(&manager, &SelectionHistoryManager::undoAvailabilityChanged);
    QSignalSpy redoSpy(&manager, &SelectionHistoryManager::redoAvailabilityChanged);
    QSignalSpy clearSpy(&manager, &SelectionHistoryManager::historyCleared);
    
    // Push state should emit undo available
    QStringList files = createFileList({"file1.txt"});
    manager.pushState(files, "Test state");
    
    QCOMPARE(undoSpy.count(), 1);
    QCOMPARE(undoSpy.last().at(0).toBool(), true);
    
    // Clear should emit signals
    manager.clear();
    QCOMPARE(clearSpy.count(), 1);
}

#include "test_selection_history_manager.moc"