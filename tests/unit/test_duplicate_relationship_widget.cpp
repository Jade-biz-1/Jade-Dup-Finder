#include <QtTest/QtTest>
#include <QApplication>
#include <QSignalSpy>
#include "duplicate_relationship_widget.h"

class TestDuplicateRelationshipWidget : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testConstruction();
    void testSetDuplicateGroups();
    void testClearVisualization();
    void testHighlightFile();
    void testZoomOperations();
    void testLayoutModes();
    void testFileSelection();
    void testSignalEmission();

private:
    DuplicateRelationshipWidget* m_widget;
    QList<DuplicateRelationshipWidget::DuplicateGroup> createTestGroups();
};

void TestDuplicateRelationshipWidget::initTestCase()
{
    m_widget = new DuplicateRelationshipWidget();
}

void TestDuplicateRelationshipWidget::cleanupTestCase()
{
    delete m_widget;
    m_widget = nullptr;
}

void TestDuplicateRelationshipWidget::testConstruction()
{
    QVERIFY(m_widget != nullptr);
    QCOMPARE(m_widget->getGroupCount(), 0);
    QCOMPARE(m_widget->getTotalFiles(), 0);
}

void TestDuplicateRelationshipWidget::testSetDuplicateGroups()
{
    auto testGroups = createTestGroups();
    m_widget->setDuplicateGroups(testGroups);
    
    QCOMPARE(m_widget->getGroupCount(), testGroups.size());
    
    int expectedTotalFiles = 0;
    for (const auto& group : testGroups) {
        expectedTotalFiles += group.files.size();
    }
    QCOMPARE(m_widget->getTotalFiles(), expectedTotalFiles);
}

void TestDuplicateRelationshipWidget::testClearVisualization()
{
    auto testGroups = createTestGroups();
    m_widget->setDuplicateGroups(testGroups);
    
    QVERIFY(m_widget->getGroupCount() > 0);
    
    m_widget->clearVisualization();
    
    QCOMPARE(m_widget->getGroupCount(), 0);
    QCOMPARE(m_widget->getTotalFiles(), 0);
}

void TestDuplicateRelationshipWidget::testHighlightFile()
{
    auto testGroups = createTestGroups();
    m_widget->setDuplicateGroups(testGroups);
    
    if (!testGroups.isEmpty() && !testGroups.first().files.isEmpty()) {
        QString filePath = testGroups.first().files.first().filePath;
        
        // This should not crash
        m_widget->highlightFile(filePath);
        
        // Test with non-existent file
        m_widget->highlightFile("/non/existent/file.txt");
    }
}

void TestDuplicateRelationshipWidget::testZoomOperations()
{
    // Test zoom operations don't crash
    m_widget->zoomIn();
    m_widget->zoomOut();
    m_widget->resetZoom();
    m_widget->fitToView();
}

void TestDuplicateRelationshipWidget::testLayoutModes()
{
    auto testGroups = createTestGroups();
    m_widget->setDuplicateGroups(testGroups);
    
    // Test that layout refresh doesn't crash
    m_widget->refreshVisualization();
}

void TestDuplicateRelationshipWidget::testFileSelection()
{
    auto testGroups = createTestGroups();
    m_widget->setDuplicateGroups(testGroups);
    
    if (!testGroups.isEmpty() && !testGroups.first().files.isEmpty()) {
        QStringList filePaths;
        filePaths << testGroups.first().files.first().filePath;
        
        m_widget->setSelectedFiles(filePaths);
        
        // Test with empty selection
        m_widget->setSelectedFiles(QStringList());
    }
}

void TestDuplicateRelationshipWidget::testSignalEmission()
{
    QSignalSpy fileClickedSpy(m_widget, &DuplicateRelationshipWidget::fileClicked);
    QSignalSpy selectionChangedSpy(m_widget, &DuplicateRelationshipWidget::selectionChanged);
    
    QVERIFY(fileClickedSpy.isValid());
    QVERIFY(selectionChangedSpy.isValid());
    
    // Signals should be emittable (even if not triggered in tests)
    QCOMPARE(fileClickedSpy.count(), 0);
    QCOMPARE(selectionChangedSpy.count(), 0);
}

QList<DuplicateRelationshipWidget::DuplicateGroup> TestDuplicateRelationshipWidget::createTestGroups()
{
    QList<DuplicateRelationshipWidget::DuplicateGroup> groups;
    
    // Create test group 1
    DuplicateRelationshipWidget::DuplicateGroup group1;
    group1.groupId = "group1";
    group1.hash = "abc123";
    group1.totalFiles = 2;
    group1.totalSize = 2048;
    group1.groupColor = QColor(255, 0, 0);
    
    DuplicateRelationshipWidget::FileNode file1;
    file1.filePath = "/test/file1.txt";
    file1.fileName = "file1.txt";
    file1.fileSize = 1024;
    file1.hash = "abc123";
    file1.isRecommended = true;
    
    DuplicateRelationshipWidget::FileNode file2;
    file2.filePath = "/test/file2.txt";
    file2.fileName = "file2.txt";
    file2.fileSize = 1024;
    file2.hash = "abc123";
    file2.isRecommended = false;
    
    group1.files << file1 << file2;
    groups << group1;
    
    // Create test group 2
    DuplicateRelationshipWidget::DuplicateGroup group2;
    group2.groupId = "group2";
    group2.hash = "def456";
    group2.totalFiles = 3;
    group2.totalSize = 3072;
    group2.groupColor = QColor(0, 255, 0);
    
    DuplicateRelationshipWidget::FileNode file3;
    file3.filePath = "/test/file3.txt";
    file3.fileName = "file3.txt";
    file3.fileSize = 1024;
    file3.hash = "def456";
    file3.isRecommended = true;
    
    DuplicateRelationshipWidget::FileNode file4;
    file4.filePath = "/test/file4.txt";
    file4.fileName = "file4.txt";
    file4.fileSize = 1024;
    file4.hash = "def456";
    file4.isRecommended = false;
    
    DuplicateRelationshipWidget::FileNode file5;
    file5.filePath = "/test/file5.txt";
    file5.fileName = "file5.txt";
    file5.fileSize = 1024;
    file5.hash = "def456";
    file5.isRecommended = false;
    
    group2.files << file3 << file4 << file5;
    groups << group2;
    
    return groups;
}

QTEST_MAIN(TestDuplicateRelationshipWidget)
#include "test_duplicate_relationship_widget.moc"