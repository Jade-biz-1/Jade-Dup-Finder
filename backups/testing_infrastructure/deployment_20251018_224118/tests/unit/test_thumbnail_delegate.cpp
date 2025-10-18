#include <QtTest/QtTest>
#include <QtWidgets/QApplication>
#include <QtWidgets/QTreeWidget>
#include "thumbnail_cache.h"
#include "thumbnail_delegate.h"

class TestThumbnailDelegate : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();

    // Test cases
    void testConstruction();
    void testThumbnailSizeConfiguration();
    void testEnableDisableThumbnails();
    void testSizeHint();
    void testPaintWithoutThumbnail();
    void testFileItemDetection();

private:
    ThumbnailCache* m_cache;
    ThumbnailDelegate* m_delegate;
    QTreeWidget* m_treeWidget;
};

void TestThumbnailDelegate::initTestCase()
{
    // Initialize test environment
    m_cache = nullptr;
    m_delegate = nullptr;
    m_treeWidget = nullptr;
}

void TestThumbnailDelegate::cleanupTestCase()
{
    // Cleanup test environment
}

void TestThumbnailDelegate::init()
{
    // Create fresh instances for each test
    m_cache = new ThumbnailCache();
    m_delegate = new ThumbnailDelegate(m_cache);
    m_treeWidget = new QTreeWidget();
    m_treeWidget->setHeaderLabels({"File", "Size", "Modified"});
}

void TestThumbnailDelegate::cleanup()
{
    // Clean up after each test
    delete m_treeWidget;
    delete m_delegate;
    delete m_cache;
    
    m_treeWidget = nullptr;
    m_delegate = nullptr;
    m_cache = nullptr;
}

void TestThumbnailDelegate::testConstruction()
{
    // Test that delegate is constructed properly
    QVERIFY(m_delegate != nullptr);
    QVERIFY(m_cache != nullptr);
    
    // Test default values
    QCOMPARE(m_delegate->thumbnailSize(), 48); // DEFAULT_THUMBNAIL_SIZE
    QCOMPARE(m_delegate->thumbnailsEnabled(), true);
}

void TestThumbnailDelegate::testThumbnailSizeConfiguration()
{
    // Test setting thumbnail size
    m_delegate->setThumbnailSize(64);
    QCOMPARE(m_delegate->thumbnailSize(), 64);
    
    m_delegate->setThumbnailSize(128);
    QCOMPARE(m_delegate->thumbnailSize(), 128);
    
    // Test invalid sizes (should be ignored)
    m_delegate->setThumbnailSize(0);
    QCOMPARE(m_delegate->thumbnailSize(), 128); // Should remain unchanged
    
    m_delegate->setThumbnailSize(-10);
    QCOMPARE(m_delegate->thumbnailSize(), 128); // Should remain unchanged
    
    m_delegate->setThumbnailSize(300); // Too large
    QCOMPARE(m_delegate->thumbnailSize(), 128); // Should remain unchanged
}

void TestThumbnailDelegate::testEnableDisableThumbnails()
{
    // Test enabling/disabling thumbnails
    QCOMPARE(m_delegate->thumbnailsEnabled(), true);
    
    m_delegate->setThumbnailsEnabled(false);
    QCOMPARE(m_delegate->thumbnailsEnabled(), false);
    
    m_delegate->setThumbnailsEnabled(true);
    QCOMPARE(m_delegate->thumbnailsEnabled(), true);
}

void TestThumbnailDelegate::testSizeHint()
{
    // Set up tree widget with delegate
    m_treeWidget->setItemDelegateForColumn(0, m_delegate);
    
    // Create a group item (top-level)
    QTreeWidgetItem* groupItem = new QTreeWidgetItem(m_treeWidget);
    groupItem->setText(0, "Group 1");
    
    // Create a file item (child)
    QTreeWidgetItem* fileItem = new QTreeWidgetItem(groupItem);
    fileItem->setText(0, "test.jpg");
    fileItem->setData(0, Qt::UserRole, "/path/to/test.jpg");
    
    // Get size hint for file item
    QStyleOptionViewItem option;
    QModelIndex fileIndex = m_treeWidget->indexFromItem(fileItem, 0);
    QSize hint = m_delegate->sizeHint(option, fileIndex);
    
    // Size hint should include space for thumbnail
    int expectedMinHeight = m_delegate->thumbnailSize() + 8; // 2 * THUMBNAIL_MARGIN
    QVERIFY(hint.height() >= expectedMinHeight);
    
    // Test with thumbnails disabled
    m_delegate->setThumbnailsEnabled(false);
    QSize hintDisabled = m_delegate->sizeHint(option, fileIndex);
    // Size should be smaller when thumbnails are disabled
    QVERIFY(hintDisabled.height() < hint.height());
}

void TestThumbnailDelegate::testPaintWithoutThumbnail()
{
    // This test verifies that paint doesn't crash when thumbnail is not available
    // We can't easily test the visual output, but we can ensure it doesn't crash
    
    m_treeWidget->setItemDelegateForColumn(0, m_delegate);
    
    // Create a file item
    QTreeWidgetItem* groupItem = new QTreeWidgetItem(m_treeWidget);
    groupItem->setText(0, "Group 1");
    
    QTreeWidgetItem* fileItem = new QTreeWidgetItem(groupItem);
    fileItem->setText(0, "test.jpg");
    fileItem->setData(0, Qt::UserRole, "/nonexistent/test.jpg");
    
    // Force a repaint (this will call paint internally)
    m_treeWidget->show();
    QTest::qWait(100); // Wait for paint events to process
    
    // If we get here without crashing, the test passes
    QVERIFY(true);
}

void TestThumbnailDelegate::testFileItemDetection()
{
    // Create tree structure
    QTreeWidgetItem* groupItem = new QTreeWidgetItem(m_treeWidget);
    groupItem->setText(0, "Group 1");
    
    QTreeWidgetItem* fileItem = new QTreeWidgetItem(groupItem);
    fileItem->setText(0, "test.jpg");
    
    // Get indices
    QModelIndex groupIndex = m_treeWidget->indexFromItem(groupItem, 0);
    QModelIndex fileIndex = m_treeWidget->indexFromItem(fileItem, 0);
    
    // Group items should not have parent
    QVERIFY(!groupIndex.parent().isValid());
    
    // File items should have parent
    QVERIFY(fileIndex.parent().isValid());
}

QTEST_MAIN(TestThumbnailDelegate)
#include "test_thumbnail_delegate.moc"
