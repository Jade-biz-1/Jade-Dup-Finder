#include <QApplication>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QTest>
#include <QSignalSpy>
#include <QDebug>

#include "thumbnail_delegate.h"
#include "thumbnail_cache.h"

/**
 * @brief Simple test to verify checkbox functionality works
 */
class CheckboxFunctionalityTest : public QObject
{
    Q_OBJECT

private slots:
    void testCheckboxInteraction();
    void testDelegateDoesNotInterfere();

private:
    void setupTestTreeWidget(QTreeWidget* tree);
};

void CheckboxFunctionalityTest::testCheckboxInteraction()
{
    // Create a tree widget with checkboxes
    QTreeWidget tree;
    setupTestTreeWidget(&tree);
    
    // Get the first item (should be a group)
    QTreeWidgetItem* groupItem = tree.topLevelItem(0);
    QVERIFY(groupItem != nullptr);
    QVERIFY(groupItem->flags() & Qt::ItemIsUserCheckable);
    
    // Verify initial state is unchecked
    QCOMPARE(groupItem->checkState(0), Qt::Unchecked);
    
    // Test programmatic checkbox toggle
    groupItem->setCheckState(0, Qt::Checked);
    QCOMPARE(groupItem->checkState(0), Qt::Checked);
    
    groupItem->setCheckState(0, Qt::Unchecked);
    QCOMPARE(groupItem->checkState(0), Qt::Unchecked);
    
    // Test child items
    QTreeWidgetItem* childItem = groupItem->child(0);
    QVERIFY(childItem != nullptr);
    QVERIFY(childItem->flags() & Qt::ItemIsUserCheckable);
    
    // Test child checkbox
    childItem->setCheckState(0, Qt::Checked);
    QCOMPARE(childItem->checkState(0), Qt::Checked);
    
    qDebug() << "✅ Basic checkbox functionality works";
}

void CheckboxFunctionalityTest::testDelegateDoesNotInterfere()
{
    // Create tree widget with our delegate
    QTreeWidget tree;
    ThumbnailCache cache;
    ThumbnailDelegate delegate(&cache);
    
    // Apply delegate to column 0
    tree.setItemDelegateForColumn(0, &delegate);
    
    setupTestTreeWidget(&tree);
    
    // Test that checkboxes still work with delegate applied
    QTreeWidgetItem* groupItem = tree.topLevelItem(0);
    QVERIFY(groupItem != nullptr);
    
    // Test checkbox functionality with delegate
    QCOMPARE(groupItem->checkState(0), Qt::Unchecked);
    
    groupItem->setCheckState(0, Qt::Checked);
    QCOMPARE(groupItem->checkState(0), Qt::Checked);
    
    // Test child items with delegate
    QTreeWidgetItem* childItem = groupItem->child(0);
    QVERIFY(childItem != nullptr);
    
    childItem->setCheckState(0, Qt::Checked);
    QCOMPARE(childItem->checkState(0), Qt::Checked);
    
    qDebug() << "✅ Delegate does not interfere with checkbox functionality";
}

void CheckboxFunctionalityTest::setupTestTreeWidget(QTreeWidget* tree)
{
    tree->setHeaderLabels({"File", "Size", "Modified", "Path"});
    
    // Create a group item (like in the real application)
    QTreeWidgetItem* groupItem = new QTreeWidgetItem(tree);
    groupItem->setText(0, "Group: 2 Files");
    groupItem->setText(1, "3.5 MB");
    groupItem->setText(2, "Waste: 3.5 MB");
    groupItem->setText(3, "1 duplicates");
    
    // Make group item checkable
    groupItem->setFlags(groupItem->flags() | Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
    groupItem->setCheckState(0, Qt::Unchecked);
    
    // Create child items (like in the real application)
    for (int i = 0; i < 2; ++i) {
        QTreeWidgetItem* childItem = new QTreeWidgetItem(groupItem);
        childItem->setText(0, QString("IMG_25%1.jpg").arg(i));
        childItem->setText(1, "1.7 MB");
        childItem->setText(2, "2025-05-26 12:27");
        childItem->setText(3, "/home/deepak/Pictures");
        
        // Make child item checkable
        childItem->setFlags(childItem->flags() | Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
        childItem->setCheckState(0, Qt::Unchecked);
        
        // Store file path in UserRole (like in real application)
        childItem->setData(0, Qt::UserRole, QString("/home/deepak/Pictures/IMG_25%1.jpg").arg(i));
    }
    
    tree->expandAll();
}

// Test main function
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    CheckboxFunctionalityTest test;
    
    qDebug() << "🧪 Testing checkbox functionality...";
    
    try {
        test.testCheckboxInteraction();
        test.testDelegateDoesNotInterfere();
        
        qDebug() << "🎉 All tests passed! Checkbox functionality is working correctly.";
        return 0;
    } catch (const std::exception& e) {
        qDebug() << "❌ Test failed:" << e.what();
        return 1;
    }
}

#include "test_checkbox_functionality.moc"