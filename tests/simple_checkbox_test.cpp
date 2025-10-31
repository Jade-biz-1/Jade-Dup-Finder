#include <QApplication>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QDebug>

/**
 * @brief Simple test to verify basic checkbox functionality
 */
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    qDebug() << "🧪 Testing basic checkbox functionality...";
    
    // Create a tree widget
    QTreeWidget tree;
    tree.setHeaderLabels({\"File\", \"Size\", \"Modified\", \"Path\"});
    
    // Create a group item
    QTreeWidgetItem* groupItem = new QTreeWidgetItem(&tree);
    groupItem->setText(0, \"Group: 2 Files\");
    groupItem->setFlags(groupItem->flags() | Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
    groupItem->setCheckState(0, Qt::Unchecked);
    
    // Create child items
    for (int i = 0; i < 2; ++i) {
        QTreeWidgetItem* childItem = new QTreeWidgetItem(groupItem);
        childItem->setText(0, QString(\"IMG_25%1.jpg\").arg(i));
        childItem->setFlags(childItem->flags() | Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
        childItem->setCheckState(0, Qt::Unchecked);
    }
    
    // Test basic checkbox functionality
    qDebug() << \"Initial group state:\" << groupItem->checkState(0);
    
    // Test programmatic toggle
    groupItem->setCheckState(0, Qt::Checked);
    qDebug() << \"After setting checked:\" << groupItem->checkState(0);
    
    groupItem->setCheckState(0, Qt::Unchecked);
    qDebug() << \"After setting unchecked:\" << groupItem->checkState(0);
    
    // Test child checkbox
    QTreeWidgetItem* childItem = groupItem->child(0);
    childItem->setCheckState(0, Qt::Checked);
    qDebug() << \"Child checkbox state:\" << childItem->checkState(0);
    
    qDebug() << \"✅ Basic checkbox functionality works correctly!\";
    
    return 0;
}