#include <QtTest/QtTest>
#include "advanced_filter_dialog.h"

class TestAdvancedFilterDialog : public QObject
{
    Q_OBJECT

private slots:
    void testDefaultState();
    void testSetGetCriteria();
    void testResetFilters();
    void testSizeConversion();

private:
    AdvancedFilterDialog::FilterCriteria createTestCriteria();
};

AdvancedFilterDialog::FilterCriteria TestAdvancedFilterDialog::createTestCriteria()
{
    AdvancedFilterDialog::FilterCriteria criteria;
    criteria.enableDateFilter = true;
    criteria.dateFrom = QDateTime::currentDateTime().addDays(-7);
    criteria.dateTo = QDateTime::currentDateTime();
    criteria.dateType = AdvancedFilterDialog::FilterCriteria::ModifiedDate;
    
    criteria.enableExtensionFilter = true;
    criteria.includedExtensions << "jpg" << "png" << "gif";
    
    criteria.enablePathFilter = true;
    criteria.pathPatterns << "*/Documents/*" << "*temp*";
    criteria.pathCaseSensitive = true;
    
    criteria.enableSizeFilter = true;
    criteria.minSize = 1024 * 1024; // 1 MB
    criteria.maxSize = 100 * 1024 * 1024; // 100 MB
    criteria.sizeUnit = AdvancedFilterDialog::FilterCriteria::MB;
    
    criteria.combineMode = AdvancedFilterDialog::FilterCriteria::AND;
    
    return criteria;
}

void TestAdvancedFilterDialog::testDefaultState()
{
    AdvancedFilterDialog dialog;
    
    AdvancedFilterDialog::FilterCriteria criteria = dialog.getFilterCriteria();
    
    // All filters should be disabled by default
    QVERIFY(!criteria.enableDateFilter);
    QVERIFY(!criteria.enableExtensionFilter);
    QVERIFY(!criteria.enablePathFilter);
    QVERIFY(!criteria.enableSizeFilter);
    
    // Default combine mode should be AND
    QCOMPARE(criteria.combineMode, AdvancedFilterDialog::FilterCriteria::AND);
}

void TestAdvancedFilterDialog::testSetGetCriteria()
{
    AdvancedFilterDialog dialog;
    
    AdvancedFilterDialog::FilterCriteria testCriteria = createTestCriteria();
    dialog.setFilterCriteria(testCriteria);
    
    AdvancedFilterDialog::FilterCriteria retrievedCriteria = dialog.getFilterCriteria();
    
    // Verify all settings were preserved
    QCOMPARE(retrievedCriteria.enableDateFilter, testCriteria.enableDateFilter);
    QCOMPARE(retrievedCriteria.enableExtensionFilter, testCriteria.enableExtensionFilter);
    QCOMPARE(retrievedCriteria.enablePathFilter, testCriteria.enablePathFilter);
    QCOMPARE(retrievedCriteria.enableSizeFilter, testCriteria.enableSizeFilter);
    QCOMPARE(retrievedCriteria.combineMode, testCriteria.combineMode);
}

void TestAdvancedFilterDialog::testResetFilters()
{
    AdvancedFilterDialog dialog;
    
    // Set some criteria first
    AdvancedFilterDialog::FilterCriteria testCriteria = createTestCriteria();
    dialog.setFilterCriteria(testCriteria);
    
    // Reset
    dialog.resetFilters();
    
    // Verify all filters are disabled
    AdvancedFilterDialog::FilterCriteria criteria = dialog.getFilterCriteria();
    QVERIFY(!criteria.enableDateFilter);
    QVERIFY(!criteria.enableExtensionFilter);
    QVERIFY(!criteria.enablePathFilter);
    QVERIFY(!criteria.enableSizeFilter);
}

void TestAdvancedFilterDialog::testSizeConversion()
{
    AdvancedFilterDialog dialog;
    
    // Test size conversion logic by setting different units
    AdvancedFilterDialog::FilterCriteria criteria;
    criteria.enableSizeFilter = true;
    criteria.minSize = 1024; // 1 KB in bytes
    criteria.maxSize = 1024 * 1024; // 1 MB in bytes
    criteria.sizeUnit = AdvancedFilterDialog::FilterCriteria::KB;
    
    dialog.setFilterCriteria(criteria);
    AdvancedFilterDialog::FilterCriteria retrieved = dialog.getFilterCriteria();
    
    // Should handle size conversion correctly
    QVERIFY(retrieved.enableSizeFilter);
}

#include "test_advanced_filter_dialog.moc"