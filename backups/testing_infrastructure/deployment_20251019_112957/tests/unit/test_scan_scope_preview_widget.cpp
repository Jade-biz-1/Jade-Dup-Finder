#include <QtTest/QtTest>
#include <QtWidgets/QApplication>
#include <QtCore/QDir>
#include <QtCore/QFile>
#include <QtCore/QTemporaryDir>
#include "scan_scope_preview_widget.h"

class ScanScopePreviewWidgetTest : public QObject
{
    Q_OBJECT

private:
    ScanScopePreviewWidget* widget = nullptr;
    QTemporaryDir* tempDir = nullptr;
    QString testPath;
    
    void createTestFile(const QString& path, qint64 size)
    {
        QFile file(path);
        if (file.open(QIODevice::WriteOnly)) {
            QByteArray data(static_cast<int>(size), 'x');
            file.write(data);
            file.close();
        }
    }
    
    void waitForCalculation()
    {
        // Wait for debounce timer and calculation
        QTest::qWait(600); // UPDATE_DELAY_MS + buffer
        QApplication::processEvents();
    }

private slots:
    void init()
    {
        widget = new ScanScopePreviewWidget();
        
        tempDir = new QTemporaryDir();
        QVERIFY(tempDir->isValid());
        
        testPath = tempDir->path();
        
        QDir dir(testPath);
        dir.mkdir("folder1");
        dir.mkdir("folder1/subfolder1");
        dir.mkdir("folder2");
        dir.mkdir(".hidden");
        
        createTestFile(testPath + "/folder1/file1.txt", 1024);
        createTestFile(testPath + "/folder1/file2.txt", 2048);
        createTestFile(testPath + "/folder1/subfolder1/file3.txt", 512);
        createTestFile(testPath + "/folder2/file4.txt", 4096);
        createTestFile(testPath + "/.hidden/file5.txt", 256);
    }
    
    void cleanup()
    {
        delete widget;
        widget = nullptr;
        delete tempDir;
        tempDir = nullptr;
    }
    
    void testInitialState()
    {
        auto stats = widget->getCurrentStats();
        
        QCOMPARE(stats.folderCount, 0);
        QCOMPARE(stats.estimatedFileCount, 0);
        QCOMPARE(stats.estimatedSize, 0);
        QVERIFY(stats.includedPaths.isEmpty());
        QVERIFY(stats.excludedPaths.isEmpty());
        QVERIFY(!stats.calculationComplete);
    }
    
    void testBasicFolderCounting()
    {
        QSignalSpy spy(widget, &ScanScopePreviewWidget::previewUpdated);
        
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, -1, false);
        waitForCalculation();
        
        QCOMPARE(spy.count(), 1);
        
        auto stats = widget->getCurrentStats();
        QVERIFY(stats.calculationComplete);
        QVERIFY(stats.folderCount > 0);
        QCOMPARE(stats.includedPaths.size(), 1);
        QCOMPARE(stats.includedPaths.first(), testPath);
    }
    
    void testFileCountEstimation()
    {
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, -1, false);
        waitForCalculation();
        
        auto stats = widget->getCurrentStats();
        
        QVERIFY(stats.estimatedFileCount >= 4);
        QVERIFY(stats.estimatedSize > 0);
    }
    
    void testIncludeHiddenFiles()
    {
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, -1, false);
        waitForCalculation();
        auto statsWithoutHidden = widget->getCurrentStats();
        
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, -1, true);
        waitForCalculation();
        auto statsWithHidden = widget->getCurrentStats();
        
        QVERIFY(statsWithHidden.estimatedFileCount >= statsWithoutHidden.estimatedFileCount);
    }
    
    void testExcludePatterns()
    {
        widget->updatePreview(QStringList{testPath}, QStringList{"*.txt"}, QStringList{}, -1, false);
        waitForCalculation();
        
        auto stats = widget->getCurrentStats();
        
        QCOMPARE(stats.estimatedFileCount, 0);
    }
    
    void testExcludeFolders()
    {
        QString excludePath = testPath + "/folder1";
        
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{excludePath}, -1, false);
        waitForCalculation();
        
        auto stats = widget->getCurrentStats();
        
        QVERIFY(stats.estimatedFileCount < 4);
        QCOMPARE(stats.excludedPaths.size(), 1);
        QCOMPARE(stats.excludedPaths.first(), excludePath);
    }
    
    void testMaxDepthLimit()
    {
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, 0, false);
        waitForCalculation();
        auto statsDepth0 = widget->getCurrentStats();
        
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, 1, false);
        waitForCalculation();
        auto statsDepth1 = widget->getCurrentStats();
        
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, -1, false);
        waitForCalculation();
        auto statsUnlimited = widget->getCurrentStats();
        
        QVERIFY(statsDepth0.estimatedFileCount <= statsDepth1.estimatedFileCount);
        QVERIFY(statsDepth1.estimatedFileCount <= statsUnlimited.estimatedFileCount);
    }
    
    void testMultiplePaths()
    {
        QString path1 = testPath + "/folder1";
        QString path2 = testPath + "/folder2";
        
        widget->updatePreview(QStringList{path1, path2}, QStringList{}, QStringList{}, -1, false);
        waitForCalculation();
        
        auto stats = widget->getCurrentStats();
        
        QCOMPARE(stats.includedPaths.size(), 2);
        QVERIFY(stats.estimatedFileCount > 0);
    }
    
    void testEmptyPathList()
    {
        widget->updatePreview(QStringList{}, QStringList{}, QStringList{}, -1, false);
        waitForCalculation();
        
        auto stats = widget->getCurrentStats();
        
        QVERIFY(stats.calculationComplete);
        QCOMPARE(stats.folderCount, 0);
        QCOMPARE(stats.estimatedFileCount, 0);
        QVERIFY(!stats.errorMessage.isEmpty());
    }
    
    void testNonExistentPath()
    {
        QString fakePath = "/this/path/does/not/exist";
        
        widget->updatePreview(QStringList{fakePath}, QStringList{}, QStringList{}, -1, false);
        waitForCalculation();
        
        auto stats = widget->getCurrentStats();
        
        QVERIFY(stats.calculationComplete);
        QCOMPARE(stats.folderCount, 0);
        QCOMPARE(stats.estimatedFileCount, 0);
    }
    
    void testDebouncedUpdates()
    {
        QSignalSpy spy(widget, &ScanScopePreviewWidget::previewUpdated);
        
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, -1, false);
        QTest::qWait(100);
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, -1, false);
        QTest::qWait(100);
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, -1, false);
        
        waitForCalculation();
        
        QCOMPARE(spy.count(), 1);
    }
    
    void testSignalEmission()
    {
        QSignalSpy startSpy(widget, &ScanScopePreviewWidget::calculationStarted);
        QSignalSpy finishSpy(widget, &ScanScopePreviewWidget::calculationFinished);
        QSignalSpy updateSpy(widget, &ScanScopePreviewWidget::previewUpdated);
        
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, -1, false);
        waitForCalculation();
        
        QCOMPARE(startSpy.count(), 1);
        QCOMPARE(finishSpy.count(), 1);
        QCOMPARE(updateSpy.count(), 1);
    }
    
    void testClearFunction()
    {
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, -1, false);
        waitForCalculation();
        
        auto statsBefore = widget->getCurrentStats();
        QVERIFY(statsBefore.estimatedFileCount > 0);
        
        widget->clear();
        
        auto statsAfter = widget->getCurrentStats();
        QCOMPARE(statsAfter.folderCount, 0);
        QCOMPARE(statsAfter.estimatedFileCount, 0);
        QCOMPARE(statsAfter.estimatedSize, 0);
    }
    
    void testSizeEstimation()
    {
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, -1, false);
        waitForCalculation();
        
        auto stats = widget->getCurrentStats();
        
        QVERIFY(stats.estimatedSize > 7000);
        QVERIFY(stats.estimatedSize < 10000);
    }
    
    void testPatternMatching()
    {
        widget->updatePreview(QStringList{testPath}, QStringList{"file1.*"}, QStringList{}, -1, false);
        waitForCalculation();
        auto stats1 = widget->getCurrentStats();
        
        widget->updatePreview(QStringList{testPath}, QStringList{"*.txt"}, QStringList{}, -1, false);
        waitForCalculation();
        auto stats2 = widget->getCurrentStats();
        
        QVERIFY(stats1.estimatedFileCount < 4);
        QCOMPARE(stats2.estimatedFileCount, 0);
    }
    
    void testUpdateAfterClear()
    {
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, -1, false);
        waitForCalculation();
        
        widget->clear();
        
        widget->updatePreview(QStringList{testPath}, QStringList{}, QStringList{}, -1, false);
        waitForCalculation();
        
        auto stats = widget->getCurrentStats();
        QVERIFY(stats.estimatedFileCount > 0);
        QVERIFY(stats.calculationComplete);
    }
};

QTEST_MAIN(ScanScopePreviewWidgetTest)
#include "test_scan_scope_preview_widget.moc"
