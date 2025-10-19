#include <QtTest/QtTest>
#include <QApplication>
#include <QTemporaryDir>
#include <QTextStream>
#include <QFile>
#include "results_window.h"

class TestHTMLExport : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void testHTMLExportBasic();
    void testHTMLExportWithThumbnails();
    void testThumbnailGeneration();
    void testHTMLStructure();

private:
    ResultsWindow* m_resultsWindow;
    QTemporaryDir* m_tempDir;
    ResultsWindow::ScanResults createTestResults();
};

void TestHTMLExport::initTestCase()
{
    m_resultsWindow = new ResultsWindow();
    m_tempDir = new QTemporaryDir();
    QVERIFY(m_tempDir->isValid());
}

void TestHTMLExport::cleanupTestCase()
{
    delete m_resultsWindow;
    delete m_tempDir;
}

void TestHTMLExport::testHTMLExportBasic()
{
    // Set up test results
    auto testResults = createTestResults();
    m_resultsWindow->displayResults(testResults);
    
    // Create temporary HTML file
    QString htmlPath = m_tempDir->path() + "/test_export.html";
    QFile htmlFile(htmlPath);
    QVERIFY(htmlFile.open(QIODevice::WriteOnly | QIODevice::Text));
    
    QTextStream out(&htmlFile);
    bool success = m_resultsWindow->exportToHTML(out, htmlPath);
    htmlFile.close();
    
    QVERIFY(success);
    QVERIFY(QFile::exists(htmlPath));
    
    // Verify HTML content
    QFile readFile(htmlPath);
    QVERIFY(readFile.open(QIODevice::ReadOnly | QIODevice::Text));
    QString content = readFile.readAll();
    readFile.close();
    
    QVERIFY(content.contains("<!DOCTYPE html>"));
    QVERIFY(content.contains("Duplicate Files Report"));
    QVERIFY(content.contains("Duplicate Groups"));
    QVERIFY(content.contains("</html>"));
}

void TestHTMLExport::testHTMLExportWithThumbnails()
{
    // This test would require actual image files
    // For now, just test that the method doesn't crash
    QString thumbnailDir = m_tempDir->path() + "/thumbnails";
    QString result = m_resultsWindow->generateThumbnailForExport(
        "/non/existent/image.jpg", thumbnailDir, "test");
    
    // Should return empty string for non-existent file
    QVERIFY(result.isEmpty());
}

void TestHTMLExport::testThumbnailGeneration()
{
    QString thumbnailDir = m_tempDir->path() + "/thumbnails";
    QDir().mkpath(thumbnailDir);
    
    // Test with non-image file
    QString result = m_resultsWindow->generateThumbnailForExport(
        "/test/document.txt", thumbnailDir, "test");
    QVERIFY(result.isEmpty());
    
    // Test with non-existent image file
    result = m_resultsWindow->generateThumbnailForExport(
        "/test/image.jpg", thumbnailDir, "test");
    QVERIFY(result.isEmpty());
}

void TestHTMLExport::testHTMLStructure()
{
    auto testResults = createTestResults();
    m_resultsWindow->displayResults(testResults);
    
    QString htmlPath = m_tempDir->path() + "/structure_test.html";
    QFile htmlFile(htmlPath);
    QVERIFY(htmlFile.open(QIODevice::WriteOnly | QIODevice::Text));
    
    QTextStream out(&htmlFile);
    bool success = m_resultsWindow->exportToHTML(out, htmlPath);
    htmlFile.close();
    
    QVERIFY(success);
    
    // Read and verify HTML structure
    QFile readFile(htmlPath);
    QVERIFY(readFile.open(QIODevice::ReadOnly | QIODevice::Text));
    QString content = readFile.readAll();
    readFile.close();
    
    // Check for required HTML elements
    QVERIFY(content.contains("<head>"));
    QVERIFY(content.contains("<style>"));
    QVERIFY(content.contains("<body>"));
    QVERIFY(content.contains("class=\"container\""));
    QVERIFY(content.contains("class=\"summary\""));
    QVERIFY(content.contains("class=\"group\""));
    QVERIFY(content.contains("class=\"file-item\""));
    
    // Check for CSS classes
    QVERIFY(content.contains(".file-thumbnail"));
    QVERIFY(content.contains(".file-info"));
    QVERIFY(content.contains(".group-header"));
}

ResultsWindow::ScanResults TestHTMLExport::createTestResults()
{
    ResultsWindow::ScanResults results;
    
    // Create test group 1
    ResultsWindow::DuplicateGroup group1;
    group1.groupId = "group1";
    group1.totalSize = 2048;
    group1.fileCount = 2;
    group1.primaryFile = "/test/file1.txt";
    
    ResultsWindow::DuplicateFile file1;
    file1.filePath = "/test/file1.txt";
    file1.fileName = "file1.txt";
    file1.fileSize = 1024;
    file1.lastModified = QDateTime::currentDateTime();
    file1.fileType = "txt";
    
    ResultsWindow::DuplicateFile file2;
    file2.filePath = "/test/copy_of_file1.txt";
    file2.fileName = "copy_of_file1.txt";
    file2.fileSize = 1024;
    file2.lastModified = QDateTime::currentDateTime();
    file2.fileType = "txt";
    
    group1.files << file1 << file2;
    
    // Create test group 2
    ResultsWindow::DuplicateGroup group2;
    group2.groupId = "group2";
    group2.totalSize = 3072;
    group2.fileCount = 3;
    group2.primaryFile = "/test/image1.jpg";
    
    ResultsWindow::DuplicateFile file3;
    file3.filePath = "/test/image1.jpg";
    file3.fileName = "image1.jpg";
    file3.fileSize = 1024;
    file3.lastModified = QDateTime::currentDateTime();
    file3.fileType = "jpg";
    
    ResultsWindow::DuplicateFile file4;
    file4.filePath = "/test/image1_copy.jpg";
    file4.fileName = "image1_copy.jpg";
    file4.fileSize = 1024;
    file4.lastModified = QDateTime::currentDateTime();
    file4.fileType = "jpg";
    
    ResultsWindow::DuplicateFile file5;
    file5.filePath = "/test/image1_backup.jpg";
    file5.fileName = "image1_backup.jpg";
    file5.fileSize = 1024;
    file5.lastModified = QDateTime::currentDateTime();
    file5.fileType = "jpg";
    
    group2.files << file3 << file4 << file5;
    
    results.duplicateGroups << group1 << group2;
    results.totalDuplicatesFound = 5;
    results.potentialSavings = 3072; // 3 files can be deleted
    results.scanTime = QDateTime::currentDateTime();
    
    return results;
}

QTEST_MAIN(TestHTMLExport)
#include "test_html_export.moc"