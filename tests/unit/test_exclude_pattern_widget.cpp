#include <QtTest/QtTest>
#include <QtWidgets/QApplication>
#include "exclude_pattern_widget.h"

class TestExcludePatternWidget : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();
    
    // Pattern validation tests
    void testValidatePattern_ValidWildcard();
    void testValidatePattern_InvalidEmpty();
    void testValidatePattern_InvalidCharacters();
    void testValidatePattern_ValidRegex();
    void testValidatePattern_InvalidRegex();
    
    // Pattern management tests
    void testAddPattern_Valid();
    void testAddPattern_Duplicate();
    void testAddPattern_Empty();
    void testRemovePattern();
    void testClearPatterns();
    void testSetPatterns();
    void testGetPatterns();
    
    // Pattern matching tests
    void testMatchesAnyPattern_SinglePattern();
    void testMatchesAnyPattern_MultiplePatterns();
    void testMatchesAnyPattern_NoMatch();
    void testMatchesAnyPattern_CaseInsensitive();
    
    // Settings persistence tests
    void testSaveToSettings();
    void testLoadFromSettings();
    
    // Signal tests
    void testPatternsChangedSignal();
    void testPatternAddedSignal();
    void testPatternRemovedSignal();

private:
    ExcludePatternWidget* m_widget;
};

void TestExcludePatternWidget::initTestCase()
{
    // Initialize test case
}

void TestExcludePatternWidget::cleanupTestCase()
{
    // Cleanup test case
}

void TestExcludePatternWidget::init()
{
    m_widget = new ExcludePatternWidget();
}

void TestExcludePatternWidget::cleanup()
{
    delete m_widget;
    m_widget = nullptr;
}

// Pattern validation tests

void TestExcludePatternWidget::testValidatePattern_ValidWildcard()
{
    QString errorMessage;
    
    QVERIFY(ExcludePatternWidget::validatePattern("*.tmp", &errorMessage));
    QVERIFY(ExcludePatternWidget::validatePattern("*.log", &errorMessage));
    QVERIFY(ExcludePatternWidget::validatePattern("test*.txt", &errorMessage));
    QVERIFY(ExcludePatternWidget::validatePattern("file?.dat", &errorMessage));
    QVERIFY(ExcludePatternWidget::validatePattern("Thumbs.db", &errorMessage));
}

void TestExcludePatternWidget::testValidatePattern_InvalidEmpty()
{
    QString errorMessage;
    
    QVERIFY(!ExcludePatternWidget::validatePattern("", &errorMessage));
    QVERIFY(!errorMessage.isEmpty());
    QVERIFY(errorMessage.contains("empty"));
    
    QVERIFY(!ExcludePatternWidget::validatePattern("   ", &errorMessage));
}

void TestExcludePatternWidget::testValidatePattern_InvalidCharacters()
{
    QString errorMessage;
    
    // Test with invalid special characters
    QVERIFY(!ExcludePatternWidget::validatePattern("test@file", &errorMessage));
    QVERIFY(!ExcludePatternWidget::validatePattern("file#name", &errorMessage));
    QVERIFY(!ExcludePatternWidget::validatePattern("test&file", &errorMessage));
}

void TestExcludePatternWidget::testValidatePattern_ValidRegex()
{
    QString errorMessage;
    
    // Simple patterns that look like regex but are valid
    QVERIFY(ExcludePatternWidget::validatePattern("test-file.txt", &errorMessage));
    QVERIFY(ExcludePatternWidget::validatePattern("file_name.dat", &errorMessage));
}

void TestExcludePatternWidget::testValidatePattern_InvalidRegex()
{
    QString errorMessage;
    
    // Invalid regex patterns
    QVERIFY(!ExcludePatternWidget::validatePattern("[unclosed", &errorMessage));
    QVERIFY(!errorMessage.isEmpty());
}

// Pattern management tests

void TestExcludePatternWidget::testAddPattern_Valid()
{
    QVERIFY(m_widget->addPattern("*.tmp"));
    QCOMPARE(m_widget->getPatterns().size(), 1);
    QVERIFY(m_widget->getPatterns().contains("*.tmp"));
    
    QVERIFY(m_widget->addPattern("*.log"));
    QCOMPARE(m_widget->getPatterns().size(), 2);
}

void TestExcludePatternWidget::testAddPattern_Duplicate()
{
    QVERIFY(m_widget->addPattern("*.tmp"));
    QVERIFY(!m_widget->addPattern("*.tmp")); // Should fail - duplicate
    QCOMPARE(m_widget->getPatterns().size(), 1);
}

void TestExcludePatternWidget::testAddPattern_Empty()
{
    QVERIFY(!m_widget->addPattern(""));
    QVERIFY(!m_widget->addPattern("   "));
    QCOMPARE(m_widget->getPatterns().size(), 0);
}

void TestExcludePatternWidget::testRemovePattern()
{
    m_widget->addPattern("*.tmp");
    m_widget->addPattern("*.log");
    QCOMPARE(m_widget->getPatterns().size(), 2);
    
    m_widget->removePattern("*.tmp");
    QCOMPARE(m_widget->getPatterns().size(), 1);
    QVERIFY(!m_widget->getPatterns().contains("*.tmp"));
    QVERIFY(m_widget->getPatterns().contains("*.log"));
}

void TestExcludePatternWidget::testClearPatterns()
{
    m_widget->addPattern("*.tmp");
    m_widget->addPattern("*.log");
    m_widget->addPattern("*.bak");
    QCOMPARE(m_widget->getPatterns().size(), 3);
    
    m_widget->clearPatterns();
    QCOMPARE(m_widget->getPatterns().size(), 0);
}

void TestExcludePatternWidget::testSetPatterns()
{
    QStringList patterns = {"*.tmp", "*.log", "*.bak"};
    m_widget->setPatterns(patterns);
    
    QCOMPARE(m_widget->getPatterns().size(), 3);
    QVERIFY(m_widget->getPatterns().contains("*.tmp"));
    QVERIFY(m_widget->getPatterns().contains("*.log"));
    QVERIFY(m_widget->getPatterns().contains("*.bak"));
}

void TestExcludePatternWidget::testGetPatterns()
{
    QVERIFY(m_widget->getPatterns().isEmpty());
    
    m_widget->addPattern("*.tmp");
    m_widget->addPattern("*.log");
    
    QStringList patterns = m_widget->getPatterns();
    QCOMPARE(patterns.size(), 2);
    QVERIFY(patterns.contains("*.tmp"));
    QVERIFY(patterns.contains("*.log"));
}

// Pattern matching tests

void TestExcludePatternWidget::testMatchesAnyPattern_SinglePattern()
{
    m_widget->addPattern("*.tmp");
    
    QVERIFY(m_widget->matchesAnyPattern("test.tmp"));
    QVERIFY(m_widget->matchesAnyPattern("file.tmp"));
    QVERIFY(!m_widget->matchesAnyPattern("test.log"));
    QVERIFY(!m_widget->matchesAnyPattern("test.txt"));
}

void TestExcludePatternWidget::testMatchesAnyPattern_MultiplePatterns()
{
    m_widget->addPattern("*.tmp");
    m_widget->addPattern("*.log");
    m_widget->addPattern("Thumbs.db");
    
    QVERIFY(m_widget->matchesAnyPattern("test.tmp"));
    QVERIFY(m_widget->matchesAnyPattern("debug.log"));
    QVERIFY(m_widget->matchesAnyPattern("Thumbs.db"));
    QVERIFY(!m_widget->matchesAnyPattern("document.txt"));
}

void TestExcludePatternWidget::testMatchesAnyPattern_NoMatch()
{
    m_widget->addPattern("*.tmp");
    m_widget->addPattern("*.log");
    
    QVERIFY(!m_widget->matchesAnyPattern("document.txt"));
    QVERIFY(!m_widget->matchesAnyPattern("image.png"));
    QVERIFY(!m_widget->matchesAnyPattern("video.mp4"));
}

void TestExcludePatternWidget::testMatchesAnyPattern_CaseInsensitive()
{
    m_widget->addPattern("*.TMP");
    
    // Should match case-insensitively
    QVERIFY(m_widget->matchesAnyPattern("test.tmp"));
    QVERIFY(m_widget->matchesAnyPattern("test.TMP"));
    QVERIFY(m_widget->matchesAnyPattern("test.Tmp"));
}

// Settings persistence tests

void TestExcludePatternWidget::testSaveToSettings()
{
    QStringList patterns = {"*.tmp", "*.log", "*.bak"};
    m_widget->setPatterns(patterns);
    
    QString testKey = "test_exclude_patterns";
    m_widget->saveToSettings(testKey);
    
    QSettings settings;
    QStringList savedPatterns = settings.value(testKey).toStringList();
    
    QCOMPARE(savedPatterns.size(), 3);
    QVERIFY(savedPatterns.contains("*.tmp"));
    QVERIFY(savedPatterns.contains("*.log"));
    QVERIFY(savedPatterns.contains("*.bak"));
    
    // Cleanup
    settings.remove(testKey);
}

void TestExcludePatternWidget::testLoadFromSettings()
{
    QString testKey = "test_exclude_patterns_load";
    QStringList patterns = {"*.tmp", "*.log", "*.bak"};
    
    // Save patterns to settings
    QSettings settings;
    settings.setValue(testKey, patterns);
    
    // Load patterns into widget
    m_widget->loadFromSettings(testKey);
    
    QCOMPARE(m_widget->getPatterns().size(), 3);
    QVERIFY(m_widget->getPatterns().contains("*.tmp"));
    QVERIFY(m_widget->getPatterns().contains("*.log"));
    QVERIFY(m_widget->getPatterns().contains("*.bak"));
    
    // Cleanup
    settings.remove(testKey);
}

// Signal tests

void TestExcludePatternWidget::testPatternsChangedSignal()
{
    QSignalSpy spy(m_widget, &ExcludePatternWidget::patternsChanged);
    
    m_widget->addPattern("*.tmp");
    QCOMPARE(spy.count(), 1);
    
    m_widget->addPattern("*.log");
    QCOMPARE(spy.count(), 2);
    
    m_widget->removePattern("*.tmp");
    QCOMPARE(spy.count(), 3);
}

void TestExcludePatternWidget::testPatternAddedSignal()
{
    QSignalSpy spy(m_widget, &ExcludePatternWidget::patternAdded);
    
    m_widget->addPattern("*.tmp");
    QCOMPARE(spy.count(), 1);
    
    QList<QVariant> arguments = spy.takeFirst();
    QCOMPARE(arguments.at(0).toString(), QString("*.tmp"));
}

void TestExcludePatternWidget::testPatternRemovedSignal()
{
    QSignalSpy spy(m_widget, &ExcludePatternWidget::patternRemoved);
    
    m_widget->addPattern("*.tmp");
    m_widget->removePattern("*.tmp");
    
    QCOMPARE(spy.count(), 1);
    QList<QVariant> arguments = spy.takeFirst();
    QCOMPARE(arguments.at(0).toString(), QString("*.tmp"));
}

QTEST_MAIN(TestExcludePatternWidget)
#include "test_exclude_pattern_widget.moc"
