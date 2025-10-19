#pragma once

#include <QString>
#include <QStringList>
#include <QVariant>
#include <QPixmap>
#include <QWidget>
#include <QApplication>
#include <QTest>
#include <QSignalSpy>
#include <QEventLoop>
#include <QTimer>
#include <QElapsedTimer>
#include <functional>
#include <memory>

/**
 * @brief Common testing utilities and helper functions
 * 
 * This class provides a comprehensive set of utilities for testing including:
 * - Widget interaction helpers
 * - Timing and synchronization utilities
 * - Data generation and validation
 * - Screenshot and visual testing support
 * - Performance measurement tools
 */
class TestUtilities {
public:
    // Widget interaction utilities
    static QWidget* findWidget(QWidget* parent, const QString& objectName);
    static QWidget* findWidgetByText(QWidget* parent, const QString& text);
    static QWidget* findWidgetByType(QWidget* parent, const QString& typeName);
    static QList<QWidget*> findAllWidgets(QWidget* parent, const QString& objectName);
    
    // Widget state verification
    static bool isWidgetVisible(QWidget* widget);
    static bool isWidgetEnabled(QWidget* widget);
    static bool hasWidgetFocus(QWidget* widget);
    static QString getWidgetText(QWidget* widget);
    static QVariant getWidgetProperty(QWidget* widget, const QString& propertyName);
    
    // User interaction simulation
    static bool clickWidget(QWidget* widget, Qt::MouseButton button = Qt::LeftButton);
    static bool doubleClickWidget(QWidget* widget);
    static bool rightClickWidget(QWidget* widget);
    static bool typeText(QWidget* widget, const QString& text);
    static bool pressKey(QWidget* widget, Qt::Key key, Qt::KeyboardModifiers modifiers = Qt::NoModifier);
    static bool pressKeySequence(QWidget* widget, const QKeySequence& sequence);
    
    // Drag and drop simulation
    static bool dragAndDrop(QWidget* source, QWidget* target);
    static bool dragAndDrop(QWidget* source, const QPoint& sourcePos, 
                           QWidget* target, const QPoint& targetPos);
    
    // Waiting and synchronization
    static bool waitForWidget(QWidget* parent, const QString& objectName, int timeoutMs = 5000);
    static bool waitForCondition(std::function<bool()> condition, int timeoutMs = 5000);
    static bool waitForSignal(QObject* sender, const char* signal, int timeoutMs = 5000);
    static bool waitForSignalWithArgs(QSignalSpy& spy, int expectedCount = 1, int timeoutMs = 5000);
    static void processEvents(int maxTimeMs = 100);
    
    // Window and dialog utilities
    static QWidget* getActiveWindow();
    static QWidget* getModalDialog();
    static bool waitForWindow(const QString& title, int timeoutMs = 5000);
    static bool waitForDialog(const QString& title, int timeoutMs = 5000);
    static bool closeDialog(QWidget* dialog, bool accept = true);
    
    // Screenshot and visual utilities
    static QPixmap captureWidget(QWidget* widget);
    static QPixmap captureScreen();
    static QPixmap captureRegion(const QRect& region);
    static bool saveScreenshot(QWidget* widget, const QString& filePath);
    static bool compareImages(const QPixmap& img1, const QPixmap& img2, double threshold = 0.95);
    static double calculateImageSimilarity(const QPixmap& img1, const QPixmap& img2);
    static QPixmap generateDifferenceImage(const QPixmap& img1, const QPixmap& img2);
    
    // Performance measurement
    static void startPerformanceMeasurement(const QString& measurementName);
    static qint64 stopPerformanceMeasurement(const QString& measurementName);
    static qint64 getMemoryUsage();
    static double getCpuUsage();
    
    // Data generation utilities
    static QString generateRandomString(int length, bool alphaNumericOnly = true);
    static QStringList generateRandomStringList(int count, int minLength = 5, int maxLength = 15);
    static QByteArray generateRandomData(int sizeBytes);
    static QString generateUniqueId();
    static QString generateTimestamp();
    
    // File and directory utilities
    static QString createTempFile(const QString& content = QString(), const QString& suffix = ".tmp");
    static QString createTempDirectory();
    static bool createFileWithContent(const QString& filePath, const QString& content);
    static bool createFileWithSize(const QString& filePath, qint64 sizeBytes);
    static QString readFileContent(const QString& filePath);
    static bool compareFiles(const QString& file1, const QString& file2);
    static qint64 getFileSize(const QString& filePath);
    static QStringList listFiles(const QString& directory, const QStringList& filters = QStringList());
    
    // Configuration and settings utilities
    static void saveTestSettings(const QString& key, const QVariant& value);
    static QVariant loadTestSettings(const QString& key, const QVariant& defaultValue = QVariant());
    static void clearTestSettings();
    static void backupSettings(const QString& backupName);
    static void restoreSettings(const QString& backupName);
    
    // Application state utilities
    static void resetApplicationState();
    static void clearApplicationCache();
    static void setApplicationProperty(const QString& name, const QVariant& value);
    static QVariant getApplicationProperty(const QString& name);
    
    // Validation utilities
    static bool validateEmailAddress(const QString& email);
    static bool validateFilePath(const QString& path);
    static bool validateUrl(const QString& url);
    static bool validateRange(double value, double min, double max);
    static bool validateStringLength(const QString& str, int minLength, int maxLength);
    
    // Error simulation utilities
    static void simulateOutOfMemory();
    static void simulateDiskFull();
    static void simulateNetworkError();
    static void simulateSlowOperation(int delayMs);
    
    // Debugging utilities
    static void dumpWidgetHierarchy(QWidget* root, int indent = 0);
    static void dumpObjectProperties(QObject* object);
    static void logTestStep(const QString& step);
    static void logTestResult(const QString& testName, bool passed, const QString& message = QString());
    
    // Platform-specific utilities
    static QString getPlatformName();
    static bool isWindows();
    static bool isMacOS();
    static bool isLinux();
    static QString getSystemInfo();
    
private:
    // Internal helper methods
    static QWidget* findWidgetRecursive(QWidget* parent, const QString& objectName);
    static bool waitForConditionWithEvents(std::function<bool()> condition, int timeoutMs);
    static void processEventsFor(int milliseconds);
    
    // Static data for performance measurements
    static QMap<QString, QElapsedTimer> s_performanceTimers;
    static QMap<QString, qint64> s_performanceResults;
};

/**
 * @brief RAII helper for performance measurement
 */
class PerformanceMeasurement {
public:
    explicit PerformanceMeasurement(const QString& name) : m_name(name) {
        TestUtilities::startPerformanceMeasurement(m_name);
    }
    
    ~PerformanceMeasurement() {
        TestUtilities::stopPerformanceMeasurement(m_name);
    }
    
    qint64 elapsed() const {
        return TestUtilities::stopPerformanceMeasurement(m_name);
    }

private:
    QString m_name;
};

/**
 * @brief RAII helper for temporary file management
 */
class TempFileGuard {
public:
    explicit TempFileGuard(const QString& content = QString(), const QString& suffix = ".tmp") {
        m_filePath = TestUtilities::createTempFile(content, suffix);
    }
    
    ~TempFileGuard() {
        if (!m_filePath.isEmpty()) {
            QFile::remove(m_filePath);
        }
    }
    
    QString path() const { return m_filePath; }
    bool isValid() const { return !m_filePath.isEmpty(); }

private:
    QString m_filePath;
};

/**
 * @brief RAII helper for settings backup and restore
 */
class SettingsGuard {
public:
    explicit SettingsGuard(const QString& backupName = QString()) 
        : m_backupName(backupName.isEmpty() ? TestUtilities::generateUniqueId() : backupName) {
        TestUtilities::backupSettings(m_backupName);
    }
    
    ~SettingsGuard() {
        TestUtilities::restoreSettings(m_backupName);
    }

private:
    QString m_backupName;
};

/**
 * @brief Utility macros for common test operations
 */
#define FIND_WIDGET(parent, name) TestUtilities::findWidget(parent, name)
#define CLICK_WIDGET(widget) TestUtilities::clickWidget(widget)
#define TYPE_TEXT(widget, text) TestUtilities::typeText(widget, text)
#define WAIT_FOR_WIDGET(parent, name) TestUtilities::waitForWidget(parent, name)
#define WAIT_FOR_CONDITION(condition) TestUtilities::waitForCondition([&]() { return condition; })
#define CAPTURE_SCREENSHOT(widget, path) TestUtilities::saveScreenshot(widget, path)

#define PERFORMANCE_MEASURE(name) PerformanceMeasurement perf_##name(#name)
#define TEMP_FILE_GUARD(content) TempFileGuard tempFile(content)
#define SETTINGS_GUARD() SettingsGuard settingsGuard

#define LOG_TEST_STEP(step) TestUtilities::logTestStep(step)
#define LOG_TEST_RESULT(name, passed, message) TestUtilities::logTestResult(name, passed, message)

// Assertion macros with better error messages
#define TEST_ASSERT_WIDGET_EXISTS(parent, name) \
    do { \
        QWidget* widget = FIND_WIDGET(parent, name); \
        if (!widget) { \
            throw std::runtime_error(QString("Widget not found: %1").arg(name).toStdString()); \
        } \
    } while(0)

#define TEST_ASSERT_WIDGET_VISIBLE(widget) \
    do { \
        if (!TestUtilities::isWidgetVisible(widget)) { \
            throw std::runtime_error(QString("Widget is not visible: %1").arg(widget->objectName()).toStdString()); \
        } \
    } while(0)

#define TEST_ASSERT_WIDGET_ENABLED(widget) \
    do { \
        if (!TestUtilities::isWidgetEnabled(widget)) { \
            throw std::runtime_error(QString("Widget is not enabled: %1").arg(widget->objectName()).toStdString()); \
        } \
    } while(0)

#define TEST_ASSERT_WIDGET_TEXT(widget, expectedText) \
    do { \
        QString actualText = TestUtilities::getWidgetText(widget); \
        if (actualText != expectedText) { \
            throw std::runtime_error(QString("Widget text mismatch. Expected: '%1', Actual: '%2'") \
                                    .arg(expectedText).arg(actualText).toStdString()); \
        } \
    } while(0)

#define TEST_ASSERT_FILES_EQUAL(file1, file2) \
    do { \
        if (!TestUtilities::compareFiles(file1, file2)) { \
            throw std::runtime_error(QString("Files are not equal: %1 vs %2").arg(file1).arg(file2).toStdString()); \
        } \
    } while(0)

#define TEST_ASSERT_PERFORMANCE(name, maxTimeMs) \
    do { \
        qint64 elapsed = TestUtilities::stopPerformanceMeasurement(name); \
        if (elapsed > maxTimeMs) { \
            throw std::runtime_error(QString("Performance test failed: %1 took %2ms (max: %3ms)") \
                                    .arg(name).arg(elapsed).arg(maxTimeMs).toStdString()); \
        } \
    } while(0)