#include "test_utilities.h"
#include <QApplication>
#include <QWidget>
#include <QTest>
#include <QEventLoop>
#include <QTimer>
#include <QPixmap>
#include <QScreen>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QSettings>
#include <QStandardPaths>
#include <QRandomGenerator>
#include <QCryptographicHash>
#include <QProcess>
#include <QThread>
#include <QDebug>
#include <QMetaObject>
#include <QMetaProperty>

// Static member initialization
QMap<QString, QElapsedTimer> TestUtilities::s_performanceTimers;
QMap<QString, qint64> TestUtilities::s_performanceResults;

// Widget interaction utilities
QWidget* TestUtilities::findWidget(QWidget* parent, const QString& objectName) {
    if (!parent) {
        return nullptr;
    }
    
    return parent->findChild<QWidget*>(objectName);
}

QWidget* TestUtilities::findWidgetByText(QWidget* parent, const QString& text) {
    if (!parent) {
        return nullptr;
    }
    
    // Check if this widget has the text
    if (parent->property("text").toString() == text) {
        return parent;
    }
    
    // Search children recursively
    const QObjectList& children = parent->children();
    for (QObject* child : children) {
        if (QWidget* widget = qobject_cast<QWidget*>(child)) {
            if (QWidget* found = findWidgetByText(widget, text)) {
                return found;
            }
        }
    }
    
    return nullptr;
}

QWidget* TestUtilities::findWidgetByType(QWidget* parent, const QString& typeName) {
    if (!parent) {
        return nullptr;
    }
    
    // Check if this widget matches the type
    if (parent->metaObject()->className() == typeName) {
        return parent;
    }
    
    // Search children recursively
    const QObjectList& children = parent->children();
    for (QObject* child : children) {
        if (QWidget* widget = qobject_cast<QWidget*>(child)) {
            if (QWidget* found = findWidgetByType(widget, typeName)) {
                return found;
            }
        }
    }
    
    return nullptr;
}

QList<QWidget*> TestUtilities::findAllWidgets(QWidget* parent, const QString& objectName) {
    QList<QWidget*> widgets;
    if (!parent) {
        return widgets;
    }
    
    return parent->findChildren<QWidget*>(objectName);
}

// Widget state verification
bool TestUtilities::isWidgetVisible(QWidget* widget) {
    return widget && widget->isVisible();
}

bool TestUtilities::isWidgetEnabled(QWidget* widget) {
    return widget && widget->isEnabled();
}

bool TestUtilities::hasWidgetFocus(QWidget* widget) {
    return widget && widget->hasFocus();
}

QString TestUtilities::getWidgetText(QWidget* widget) {
    if (!widget) {
        return QString();
    }
    
    // Try common text properties
    QVariant text = widget->property("text");
    if (text.isValid()) {
        return text.toString();
    }
    
    // Try windowTitle for windows/dialogs
    if (!widget->windowTitle().isEmpty()) {
        return widget->windowTitle();
    }
    
    return QString();
}

QVariant TestUtilities::getWidgetProperty(QWidget* widget, const QString& propertyName) {
    if (!widget) {
        return QVariant();
    }
    
    return widget->property(propertyName.toUtf8().constData());
}

// User interaction simulation
bool TestUtilities::clickWidget(QWidget* widget, Qt::MouseButton button) {
    if (!widget || !widget->isVisible() || !widget->isEnabled()) {
        return false;
    }
    
    QTest::mouseClick(widget, button);
    processEvents();
    return true;
}

bool TestUtilities::doubleClickWidget(QWidget* widget) {
    if (!widget || !widget->isVisible() || !widget->isEnabled()) {
        return false;
    }
    
    QTest::mouseDClick(widget, Qt::LeftButton);
    processEvents();
    return true;
}

bool TestUtilities::rightClickWidget(QWidget* widget) {
    return clickWidget(widget, Qt::RightButton);
}

bool TestUtilities::typeText(QWidget* widget, const QString& text) {
    if (!widget || !widget->isVisible() || !widget->isEnabled()) {
        return false;
    }
    
    widget->setFocus();
    processEvents(50);
    
    QTest::keyClicks(widget, text);
    processEvents();
    return true;
}

bool TestUtilities::pressKey(QWidget* widget, Qt::Key key, Qt::KeyboardModifiers modifiers) {
    if (!widget || !widget->isVisible() || !widget->isEnabled()) {
        return false;
    }
    
    QTest::keyClick(widget, key, modifiers);
    processEvents();
    return true;
}

bool TestUtilities::pressKeySequence(QWidget* widget, const QKeySequence& sequence) {
    if (!widget || !widget->isVisible() || !widget->isEnabled()) {
        return false;
    }
    
    // Convert key sequence to individual key presses
    for (int i = 0; i < sequence.count(); ++i) {
        QKeyCombination combination = sequence[i];
        Qt::Key key = combination.key();
        Qt::KeyboardModifiers modifiers = combination.keyboardModifiers();
        
        QTest::keyClick(widget, key, modifiers);
        processEvents(10);
    }
    
    return true;
}

// Waiting and synchronization
bool TestUtilities::waitForWidget(QWidget* parent, const QString& objectName, int timeoutMs) {
    return waitForCondition([parent, objectName]() {
        return findWidget(parent, objectName) != nullptr;
    }, timeoutMs);
}

bool TestUtilities::waitForCondition(std::function<bool()> condition, int timeoutMs) {
    QElapsedTimer timer;
    timer.start();
    
    while (timer.elapsed() < timeoutMs) {
        if (condition()) {
            return true;
        }
        
        processEvents(10);
        QThread::msleep(10);
    }
    
    return false;
}

bool TestUtilities::waitForSignal(QObject* sender, const char* signal, int timeoutMs) {
    if (!sender) {
        return false;
    }
    
    QEventLoop loop;
    QTimer timer;
    timer.setSingleShot(true);
    
    QObject::connect(&timer, &QTimer::timeout, &loop, &QEventLoop::quit);
    QObject::connect(sender, signal, &loop, SLOT(quit()));
    
    timer.start(timeoutMs);
    loop.exec();
    
    return timer.isActive(); // Returns true if signal was received before timeout
}

bool TestUtilities::waitForSignalWithArgs(QSignalSpy& spy, int expectedCount, int timeoutMs) {
    return spy.wait(timeoutMs) && spy.count() >= expectedCount;
}

void TestUtilities::processEvents(int maxTimeMs) {
    QElapsedTimer timer;
    timer.start();
    
    while (timer.elapsed() < maxTimeMs) {
        QApplication::processEvents();
        QThread::msleep(1);
    }
}

// Window and dialog utilities
QWidget* TestUtilities::getActiveWindow() {
    return QApplication::activeWindow();
}

QWidget* TestUtilities::getModalDialog() {
    return QApplication::activeModalWidget();
}

bool TestUtilities::waitForWindow(const QString& title, int timeoutMs) {
    return waitForCondition([title]() {
        QWidget* window = getActiveWindow();
        return window && window->windowTitle() == title;
    }, timeoutMs);
}

bool TestUtilities::waitForDialog(const QString& title, int timeoutMs) {
    return waitForCondition([title]() {
        QWidget* dialog = getModalDialog();
        return dialog && dialog->windowTitle() == title;
    }, timeoutMs);
}

bool TestUtilities::closeDialog(QWidget* dialog, bool accept) {
    if (!dialog) {
        return false;
    }
    
    if (accept) {
        QTest::keyClick(dialog, Qt::Key_Return);
    } else {
        QTest::keyClick(dialog, Qt::Key_Escape);
    }
    
    processEvents();
    return true;
}

// Screenshot and visual utilities
QPixmap TestUtilities::captureWidget(QWidget* widget) {
    if (!widget) {
        return QPixmap();
    }
    
    return widget->grab();
}

QPixmap TestUtilities::captureScreen() {
    QScreen* screen = QApplication::primaryScreen();
    if (!screen) {
        return QPixmap();
    }
    
    return screen->grabWindow(0);
}

QPixmap TestUtilities::captureRegion(const QRect& region) {
    QScreen* screen = QApplication::primaryScreen();
    if (!screen) {
        return QPixmap();
    }
    
    return screen->grabWindow(0, region.x(), region.y(), region.width(), region.height());
}

bool TestUtilities::saveScreenshot(QWidget* widget, const QString& filePath) {
    QPixmap screenshot = captureWidget(widget);
    if (screenshot.isNull()) {
        return false;
    }
    
    // Ensure directory exists
    QFileInfo fileInfo(filePath);
    QDir().mkpath(fileInfo.absolutePath());
    
    return screenshot.save(filePath);
}

bool TestUtilities::compareImages(const QPixmap& img1, const QPixmap& img2, double threshold) {
    if (img1.size() != img2.size()) {
        return false;
    }
    
    double similarity = calculateImageSimilarity(img1, img2);
    return similarity >= threshold;
}

double TestUtilities::calculateImageSimilarity(const QPixmap& img1, const QPixmap& img2) {
    if (img1.size() != img2.size()) {
        return 0.0;
    }
    
    QImage image1 = img1.toImage();
    QImage image2 = img2.toImage();
    
    if (image1.format() != image2.format()) {
        image2 = image2.convertToFormat(image1.format());
    }
    
    int width = image1.width();
    int height = image1.height();
    int totalPixels = width * height;
    int matchingPixels = 0;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (image1.pixel(x, y) == image2.pixel(x, y)) {
                matchingPixels++;
            }
        }
    }
    
    return (double)matchingPixels / totalPixels;
}

// Performance measurement
void TestUtilities::startPerformanceMeasurement(const QString& measurementName) {
    s_performanceTimers[measurementName].start();
}

qint64 TestUtilities::stopPerformanceMeasurement(const QString& measurementName) {
    if (!s_performanceTimers.contains(measurementName)) {
        return -1;
    }
    
    qint64 elapsed = s_performanceTimers[measurementName].elapsed();
    s_performanceResults[measurementName] = elapsed;
    s_performanceTimers.remove(measurementName);
    
    return elapsed;
}

qint64 TestUtilities::getMemoryUsage() {
    // Platform-specific memory usage implementation would go here
    // For now, return a placeholder value
    return 0;
}

double TestUtilities::getCpuUsage() {
    // Platform-specific CPU usage implementation would go here
    // For now, return a placeholder value
    return 0.0;
}

// Data generation utilities
QString TestUtilities::generateRandomString(int length, bool alphaNumericOnly) {
    const QString chars = alphaNumericOnly ? 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" :
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?";
    
    QString result;
    result.reserve(length);
    
    for (int i = 0; i < length; ++i) {
        result.append(chars[QRandomGenerator::global()->bounded(chars.length())]);
    }
    
    return result;
}

QStringList TestUtilities::generateRandomStringList(int count, int minLength, int maxLength) {
    QStringList result;
    result.reserve(count);
    
    for (int i = 0; i < count; ++i) {
        int length = minLength + QRandomGenerator::global()->bounded(maxLength - minLength + 1);
        result.append(generateRandomString(length));
    }
    
    return result;
}

QByteArray TestUtilities::generateRandomData(int sizeBytes) {
    QByteArray data;
    data.reserve(sizeBytes);
    
    for (int i = 0; i < sizeBytes; ++i) {
        data.append(static_cast<char>(QRandomGenerator::global()->bounded(256)));
    }
    
    return data;
}

QString TestUtilities::generateUniqueId() {
    QString timestamp = QString::number(QDateTime::currentMSecsSinceEpoch());
    QString random = generateRandomString(8);
    return timestamp + "_" + random;
}

QString TestUtilities::generateTimestamp() {
    return QDateTime::currentDateTime().toString("yyyy-MM-dd_hh-mm-ss-zzz");
}

// File and directory utilities
QString TestUtilities::createTempFile(const QString& content, const QString& suffix) {
    QString tempDir = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    QString fileName = generateUniqueId() + suffix;
    QString filePath = tempDir + "/" + fileName;
    
    if (createFileWithContent(filePath, content)) {
        return filePath;
    }
    
    return QString();
}

QString TestUtilities::createTempDirectory() {
    QString tempDir = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    QString dirName = generateUniqueId();
    QString dirPath = tempDir + "/" + dirName;
    
    if (QDir().mkpath(dirPath)) {
        return dirPath;
    }
    
    return QString();
}

bool TestUtilities::createFileWithContent(const QString& filePath, const QString& content) {
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }
    
    QTextStream stream(&file);
    stream << content;
    return true;
}

bool TestUtilities::createFileWithSize(const QString& filePath, qint64 sizeBytes) {
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly)) {
        return false;
    }
    
    // Write data in chunks to avoid memory issues with large files
    const int chunkSize = 1024 * 1024; // 1MB chunks
    QByteArray chunk(chunkSize, 'X');
    
    qint64 remaining = sizeBytes;
    while (remaining > 0) {
        int writeSize = qMin(remaining, (qint64)chunkSize);
        if (file.write(chunk.left(writeSize)) != writeSize) {
            return false;
        }
        remaining -= writeSize;
    }
    
    return true;
}

QString TestUtilities::readFileContent(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return QString();
    }
    
    QTextStream stream(&file);
    return stream.readAll();
}

bool TestUtilities::compareFiles(const QString& file1, const QString& file2) {
    QFile f1(file1), f2(file2);
    
    if (!f1.open(QIODevice::ReadOnly) || !f2.open(QIODevice::ReadOnly)) {
        return false;
    }
    
    if (f1.size() != f2.size()) {
        return false;
    }
    
    const int bufferSize = 8192;
    while (!f1.atEnd() && !f2.atEnd()) {
        QByteArray data1 = f1.read(bufferSize);
        QByteArray data2 = f2.read(bufferSize);
        
        if (data1 != data2) {
            return false;
        }
    }
    
    return f1.atEnd() && f2.atEnd();
}

qint64 TestUtilities::getFileSize(const QString& filePath) {
    QFileInfo info(filePath);
    return info.exists() ? info.size() : -1;
}

QStringList TestUtilities::listFiles(const QString& directory, const QStringList& filters) {
    QDir dir(directory);
    if (!dir.exists()) {
        return QStringList();
    }
    
    QDir::Filters dirFilters = QDir::Files | QDir::NoDotAndDotDot;
    return dir.entryList(filters, dirFilters);
}

// Configuration and settings utilities
void TestUtilities::saveTestSettings(const QString& key, const QVariant& value) {
    QSettings settings("DupFinderTests", "TestUtilities");
    settings.setValue(key, value);
}

QVariant TestUtilities::loadTestSettings(const QString& key, const QVariant& defaultValue) {
    QSettings settings("DupFinderTests", "TestUtilities");
    return settings.value(key, defaultValue);
}

void TestUtilities::clearTestSettings() {
    QSettings settings("DupFinderTests", "TestUtilities");
    settings.clear();
}

void TestUtilities::backupSettings(const QString& backupName) {
    QSettings settings("DupFinderTests", "TestUtilities");
    QSettings backup("DupFinderTests", "TestUtilities_" + backupName);
    
    // Copy all settings to backup
    for (const QString& key : settings.allKeys()) {
        backup.setValue(key, settings.value(key));
    }
}

void TestUtilities::restoreSettings(const QString& backupName) {
    QSettings settings("DupFinderTests", "TestUtilities");
    QSettings backup("DupFinderTests", "TestUtilities_" + backupName);
    
    // Clear current settings
    settings.clear();
    
    // Restore from backup
    for (const QString& key : backup.allKeys()) {
        settings.setValue(key, backup.value(key));
    }
}

// Platform-specific utilities
QString TestUtilities::getPlatformName() {
#ifdef Q_OS_WIN
    return "Windows";
#elif defined(Q_OS_MAC)
    return "macOS";
#elif defined(Q_OS_LINUX)
    return "Linux";
#else
    return "Unknown";
#endif
}

bool TestUtilities::isWindows() {
#ifdef Q_OS_WIN
    return true;
#else
    return false;
#endif
}

bool TestUtilities::isMacOS() {
#ifdef Q_OS_MAC
    return true;
#else
    return false;
#endif
}

bool TestUtilities::isLinux() {
#ifdef Q_OS_LINUX
    return true;
#else
    return false;
#endif
}

QString TestUtilities::getSystemInfo() {
    QString info;
    info += "Platform: " + getPlatformName() + "\n";
    info += "Qt Version: " + QString(QT_VERSION_STR) + "\n";
    info += "Application: " + QCoreApplication::applicationName() + "\n";
    info += "Version: " + QCoreApplication::applicationVersion() + "\n";
    return info;
}

// Debugging utilities
void TestUtilities::dumpWidgetHierarchy(QWidget* root, int indent) {
    if (!root) {
        return;
    }
    
    QString indentStr(indent * 2, ' ');
    qDebug() << qPrintable(indentStr + root->metaObject()->className() + 
                          " (" + root->objectName() + ")");
    
    const QObjectList& children = root->children();
    for (QObject* child : children) {
        if (QWidget* widget = qobject_cast<QWidget*>(child)) {
            dumpWidgetHierarchy(widget, indent + 1);
        }
    }
}

void TestUtilities::dumpObjectProperties(QObject* object) {
    if (!object) {
        return;
    }
    
    const QMetaObject* metaObject = object->metaObject();
    qDebug() << "Object:" << metaObject->className() << "(" << object->objectName() << ")";
    
    for (int i = 0; i < metaObject->propertyCount(); ++i) {
        QMetaProperty property = metaObject->property(i);
        QVariant value = property.read(object);
        qDebug() << "  " << property.name() << ":" << value.toString();
    }
}

void TestUtilities::logTestStep(const QString& step) {
    qDebug() << "[TEST STEP]" << step;
}

void TestUtilities::logTestResult(const QString& testName, bool passed, const QString& message) {
    QString status = passed ? "PASS" : "FAIL";
    QString logMessage = QString("[%1] %2").arg(status).arg(testName);
    if (!message.isEmpty()) {
        logMessage += ": " + message;
    }
    qDebug() << qPrintable(logMessage);
}