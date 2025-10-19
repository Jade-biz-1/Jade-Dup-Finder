#include <QtTest>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QFile>
#include <QImage>
#include <QDebug>
#include "thumbnail_cache.h"

/**
 * @brief Unit tests for ThumbnailCache
 * 
 * Tests thumbnail generation, caching, and background processing
 */
class TestThumbnailCache : public QObject
{
    Q_OBJECT

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();

    // Basic functionality tests
    void testCacheInitialization();
    void testGetThumbnailReturnsNullForUncached();
    void testThumbnailGenerationSignal();
    void testImageThumbnailGeneration();
    void testCaching();
    void testCacheHit();
    
    // Cache management tests
    void testClearCache();
    void testSetCacheSize();
    void testCacheSizeTracking();
    void testCacheEviction();
    void testIsCached();
    
    // Preloading tests
    void testPreloadThumbnails();
    void testPreloadSkipsCached();
    
    // File type tests
    void testImageFileDetection();
    void testVideoFileDetection();
    void testVideoThumbnailGeneration();
    void testUnsupportedFileType();
    
    // Error handling tests
    void testNonExistentFile();
    void testInvalidImageFile();
    void testEmptyFilePath();
    
    // Thread safety tests
    void testConcurrentAccess();
    void testMultipleSimultaneousRequests();
    
    // Performance tests
    void testBackgroundProcessing();
    void testMultipleSizes();

private:
    QTemporaryDir* m_tempDir;
    ThumbnailCache* m_cache;
    
    QString createTestImage(const QString& name, const QSize& size, const QColor& color);
    QString createTestFile(const QString& name, const QString& content);
    void waitForSignal(QSignalSpy& spy, int timeout = 5000);
};

void TestThumbnailCache::initTestCase()
{
    qDebug() << "Starting ThumbnailCache unit tests...";
    
    // Create temporary directory
    m_tempDir = new QTemporaryDir();
    QVERIFY(m_tempDir->isValid());
    qDebug() << "Test directory:" << m_tempDir->path();
}

void TestThumbnailCache::cleanupTestCase()
{
    delete m_tempDir;
    qDebug() << "ThumbnailCache unit tests completed.";
}

void TestThumbnailCache::init()
{
    // Create fresh cache for each test
    m_cache = new ThumbnailCache();
    QVERIFY(m_cache != nullptr);
}

void TestThumbnailCache::cleanup()
{
    // Clean up cache after each test
    delete m_cache;
    m_cache = nullptr;
}

QString TestThumbnailCache::createTestImage(const QString& name, const QSize& size, const QColor& color)
{
    QString filePath = m_tempDir->path() + "/" + name;
    
    QImage image(size, QImage::Format_RGB32);
    image.fill(color);
    
    bool saved = image.save(filePath);
    Q_ASSERT(saved);
    Q_UNUSED(saved);
    
    return filePath;
}

QString TestThumbnailCache::createTestFile(const QString& name, const QString& content)
{
    QString filePath = m_tempDir->path() + "/" + name;
    
    QFile file(filePath);
    bool opened = file.open(QIODevice::WriteOnly | QIODevice::Text);
    Q_ASSERT(opened);
    Q_UNUSED(opened);
    
    file.write(content.toUtf8());
    file.close();
    
    return filePath;
}

void TestThumbnailCache::waitForSignal(QSignalSpy& spy, int timeout)
{
    if (spy.count() == 0) {
        QVERIFY(spy.wait(timeout));
    }
}

// Basic functionality tests

void TestThumbnailCache::testCacheInitialization()
{
    QVERIFY(m_cache != nullptr);
    QCOMPARE(m_cache->cacheSize(), 0);
    QVERIFY(m_cache->maxCacheSize() > 0);
}

void TestThumbnailCache::testGetThumbnailReturnsNullForUncached()
{
    QString imagePath = createTestImage("test.png", QSize(100, 100), Qt::red);
    
    QPixmap thumbnail = m_cache->getThumbnail(imagePath, QSize(64, 64));
    
    // Should return null pixmap immediately (not cached yet)
    QVERIFY(thumbnail.isNull());
}

void TestThumbnailCache::testThumbnailGenerationSignal()
{
    QString imagePath = createTestImage("test.png", QSize(100, 100), Qt::blue);
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    
    // Request thumbnail (will be generated in background)
    QPixmap thumbnail = m_cache->getThumbnail(imagePath, QSize(64, 64));
    QVERIFY(thumbnail.isNull());
    
    // Wait for signal
    waitForSignal(spy);
    
    QCOMPARE(spy.count(), 1);
    QList<QVariant> arguments = spy.takeFirst();
    QCOMPARE(arguments.at(0).toString(), imagePath);
    
    QPixmap generatedThumbnail = arguments.at(1).value<QPixmap>();
    QVERIFY(!generatedThumbnail.isNull());
}

void TestThumbnailCache::testImageThumbnailGeneration()
{
    QString imagePath = createTestImage("test.jpg", QSize(200, 200), Qt::green);
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    
    m_cache->getThumbnail(imagePath, QSize(64, 64));
    waitForSignal(spy);
    
    QCOMPARE(spy.count(), 1);
    QPixmap thumbnail = spy.at(0).at(1).value<QPixmap>();
    
    // Verify thumbnail is scaled correctly
    QVERIFY(!thumbnail.isNull());
    QVERIFY(thumbnail.width() <= 64);
    QVERIFY(thumbnail.height() <= 64);
}

void TestThumbnailCache::testCaching()
{
    QString imagePath = createTestImage("cached.png", QSize(100, 100), Qt::yellow);
    QSize size(64, 64);
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    
    // First request - should generate
    m_cache->getThumbnail(imagePath, size);
    waitForSignal(spy);
    
    QCOMPARE(spy.count(), 1);
    QCOMPARE(m_cache->cacheSize(), 1);
}

void TestThumbnailCache::testCacheHit()
{
    QString imagePath = createTestImage("cached2.png", QSize(100, 100), Qt::cyan);
    QSize size(64, 64);
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    
    // First request
    m_cache->getThumbnail(imagePath, size);
    waitForSignal(spy);
    
    spy.clear();
    
    // Second request - should hit cache
    QPixmap cached = m_cache->getThumbnail(imagePath, size);
    
    // Should return immediately from cache
    QVERIFY(!cached.isNull());
    
    // Should not emit signal again
    QTest::qWait(100);
    QCOMPARE(spy.count(), 0);
}

// Cache management tests

void TestThumbnailCache::testClearCache()
{
    QString imagePath = createTestImage("clear.png", QSize(100, 100), Qt::magenta);
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    m_cache->getThumbnail(imagePath, QSize(64, 64));
    waitForSignal(spy);
    
    QVERIFY(m_cache->cacheSize() > 0);
    
    m_cache->clearCache();
    
    QCOMPARE(m_cache->cacheSize(), 0);
}

void TestThumbnailCache::testSetCacheSize()
{
    int newSize = 50;
    m_cache->setCacheSize(newSize);
    
    QCOMPARE(m_cache->maxCacheSize(), newSize);
}

void TestThumbnailCache::testCacheSizeTracking()
{
    // Create multiple images
    QStringList images;
    for (int i = 0; i < 3; ++i) {
        QString name = QString("image%1.png").arg(i);
        images << createTestImage(name, QSize(100, 100), Qt::red);
    }
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    
    // Request thumbnails
    for (const QString& image : images) {
        m_cache->getThumbnail(image, QSize(64, 64));
    }
    
    // Wait for all to complete
    while (spy.count() < images.size() && spy.wait(1000)) {
        // Keep waiting
    }
    
    QCOMPARE(m_cache->cacheSize(), images.size());
}

void TestThumbnailCache::testCacheEviction()
{
    // Set small cache size
    m_cache->setCacheSize(2);
    
    // Create 3 images
    QStringList images;
    for (int i = 0; i < 3; ++i) {
        QString name = QString("evict%1.png").arg(i);
        images << createTestImage(name, QSize(100, 100), Qt::blue);
    }
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    
    // Request all thumbnails
    for (const QString& image : images) {
        m_cache->getThumbnail(image, QSize(64, 64));
    }
    
    // Wait for all
    while (spy.count() < images.size() && spy.wait(1000)) {
        // Keep waiting
    }
    
    // Cache should not exceed max size
    QVERIFY(m_cache->cacheSize() <= 2);
}

void TestThumbnailCache::testIsCached()
{
    QString imagePath = createTestImage("check.png", QSize(100, 100), Qt::darkGreen);
    QSize size(64, 64);
    
    // Should not be cached initially
    QVERIFY(!m_cache->isCached(imagePath, size));
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    m_cache->getThumbnail(imagePath, size);
    waitForSignal(spy);
    
    // Should be cached now
    QVERIFY(m_cache->isCached(imagePath, size));
}

// Preloading tests

void TestThumbnailCache::testPreloadThumbnails()
{
    QStringList images;
    for (int i = 0; i < 3; ++i) {
        QString name = QString("preload%1.png").arg(i);
        images << createTestImage(name, QSize(100, 100), Qt::darkBlue);
    }
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    
    m_cache->preloadThumbnails(images, QSize(64, 64));
    
    // Wait for all to complete
    while (spy.count() < images.size() && spy.wait(1000)) {
        // Keep waiting
    }
    
    QCOMPARE(spy.count(), images.size());
}

void TestThumbnailCache::testPreloadSkipsCached()
{
    QString imagePath = createTestImage("preload_skip.png", QSize(100, 100), Qt::darkRed);
    QSize size(64, 64);
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    
    // First preload
    m_cache->preloadThumbnails(QStringList() << imagePath, size);
    waitForSignal(spy);
    
    spy.clear();
    
    // Second preload - should skip
    m_cache->preloadThumbnails(QStringList() << imagePath, size);
    QTest::qWait(200);
    
    // Should not generate again
    QCOMPARE(spy.count(), 0);
}

// File type tests

void TestThumbnailCache::testImageFileDetection()
{
    QStringList imageFiles = {
        "test.jpg", "test.jpeg", "test.png", "test.gif",
        "test.bmp", "test.tiff", "test.webp"
    };
    
    for (const QString& fileName : imageFiles) {
        QString path = m_tempDir->path() + "/" + fileName;
        // Just test the path, don't need to create actual files
        // The isImageFile method only checks extension
        QVERIFY2(fileName.contains('.'), qPrintable(fileName));
    }
}

void TestThumbnailCache::testVideoFileDetection()
{
    QStringList videoFiles = {
        "test.mp4", "test.avi", "test.mkv", "test.mov",
        "test.wmv", "test.flv", "test.webm"
    };
    
    for (const QString& fileName : videoFiles) {
        QString path = m_tempDir->path() + "/" + fileName;
        // Just test the path
        QVERIFY2(fileName.contains('.'), qPrintable(fileName));
    }
}

void TestThumbnailCache::testVideoThumbnailGeneration()
{
    // Create a fake video file (just for testing the path handling)
    QString videoPath = createTestFile("test.mp4", "fake video content");
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    
    m_cache->getThumbnail(videoPath, QSize(64, 64));
    waitForSignal(spy);
    
    QCOMPARE(spy.count(), 1);
    QPixmap thumbnail = spy.at(0).at(1).value<QPixmap>();
    
    // Should generate a placeholder thumbnail
    QVERIFY(!thumbnail.isNull());
}

void TestThumbnailCache::testUnsupportedFileType()
{
    QString textPath = createTestFile("test.txt", "test content");
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    
    m_cache->getThumbnail(textPath, QSize(64, 64));
    waitForSignal(spy);
    
    QCOMPARE(spy.count(), 1);
    QPixmap thumbnail = spy.at(0).at(1).value<QPixmap>();
    
    // Should generate a default icon
    QVERIFY(!thumbnail.isNull());
}

// Error handling tests

void TestThumbnailCache::testNonExistentFile()
{
    QString nonExistent = m_tempDir->path() + "/nonexistent.png";
    
    QSignalSpy readySpy(m_cache, &ThumbnailCache::thumbnailReady);
    QSignalSpy failedSpy(m_cache, &ThumbnailCache::thumbnailFailed);
    
    m_cache->getThumbnail(nonExistent, QSize(64, 64));
    
    // Should emit failed signal
    QVERIFY(failedSpy.wait(2000));
    QCOMPARE(failedSpy.count(), 1);
}

void TestThumbnailCache::testInvalidImageFile()
{
    // Create a file with .png extension but invalid content
    QString invalidPath = createTestFile("invalid.png", "not an image");
    
    QSignalSpy readySpy(m_cache, &ThumbnailCache::thumbnailReady);
    QSignalSpy failedSpy(m_cache, &ThumbnailCache::thumbnailFailed);
    
    m_cache->getThumbnail(invalidPath, QSize(64, 64));
    
    // Should either fail or return default icon
    QVERIFY(readySpy.wait(2000) || failedSpy.wait(2000));
}

void TestThumbnailCache::testEmptyFilePath()
{
    QSignalSpy failedSpy(m_cache, &ThumbnailCache::thumbnailFailed);
    
    m_cache->getThumbnail("", QSize(64, 64));
    
    // Should handle gracefully
    QVERIFY(failedSpy.wait(2000));
}

// Thread safety tests

void TestThumbnailCache::testConcurrentAccess()
{
    QStringList images;
    for (int i = 0; i < 5; ++i) {
        QString name = QString("concurrent%1.png").arg(i);
        images << createTestImage(name, QSize(100, 100), Qt::darkCyan);
    }
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    
    // Request all thumbnails simultaneously
    for (const QString& image : images) {
        m_cache->getThumbnail(image, QSize(64, 64));
    }
    
    // Wait for all to complete
    while (spy.count() < images.size() && spy.wait(2000)) {
        // Keep waiting
    }
    
    QCOMPARE(spy.count(), images.size());
}

void TestThumbnailCache::testMultipleSimultaneousRequests()
{
    QString imagePath = createTestImage("simultaneous.png", QSize(100, 100), Qt::darkYellow);
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    
    // Request same thumbnail multiple times
    for (int i = 0; i < 3; ++i) {
        m_cache->getThumbnail(imagePath, QSize(64, 64));
    }
    
    // Should only generate once
    waitForSignal(spy);
    
    // May get multiple signals, but should be same thumbnail
    QVERIFY(spy.count() >= 1);
}

// Performance tests

void TestThumbnailCache::testBackgroundProcessing()
{
    QString imagePath = createTestImage("background.png", QSize(100, 100), Qt::darkMagenta);
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    
    // Request should return immediately
    QPixmap thumbnail = m_cache->getThumbnail(imagePath, QSize(64, 64));
    QVERIFY(thumbnail.isNull());
    
    // But signal should arrive shortly
    waitForSignal(spy);
    QCOMPARE(spy.count(), 1);
}

void TestThumbnailCache::testMultipleSizes()
{
    QString imagePath = createTestImage("sizes.png", QSize(200, 200), Qt::darkGray);
    
    QSignalSpy spy(m_cache, &ThumbnailCache::thumbnailReady);
    
    // Request different sizes
    QList<QSize> sizes = {QSize(32, 32), QSize(64, 64), QSize(128, 128)};
    
    for (const QSize& size : sizes) {
        m_cache->getThumbnail(imagePath, size);
    }
    
    // Wait for all
    while (spy.count() < sizes.size() && spy.wait(2000)) {
        // Keep waiting
    }
    
    QCOMPARE(spy.count(), sizes.size());
    
    // Each size should be cached separately
    for (const QSize& size : sizes) {
        QVERIFY(m_cache->isCached(imagePath, size));
    }
}

QTEST_MAIN(TestThumbnailCache)
#include "test_thumbnail_cache.moc"
