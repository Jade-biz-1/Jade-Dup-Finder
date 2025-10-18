#include "test_data_generator.h"
#include <QStandardPaths>
#include <QCoreApplication>
#include <QTextStream>
#include <QDataStream>
#include <QBuffer>
#include <QImageWriter>
#include <QDirIterator>
#include <QProcess>
#include <QThread>
#include <QDebug>
#include <QUuid>
#include <QRegularExpression>

TestDataGenerator::TestDataGenerator()
    : m_tempDir(nullptr)
    , m_randomGenerator(QRandomGenerator::global())
    , m_cleanupOnDestruction(true)
{
    // Create temporary directory for test data
    if (m_customTempDirectory.isEmpty()) {
        m_tempDir = new QTemporaryDir();
        if (!m_tempDir->isValid()) {
            qWarning() << "Failed to create temporary directory for test data";
        }
    }
}

TestDataGenerator::~TestDataGenerator() {
    if (m_cleanupOnDestruction) {
        cleanupGeneratedData();
    }
    delete m_tempDir;
}

QString TestDataGenerator::generateTestDirectory(const DirectorySpec& spec, const QString& basePath) {
    if (!validateDirectorySpec(spec)) {
        qWarning() << "Invalid directory specification";
        return QString();
    }
    
    QString targetPath = basePath;
    if (targetPath.isEmpty()) {
        targetPath = m_customTempDirectory.isEmpty() ? m_tempDir->path() : m_customTempDirectory;
    }
    
    QString directoryPath = QDir(targetPath).absoluteFilePath(spec.name);
    
    // Create the directory structure
    QString createdPath = createDirectoryStructure(directoryPath, spec);
    if (!createdPath.isEmpty()) {
        registerGeneratedPath(createdPath);
        qDebug() << "Generated test directory:" << createdPath;
    }
    
    return createdPath;
}

QString TestDataGenerator::generateTestFile(const FileSpec& spec, const QString& directory) {
    if (!validateFileSpec(spec)) {
        qWarning() << "Invalid file specification";
        return QString();
    }
    
    QString targetDir = directory;
    if (targetDir.isEmpty()) {
        targetDir = m_customTempDirectory.isEmpty() ? m_tempDir->path() : m_customTempDirectory;
    }
    
    QDir().mkpath(targetDir);
    QString filePath = QDir(targetDir).absoluteFilePath(spec.fileName);
    
    // Generate content based on specification
    QByteArray content;
    if (!spec.content.isEmpty()) {
        content = spec.content.toUtf8();
    } else if (spec.sizeBytes > 0) {
        content = generateFileContent(spec.extension, spec.sizeBytes);
    } else {
        // Generate default content based on file type
        content = generateFileContent(spec.extension, 1024); // 1KB default
    }
    
    // Create the file
    createFileWithContent(filePath, content, spec.lastModified);
    
    // Set platform-specific attributes if specified
    if (!spec.metadata.isEmpty()) {
        setPlatformSpecificAttributes(filePath, spec.metadata);
    }
    
    registerGeneratedPath(filePath);
    return filePath;
}

QStringList TestDataGenerator::generateFileHierarchy(TestScenario scenario, const QString& basePath) {
    DirectorySpec spec = getScenarioSpec(scenario);
    QString generatedPath = generateTestDirectory(spec, basePath);
    
    if (generatedPath.isEmpty()) {
        return QStringList();
    }
    
    // Return list of all generated files
    QStringList files;
    QDirIterator iterator(generatedPath, QDir::Files, QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        files.append(iterator.next());
    }
    
    return files;
}

QString TestDataGenerator::createDuplicateFile(const QString& originalFile, const QString& targetDirectory) {
    if (!QFile::exists(originalFile)) {
        qWarning() << "Original file does not exist:" << originalFile;
        return QString();
    }
    
    QFileInfo originalInfo(originalFile);
    QString targetDir = targetDirectory.isEmpty() ? originalInfo.absolutePath() : targetDirectory;
    QDir().mkpath(targetDir);
    
    // Generate unique name for duplicate
    QString duplicateName = generateUniqueFileName(
        originalInfo.baseName() + "_duplicate", 
        originalInfo.suffix()
    );
    QString duplicatePath = QDir(targetDir).absoluteFilePath(duplicateName);
    
    // Copy the file
    if (QFile::copy(originalFile, duplicatePath)) {
        registerGeneratedPath(duplicatePath);
        return duplicatePath;
    }
    
    qWarning() << "Failed to create duplicate file:" << duplicatePath;
    return QString();
}

QStringList TestDataGenerator::createDuplicateSet(const QString& originalFile, int duplicateCount, const QString& baseDirectory) {
    QStringList duplicates;
    
    for (int i = 0; i < duplicateCount; ++i) {
        QString duplicate = createDuplicateFile(originalFile, baseDirectory);
        if (!duplicate.isEmpty()) {
            duplicates.append(duplicate);
        }
    }
    
    return duplicates;
}

void TestDataGenerator::generateDuplicateScenario(const QString& directory, double duplicateRatio) {
    QDir dir(directory);
    if (!dir.exists()) {
        qWarning() << "Directory does not exist:" << directory;
        return;
    }
    
    // Get all files in directory
    QStringList files = dir.entryList(QDir::Files);
    int duplicateCount = static_cast<int>(files.size() * duplicateRatio);
    
    // Create duplicates for random files
    for (int i = 0; i < duplicateCount && i < files.size(); ++i) {
        int randomIndex = m_randomGenerator->bounded(files.size());
        QString originalFile = dir.absoluteFilePath(files[randomIndex]);
        createDuplicateFile(originalFile, directory);
    }
}

QByteArray TestDataGenerator::generateFileContent(const QString& fileType, qint64 sizeBytes) {
    QString lowerType = fileType.toLower();
    
    if (lowerType == "txt" || lowerType == "log" || lowerType == "md") {
        return generateTextFileContent(sizeBytes);
    } else if (lowerType == "jpg" || lowerType == "jpeg" || lowerType == "png" || lowerType == "bmp") {
        return generateImageFileContent(lowerType);
    } else if (lowerType == "pdf" || lowerType == "doc" || lowerType == "docx") {
        return generateDocumentContent(lowerType, sizeBytes);
    } else {
        return generateBinaryFileContent(sizeBytes);
    }
}

QByteArray TestDataGenerator::generateRandomBinaryData(qint64 sizeBytes) {
    QByteArray data;
    data.reserve(sizeBytes);
    
    for (qint64 i = 0; i < sizeBytes; ++i) {
        data.append(static_cast<char>(m_randomGenerator->bounded(256)));
    }
    
    return data;
}

QString TestDataGenerator::generateTextContent(qint64 sizeBytes, const QString& pattern) {
    QString content;
    content.reserve(sizeBytes);
    
    QString basePattern = pattern.isEmpty() ? 
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " : pattern;
    
    while (content.length() < sizeBytes) {
        content += basePattern;
        if (content.length() % 100 == 0) {
            content += "\n"; // Add line breaks
        }
    }
    
    return content.left(sizeBytes);
}

QPixmap TestDataGenerator::generateImageContent(int width, int height, const QString& format) {
    QPixmap pixmap(width, height);
    
    // Generate random colored image
    QImage image(width, height, QImage::Format_RGB32);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            QRgb color = qRgb(
                m_randomGenerator->bounded(256),
                m_randomGenerator->bounded(256),
                m_randomGenerator->bounded(256)
            );
            image.setPixel(x, y, color);
        }
    }
    
    return QPixmap::fromImage(image);
}

QString TestDataGenerator::generateDocumentFile(const QString& directory, const QString& type) {
    FileSpec spec;
    spec.fileName = generateUniqueFileName("document", type);
    spec.extension = type;
    spec.sizeBytes = 5000 + m_randomGenerator->bounded(10000); // 5-15KB
    spec.lastModified = generateRandomDateTime();
    
    return generateTestFile(spec, directory);
}

QString TestDataGenerator::generateImageFile(const QString& directory, const QString& format) {
    FileSpec spec;
    spec.fileName = generateUniqueFileName("image", format);
    spec.extension = format;
    spec.sizeBytes = 50000 + m_randomGenerator->bounded(200000); // 50-250KB
    spec.lastModified = generateRandomDateTime();
    
    return generateTestFile(spec, directory);
}

QString TestDataGenerator::generateMediaFile(const QString& directory, const QString& format) {
    FileSpec spec;
    spec.fileName = generateUniqueFileName("media", format);
    spec.extension = format;
    spec.sizeBytes = 1000000 + m_randomGenerator->bounded(5000000); // 1-6MB
    spec.lastModified = generateRandomDateTime();
    
    return generateTestFile(spec, directory);
}

void TestDataGenerator::generateCorruptedFiles(const QString& directory, int count) {
    for (int i = 0; i < count; ++i) {
        FileSpec spec;
        spec.fileName = generateUniqueFileName("corrupted", "dat");
        spec.extension = "dat";
        
        // Generate partially corrupted content
        QByteArray content = generateRandomBinaryData(1024);
        // Introduce corruption patterns
        for (int j = 0; j < content.size(); j += 100) {
            content[j] = 0xFF; // Corruption marker
        }
        spec.content = QString::fromLatin1(content);
        
        generateTestFile(spec, directory);
    }
}

void TestDataGenerator::generateLongPathFiles(const QString& directory, int pathLength) {
    QString longPath = directory;
    QString segment = "very_long_directory_name_segment";
    
    // Build long path
    while (longPath.length() < pathLength - segment.length() - 20) {
        longPath = QDir(longPath).absoluteFilePath(segment);
        QDir().mkpath(longPath);
    }
    
    // Create file with long path
    FileSpec spec;
    spec.fileName = "file_with_very_long_path.txt";
    spec.content = "This file has a very long path for testing purposes.";
    
    generateTestFile(spec, longPath);
}

void TestDataGenerator::generateSpecialCharacterFiles(const QString& directory) {
    QStringList specialNames = {
        "file with spaces.txt",
        "file-with-dashes.txt",
        "file_with_underscores.txt",
        "file.with.dots.txt",
        "file(with)parentheses.txt",
        "file[with]brackets.txt",
        "file{with}braces.txt",
        "file@with#special$chars%.txt"
    };
    
    for (const QString& name : specialNames) {
        FileSpec spec;
        spec.fileName = name;
        spec.content = QString("Content for file: %1").arg(name);
        
        generateTestFile(spec, directory);
    }
}

void TestDataGenerator::generateSymlinks(const QString& directory, const QStringList& targets) {
    for (int i = 0; i < targets.size(); ++i) {
        QString linkName = QString("symlink_%1").arg(i + 1);
        QString linkPath = QDir(directory).absoluteFilePath(linkName);
        
        if (createSymbolicLink(linkPath, targets[i])) {
            registerGeneratedPath(linkPath);
        }
    }
}

void TestDataGenerator::generateHiddenFiles(const QString& directory, int count) {
    for (int i = 0; i < count; ++i) {
        FileSpec spec;
        spec.fileName = QString(".hidden_file_%1.txt").arg(i + 1);
        spec.content = "This is a hidden file for testing purposes.";
        
        QString filePath = generateTestFile(spec, directory);
        if (!filePath.isEmpty()) {
            setFileHidden(filePath, true);
        }
    }
}

QString TestDataGenerator::generatePerformanceDataset(qint64 totalSizeBytes, const QString& basePath) {
    DirectorySpec spec;
    spec.name = "performance_dataset";
    spec.depth = 3;
    spec.filesPerDirectory = 50;
    spec.subdirectories = 5;
    spec.minFileSize = totalSizeBytes / 1000; // Distribute size across files
    spec.maxFileSize = totalSizeBytes / 100;
    spec.duplicateRatio = 0.1; // 10% duplicates for performance testing
    
    return generateTestDirectory(spec, basePath);
}

QString TestDataGenerator::generateLargeFileSet(int fileCount, qint64 fileSizeBytes, const QString& basePath) {
    QString targetPath = basePath.isEmpty() ? 
        (m_customTempDirectory.isEmpty() ? m_tempDir->path() : m_customTempDirectory) : basePath;
    
    QString directoryPath = QDir(targetPath).absoluteFilePath("large_file_set");
    QDir().mkpath(directoryPath);
    
    for (int i = 0; i < fileCount; ++i) {
        FileSpec spec;
        spec.fileName = QString("large_file_%1.dat").arg(i + 1, 4, 10, QChar('0'));
        spec.sizeBytes = fileSizeBytes;
        spec.extension = "dat";
        
        generateTestFile(spec, directoryPath);
        
        // Log progress for large operations
        if ((i + 1) % 10 == 0) {
            logGenerationProgress("Large file generation", i + 1, fileCount);
        }
    }
    
    registerGeneratedPath(directoryPath);
    return directoryPath;
}

QString TestDataGenerator::generateDeepDirectoryStructure(int depth, int width, const QString& basePath) {
    QString targetPath = basePath.isEmpty() ? 
        (m_customTempDirectory.isEmpty() ? m_tempDir->path() : m_customTempDirectory) : basePath;
    
    QString rootPath = QDir(targetPath).absoluteFilePath("deep_structure");
    QDir().mkpath(rootPath);
    
    // Create deep structure recursively
    QString currentPath = rootPath;
    for (int d = 0; d < depth; ++d) {
        // Create width directories at each level
        for (int w = 0; w < width; ++w) {
            QString dirName = QString("level_%1_dir_%2").arg(d + 1).arg(w + 1);
            QString dirPath = QDir(currentPath).absoluteFilePath(dirName);
            QDir().mkpath(dirPath);
            
            // Add some files at each level
            for (int f = 0; f < 3; ++f) {
                FileSpec spec;
                spec.fileName = QString("file_%1.txt").arg(f + 1);
                spec.content = QString("Content at depth %1, width %2, file %3").arg(d + 1).arg(w + 1).arg(f + 1);
                generateTestFile(spec, dirPath);
            }
            
            // Use first directory for next depth level
            if (w == 0) {
                currentPath = dirPath;
            }
        }
    }
    
    registerGeneratedPath(rootPath);
    return rootPath;
}

void TestDataGenerator::registerGeneratedPath(const QString& path) {
    if (!m_generatedPaths.contains(path)) {
        m_generatedPaths.append(path);
    }
}

void TestDataGenerator::cleanupGeneratedData() {
    for (const QString& path : m_generatedPaths) {
        cleanupPath(path);
    }
    m_generatedPaths.clear();
}

void TestDataGenerator::cleanupPath(const QString& path) {
    QFileInfo info(path);
    if (info.isFile()) {
        if (!QFile::remove(path)) {
            qWarning() << "Failed to remove file:" << path;
        }
    } else if (info.isDir()) {
        QDir dir(path);
        if (!dir.removeRecursively()) {
            qWarning() << "Failed to remove directory:" << path;
        }
    }
}

QStringList TestDataGenerator::getGeneratedPaths() const {
    return m_generatedPaths;
}

bool TestDataGenerator::verifyGeneratedStructure(const QString& path, const DirectorySpec& expectedSpec) {
    QDir dir(path);
    if (!dir.exists()) {
        return false;
    }
    
    // Count files and directories
    QStringList files = dir.entryList(QDir::Files);
    QStringList subdirs = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    
    // Basic validation
    bool valid = true;
    if (files.size() < expectedSpec.filesPerDirectory * 0.8) { // Allow 20% variance
        qWarning() << "File count mismatch. Expected ~" << expectedSpec.filesPerDirectory << ", got" << files.size();
        valid = false;
    }
    
    if (subdirs.size() != expectedSpec.subdirectories) {
        qWarning() << "Subdirectory count mismatch. Expected" << expectedSpec.subdirectories << ", got" << subdirs.size();
        valid = false;
    }
    
    return valid;
}

QMap<QString, qint64> TestDataGenerator::analyzeGeneratedData(const QString& path) {
    QMap<QString, qint64> analysis;
    
    qint64 totalSize = 0;
    qint64 fileCount = 0;
    qint64 dirCount = 0;
    
    QDirIterator iterator(path, QDir::AllEntries | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);
    while (iterator.hasNext()) {
        QString itemPath = iterator.next();
        QFileInfo info(itemPath);
        
        if (info.isFile()) {
            fileCount++;
            totalSize += info.size();
        } else if (info.isDir()) {
            dirCount++;
        }
    }
    
    analysis["totalSize"] = totalSize;
    analysis["fileCount"] = fileCount;
    analysis["directoryCount"] = dirCount;
    analysis["averageFileSize"] = fileCount > 0 ? totalSize / fileCount : 0;
    
    return analysis;
}

QString TestDataGenerator::generateDatasetReport(const QString& path) {
    QMap<QString, qint64> analysis = analyzeGeneratedData(path);
    
    QString report;
    QTextStream stream(&report);
    
    stream << "Test Dataset Analysis Report\n";
    stream << "============================\n";
    stream << "Path: " << path << "\n";
    stream << "Generated: " << QDateTime::currentDateTime().toString() << "\n\n";
    stream << "Statistics:\n";
    stream << "- Total Size: " << analysis["totalSize"] << " bytes\n";
    stream << "- File Count: " << analysis["fileCount"] << "\n";
    stream << "- Directory Count: " << analysis["directoryCount"] << "\n";
    stream << "- Average File Size: " << analysis["averageFileSize"] << " bytes\n";
    
    return report;
}

void TestDataGenerator::setRandomSeed(quint32 seed) {
    m_randomGenerator->seed(seed);
}

void TestDataGenerator::setTemporaryDirectory(const QString& tempDir) {
    m_customTempDirectory = tempDir;
    QDir().mkpath(tempDir);
}

void TestDataGenerator::setCleanupOnDestruction(bool cleanup) {
    m_cleanupOnDestruction = cleanup;
}

// Static utility functions
QString TestDataGenerator::generateUniqueFileName(const QString& baseName, const QString& extension) {
    QString timestamp = QString::number(QDateTime::currentMSecsSinceEpoch());
    QString randomSuffix = QString::number(QRandomGenerator::global()->bounded(1000), 16);
    
    QString fileName = QString("%1_%2_%3").arg(baseName).arg(timestamp).arg(randomSuffix);
    if (!extension.isEmpty()) {
        fileName += "." + extension;
    }
    
    return fileName;
}

QString TestDataGenerator::generateRandomString(int length, bool alphaNumericOnly) {
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

QDateTime TestDataGenerator::generateRandomDateTime(const QDateTime& start, const QDateTime& end) {
    qint64 startMs = start.toMSecsSinceEpoch();
    qint64 endMs = end.toMSecsSinceEpoch();
    qint64 randomMs = startMs + QRandomGenerator::global()->bounded(endMs - startMs);
    
    return QDateTime::fromMSecsSinceEpoch(randomMs);
}

QString TestDataGenerator::calculateFileHash(const QString& filePath, QCryptographicHash::Algorithm algorithm) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        return QString();
    }
    
    QCryptographicHash hash(algorithm);
    hash.addData(&file);
    
    return hash.result().toHex();
}

TestDataGenerator::DirectorySpec TestDataGenerator::getScenarioSpec(TestScenario scenario) {
    DirectorySpec spec;
    
    switch (scenario) {
        case TestScenario::EmptyDirectory:
            spec.name = "empty_directory";
            spec.filesPerDirectory = 0;
            spec.subdirectories = 0;
            break;
            
        case TestScenario::SingleFile:
            spec.name = "single_file";
            spec.filesPerDirectory = 1;
            spec.subdirectories = 0;
            break;
            
        case TestScenario::SmallDataset:
            spec.name = "small_dataset";
            spec.depth = 2;
            spec.filesPerDirectory = 10;
            spec.subdirectories = 3;
            spec.duplicateRatio = 0.1;
            break;
            
        case TestScenario::MediumDataset:
            spec.name = "medium_dataset";
            spec.depth = 3;
            spec.filesPerDirectory = 25;
            spec.subdirectories = 5;
            spec.duplicateRatio = 0.2;
            break;
            
        case TestScenario::LargeDataset:
            spec.name = "large_dataset";
            spec.depth = 4;
            spec.filesPerDirectory = 50;
            spec.subdirectories = 8;
            spec.duplicateRatio = 0.3;
            break;
            
        case TestScenario::DeepHierarchy:
            spec.name = "deep_hierarchy";
            spec.depth = 10;
            spec.filesPerDirectory = 5;
            spec.subdirectories = 2;
            break;
            
        case TestScenario::WideHierarchy:
            spec.name = "wide_hierarchy";
            spec.depth = 2;
            spec.filesPerDirectory = 20;
            spec.subdirectories = 15;
            break;
            
        case TestScenario::DuplicateHeavy:
            spec.name = "duplicate_heavy";
            spec.depth = 3;
            spec.filesPerDirectory = 20;
            spec.subdirectories = 4;
            spec.duplicateRatio = 0.7;
            break;
            
        case TestScenario::PerformanceStress:
            spec.name = "performance_stress";
            spec.depth = 3;
            spec.filesPerDirectory = 100;
            spec.subdirectories = 5;
            spec.minFileSize = 1048576; // 1MB
            spec.maxFileSize = 10485760; // 10MB
            break;
            
        case TestScenario::EdgeCases:
            spec.name = "edge_cases";
            spec.filesPerDirectory = 15;
            spec.subdirectories = 3;
            spec.createSymlinks = true;
            spec.createHiddenFiles = true;
            break;
            
        default:
            spec.name = "default_scenario";
            spec.filesPerDirectory = 10;
            spec.subdirectories = 3;
            break;
    }
    
    return spec;
}

// Private helper methods implementation
QString TestDataGenerator::createDirectoryStructure(const QString& basePath, const DirectorySpec& spec, int currentDepth) {
    QDir().mkpath(basePath);
    
    // Populate current directory with files
    populateDirectoryWithFiles(basePath, spec);
    
    // Create subdirectories if we haven't reached max depth
    if (currentDepth < spec.depth) {
        for (int i = 0; i < spec.subdirectories; ++i) {
            QString subDirName = QString("subdir_%1").arg(i + 1, 2, 10, QChar('0'));
            QString subDirPath = QDir(basePath).absoluteFilePath(subDirName);
            createDirectoryStructure(subDirPath, spec, currentDepth + 1);
        }
    }
    
    // Generate duplicates if specified
    if (spec.duplicateRatio > 0) {
        generateDuplicateScenario(basePath, spec.duplicateRatio);
    }
    
    // Create special files if requested
    if (spec.createSymlinks) {
        QStringList files = QDir(basePath).entryList(QDir::Files);
        if (!files.isEmpty()) {
            generateSymlinks(basePath, {QDir(basePath).absoluteFilePath(files.first())});
        }
    }
    
    if (spec.createHiddenFiles) {
        generateHiddenFiles(basePath, 2);
    }
    
    return basePath;
}

void TestDataGenerator::populateDirectoryWithFiles(const QString& directory, const DirectorySpec& spec) {
    for (int i = 0; i < spec.filesPerDirectory; ++i) {
        FileSpec fileSpec = generateRandomFileSpec(spec);
        fileSpec.fileName = QString("file_%1.%2").arg(i + 1, 3, 10, QChar('0')).arg(fileSpec.extension);
        generateTestFile(fileSpec, directory);
    }
}

TestDataGenerator::FileSpec TestDataGenerator::generateRandomFileSpec(const DirectorySpec& dirSpec) {
    FileSpec spec;
    spec.extension = selectRandomFileType(dirSpec.fileTypes);
    spec.sizeBytes = generateRandomFileSize(dirSpec.minFileSize, dirSpec.maxFileSize);
    spec.lastModified = generateRandomDateTime();
    return spec;
}

QString TestDataGenerator::selectRandomFileType(const QStringList& types) {
    if (types.isEmpty()) {
        return "txt";
    }
    return types[m_randomGenerator->bounded(types.size())];
}

qint64 TestDataGenerator::generateRandomFileSize(qint64 minSize, qint64 maxSize) {
    if (minSize >= maxSize) {
        return minSize;
    }
    return minSize + m_randomGenerator->bounded(maxSize - minSize);
}

void TestDataGenerator::createFileWithContent(const QString& filePath, const QByteArray& content, const QDateTime& lastModified) {
    QFile file(filePath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(content);
        file.close();
        
        // Set last modified time if specified
        if (lastModified.isValid()) {
            file.setFileTime(lastModified, QFileDevice::FileModificationTime);
        }
    } else {
        qWarning() << "Failed to create file:" << filePath;
    }
}

// Content generation methods
QByteArray TestDataGenerator::generateTextFileContent(qint64 sizeBytes) {
    return generateTextContent(sizeBytes).toUtf8();
}

QByteArray TestDataGenerator::generateBinaryFileContent(qint64 sizeBytes) {
    return generateRandomBinaryData(sizeBytes);
}

QByteArray TestDataGenerator::generateImageFileContent(const QString& format, int width, int height) {
    QPixmap pixmap = generateImageContent(width, height, format);
    
    QByteArray data;
    QBuffer buffer(&data);
    buffer.open(QIODevice::WriteOnly);
    
    QString imageFormat = format.toUpper();
    if (imageFormat == "JPG") imageFormat = "JPEG";
    
    pixmap.save(&buffer, imageFormat.toUtf8().constData());
    return data;
}

QByteArray TestDataGenerator::generateDocumentContent(const QString& type, qint64 sizeBytes) {
    // For now, generate text content for document types
    // In a real implementation, you might generate actual document formats
    QString content = QString("Document Type: %1\n\n").arg(type.toUpper());
    content += generateTextContent(sizeBytes - content.length());
    return content.toUtf8();
}

// Platform-specific implementations (simplified)
void TestDataGenerator::setPlatformSpecificAttributes(const QString& filePath, const QMap<QString, QVariant>& attributes) {
    Q_UNUSED(filePath)
    Q_UNUSED(attributes)
    // Platform-specific attribute setting would be implemented here
}

bool TestDataGenerator::createSymbolicLink(const QString& linkPath, const QString& targetPath) {
#ifdef Q_OS_WIN
    // Windows symbolic link creation
    return QFile::link(targetPath, linkPath);
#else
    // Unix/Linux symbolic link creation
    return QFile::link(targetPath, linkPath);
#endif
}

void TestDataGenerator::setFileHidden(const QString& filePath, bool hidden) {
    Q_UNUSED(filePath)
    Q_UNUSED(hidden)
    // Platform-specific hidden file attribute setting
#ifdef Q_OS_WIN
    // Windows hidden attribute
#else
    // Unix hidden files start with '.'
#endif
}

bool TestDataGenerator::validateDirectorySpec(const DirectorySpec& spec) {
    return !spec.name.isEmpty() && 
           spec.depth >= 0 && 
           spec.filesPerDirectory >= 0 && 
           spec.subdirectories >= 0 &&
           spec.minFileSize >= 0 &&
           spec.maxFileSize >= spec.minFileSize;
}

bool TestDataGenerator::validateFileSpec(const FileSpec& spec) {
    return !spec.fileName.isEmpty() && spec.sizeBytes >= 0;
}

void TestDataGenerator::logGenerationProgress(const QString& operation, int current, int total) {
    if (total > 0 && (current % (total / 10 + 1) == 0)) {
        int percentage = (current * 100) / total;
        qDebug() << QString("%1: %2% (%3/%4)").arg(operation).arg(percentage).arg(current).arg(total);
    }
}

// Static utility implementations
QStringList TestDataGenerator::getCommonFileExtensions() {
    return {"txt", "log", "dat", "bin", "tmp", "bak", "cfg", "ini", "xml", "json"};
}

QStringList TestDataGenerator::getImageExtensions() {
    return {"jpg", "jpeg", "png", "bmp", "gif", "tiff", "svg", "webp"};
}

QStringList TestDataGenerator::getDocumentExtensions() {
    return {"pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "ods", "odp"};
}

QStringList TestDataGenerator::getMediaExtensions() {
    return {"mp4", "avi", "mkv", "mov", "wmv", "mp3", "wav", "flac", "ogg", "aac"};
}