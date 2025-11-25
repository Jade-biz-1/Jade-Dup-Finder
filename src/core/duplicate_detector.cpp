#include "duplicate_detector.h"
#include "detection_algorithm_factory.h"
#include "logger.h"

#include <QDir>
#include <QFileInfo>
#include <QMutexLocker>
#include <QCoreApplication>
#include <QEventLoop>
#include <QtAlgorithms>
#include <QRegularExpression>
#include <QTimer>
#include <memory>

// DuplicateDetector Implementation - Phase 1 with Size-Based Pre-filtering

DuplicateDetector::DuplicateDetector(QObject* parent)
    : QObject(parent)
    , m_isDetecting(false)
    , m_cancelRequested(false)
{
    Logger::instance()->debug(LogCategories::DUPLICATE, "DuplicateDetector created");
    
    // Register meta types for signals
    qRegisterMetaType<DuplicateDetector::FileInfo>("DuplicateDetector::FileInfo");
    qRegisterMetaType<DuplicateDetector::DuplicateGroup>("DuplicateDetector::DuplicateGroup");
    qRegisterMetaType<DuplicateDetector::DetectionProgress>("DuplicateDetector::DetectionProgress");
    qRegisterMetaType<DuplicateDetector::DetectionStatistics>("DuplicateDetector::DetectionStatistics");
    
    LOG_DEBUG(LogCategories::DUPLICATE, "DuplicateDetector initialized with size-based pre-filtering");
}

DuplicateDetector::~DuplicateDetector()
{
    // Cancel any ongoing detection
    cancelDetection();
    
    LOG_DEBUG(LogCategories::DUPLICATE, "DuplicateDetector destroyed");
}

void DuplicateDetector::setOptions(const DetectionOptions& options)
{
    QMutexLocker locker(&m_mutex);
    m_options = options;
    LOG_DEBUG(LogCategories::DUPLICATE, QString("Options updated - Level: %1, GroupBySize: %2")
              .arg(static_cast<int>(options.level))
              .arg(options.groupBySize));
}

DuplicateDetector::DetectionOptions DuplicateDetector::getOptions() const
{
    QMutexLocker locker(&m_mutex);
    return m_options;
}

void DuplicateDetector::findDuplicates(const QList<FileInfo>& files)
{
    QMutexLocker locker(&m_mutex);
    
    if (m_isDetecting) {
        Logger::instance()->warning(LogCategories::DUPLICATE, "Detection already in progress, ignoring new request");
        return;
    }
    
    Logger::instance()->info(LogCategories::DUPLICATE, QString("Starting duplicate detection for %1 files").arg(files.size()));
    
    // Initialize detection state
    m_inputFiles = files;
    m_duplicateGroups.clear();
    m_sizeGroups.clear();
    m_pendingHashes.clear();
    m_hashGroups.clear();  // Clear incremental hash groups
    m_isDetecting = true;
    m_cancelRequested = false;
    
    // Reset statistics
    m_statistics = DetectionStatistics();
    m_detectionStartTime = QTime::currentTime();
    
    // Initialize progress
    m_progress = DetectionProgress();
    m_progress.totalFiles = static_cast<int>(files.size());
    
    emit detectionStarted(static_cast<int>(files.size()));
    
    // Start with size-based pre-filtering (DD-001)
    locker.unlock();
    
    // Filter files based on options
    QList<FileInfo> filteredFiles;
    for (const FileInfo& file : files) {
        if (shouldIncludeFile(file)) {
            filteredFiles.append(file);
        }
    }
    
    Logger::instance()->info(LogCategories::DUPLICATE, QString("Filtered to %1 files for processing").arg(filteredFiles.size()));
    
    if (filteredFiles.isEmpty()) {
        Logger::instance()->warning(LogCategories::DUPLICATE, "No files to process after filtering");
        locker.relock();
        m_isDetecting = false;
        locker.unlock();
        emit detectionCompleted(0);
        return;
    }
    
    // Phase 1: Group files by size
    updateProgress(DetectionProgress::SizeGrouping, 0, static_cast<int>(filteredFiles.size()));
    
    QHash<qint64, QList<FileInfo>> sizeGroups = groupFilesBySize(filteredFiles);
    
    locker.relock();
    m_sizeGroups = sizeGroups;
    m_statistics.totalFilesProcessed = static_cast<int>(filteredFiles.size());
    locker.unlock();
    
    // Process size groups
    processSizeGroups(sizeGroups);
}

void DuplicateDetector::findDuplicates(const QList<FileScanner::FileInfo>& scanResults)
{
    // Convert FileScanner results to DuplicateDetector format
    QList<FileInfo> files;
    files.reserve(scanResults.size());
    
    for (const FileScanner::FileInfo& scanInfo : scanResults) {
        files.append(FileInfo::fromScannerInfo(scanInfo));
    }
    
    findDuplicates(files);
}

QList<DuplicateDetector::DuplicateGroup> DuplicateDetector::findDuplicatesSync(const QList<FileInfo>& files)
{
    // Synchronous version - blocks until complete
    LOG_INFO(LogCategories::DUPLICATE, QString("Starting synchronous duplicate detection for %1 files").arg(files.size()));
    
    QList<DuplicateGroup> result;
    
    if (files.isEmpty()) {
        LOG_WARNING(LogCategories::DUPLICATE, "No files to process");
        return result;
    }
    
    // Filter files based on options
    QList<FileInfo> filteredFiles;
    for (const FileInfo& file : files) {
        if (shouldIncludeFile(file)) {
            filteredFiles.append(file);
        }
    }
    
    LOG_INFO(LogCategories::DUPLICATE, QString("Filtered to %1 files for processing").arg(filteredFiles.size()));
    
    if (filteredFiles.isEmpty()) {
        LOG_WARNING(LogCategories::DUPLICATE, "No files passed filtering");
        return result;
    }
    
    // Phase 1: Group files by size
    LOG_DEBUG(LogCategories::DUPLICATE, "Phase 1 - Grouping by size");
    QHash<qint64, QList<FileInfo>> sizeGroups;
    for (const FileInfo& file : filteredFiles) {
        sizeGroups[file.fileSize].append(file);
    }
    
    LOG_DEBUG(LogCategories::DUPLICATE, QString("Created %1 size groups").arg(sizeGroups.size()));
    
    // Phase 2: Extract files with duplicate sizes (potential duplicates)
    QList<FileInfo> potentialDuplicates;
    for (auto it = sizeGroups.begin(); it != sizeGroups.end(); ++it) {
        if (it.value().size() > 1) {  // Only groups with multiple files
            potentialDuplicates.append(it.value());
        }
    }
    
    LOG_INFO(LogCategories::DUPLICATE, QString("Found %1 potential duplicate files").arg(potentialDuplicates.size()));
    
    if (potentialDuplicates.isEmpty()) {
        LOG_INFO(LogCategories::DUPLICATE, "No potential duplicates found");
        return result;
    }
    
    // Phase 3: Calculate signatures using selected algorithm
    LOG_DEBUG(LogCategories::DUPLICATE, "Phase 2 - Calculating signatures with selected algorithm");
    
    // Create algorithm instance
    auto algorithm = DetectionAlgorithmFactory::create(m_options.algorithmType);
    if (!algorithm) {
        LOG_ERROR(LogCategories::DUPLICATE, "Failed to create detection algorithm");
        return result;
    }
    
    // Configure algorithm
    algorithm->setConfiguration(m_options.algorithmConfig);
    
    LOG_INFO(LogCategories::DUPLICATE, QString("Using algorithm: %1").arg(algorithm->name()));
    
    QList<FileInfo> filesWithSignatures;
    int signaturesCalculated = 0;
    int signatureErrors = 0;
    
    for (FileInfo file : potentialDuplicates) {
        try {
            QByteArray signature = algorithm->computeSignature(file.filePath);
            if (!signature.isEmpty()) {
                file.hash = QString::fromUtf8(signature); // Store signature as hash for compatibility
                filesWithSignatures.append(file);
                signaturesCalculated++;
            } else {
                LOG_WARNING(LogCategories::DUPLICATE, QString("Failed to calculate signature for %1").arg(file.filePath));
                signatureErrors++;
            }
        } catch (...) {
            LOG_WARNING(LogCategories::DUPLICATE, QString("Exception calculating signature for %1").arg(file.filePath));
            signatureErrors++;
        }
    }
    
    LOG_INFO(LogCategories::DUPLICATE, QString("Calculated %1 signatures, %2 errors").arg(signaturesCalculated).arg(signatureErrors));
    
    if (filesWithSignatures.isEmpty()) {
        LOG_WARNING(LogCategories::DUPLICATE, "No signatures calculated successfully");
        return result;
    }
    
    // Phase 4: Group files by signature (using similarity comparison for advanced algorithms)
    LOG_DEBUG(LogCategories::DUPLICATE, "Phase 3 - Grouping by signature similarity");
    QHash<QString, QList<FileInfo>> signatureGroups;
    
    // For exact algorithms, use direct signature matching
    if (m_options.algorithmType == DetectionAlgorithmFactory::ExactHash) {
        for (const FileInfo& file : filesWithSignatures) {
            if (!file.hash.isEmpty()) {
                signatureGroups[file.hash].append(file);
            }
        }
    } else {
        // For similarity algorithms, use similarity comparison
        QList<FileInfo> processedFiles;
        
        for (const FileInfo& file : filesWithSignatures) {
            if (file.hash.isEmpty()) continue;
            
            bool foundGroup = false;
            QByteArray fileSignature = file.hash.toUtf8();
            
            // Check against existing groups
            for (auto it = signatureGroups.begin(); it != signatureGroups.end(); ++it) {
                if (!it.value().isEmpty()) {
                    QByteArray groupSignature = it.value().first().hash.toUtf8();
                    double similarity = algorithm->similarityScore(fileSignature, groupSignature);
                    
                    if (similarity >= m_options.similarityThreshold) {
                        it.value().append(file);
                        foundGroup = true;
                        break;
                    }
                }
            }
            
            // Create new group if no similar group found
            if (!foundGroup) {
                signatureGroups[file.hash].append(file);
            }
        }
    }
    
    LOG_DEBUG(LogCategories::DUPLICATE, QString("Created %1 signature groups").arg(signatureGroups.size()));
    
    // Phase 5: Create duplicate groups (only groups with multiple files)
    LOG_DEBUG(LogCategories::DUPLICATE, "Phase 4 - Creating duplicate groups");
    for (auto it = signatureGroups.begin(); it != signatureGroups.end(); ++it) {
        const QList<FileInfo>& groupFiles = it.value();
        
        if (groupFiles.size() > 1) {  // Only actual duplicates
            DuplicateGroup group = createGroup(groupFiles);
            group.hash = it.key();
            group.algorithmUsed = algorithm->name();
            
            // Calculate average similarity score for the group
            if (m_options.algorithmType != DetectionAlgorithmFactory::ExactHash && groupFiles.size() > 1) {
                double totalSimilarity = 0.0;
                int comparisons = 0;
                
                for (int i = 0; i < groupFiles.size() - 1; ++i) {
                    for (int j = i + 1; j < groupFiles.size(); ++j) {
                        QByteArray sig1 = groupFiles[i].hash.toUtf8();
                        QByteArray sig2 = groupFiles[j].hash.toUtf8();
                        totalSimilarity += algorithm->similarityScore(sig1, sig2);
                        comparisons++;
                    }
                }
                
                group.similarityScore = comparisons > 0 ? totalSimilarity / comparisons : 1.0;
            } else {
                group.similarityScore = 1.0; // Exact match
            }
            
            result.append(group);
        }
    }
    
    // Sort groups by wasted space (largest first)
    std::sort(result.begin(), result.end(), [](const DuplicateGroup& a, const DuplicateGroup& b) {
        return a.wastedSpace > b.wastedSpace;
    });
    
    // Phase 6: Generate recommendations
    if (!result.isEmpty() && m_options.level >= DetectionLevel::Standard) {
        LOG_DEBUG(LogCategories::DUPLICATE, "Phase 5 - Generating recommendations");
        generateRecommendations(result);
    }
    
    LOG_INFO(LogCategories::DUPLICATE, QString("Synchronous detection complete - Found %1 duplicate groups").arg(result.size()));
    
    return result;
}

void DuplicateDetector::cancelDetection()
{
    QMutexLocker locker(&m_mutex);
    if (!m_isDetecting) {
        Logger::instance()->debug(LogCategories::DUPLICATE, "Cancel requested but no detection in progress");
        return;
    }
    
    Logger::instance()->info(LogCategories::DUPLICATE, "Cancelling duplicate detection");
    m_cancelRequested = true;
    
    m_isDetecting = false;
    locker.unlock();
    
    emit detectionCancelled();
}

bool DuplicateDetector::isDetecting() const
{
    QMutexLocker locker(&m_mutex);
    return m_isDetecting;
}

QList<DuplicateDetector::DuplicateGroup> DuplicateDetector::getDuplicateGroups() const
{
    QMutexLocker locker(&m_mutex);
    return m_duplicateGroups;
}

int DuplicateDetector::getTotalDuplicateGroups() const
{
    QMutexLocker locker(&m_mutex);
    return static_cast<int>(m_duplicateGroups.size());
}

qint64 DuplicateDetector::getTotalWastedSpace() const
{
    QMutexLocker locker(&m_mutex);
    
    qint64 totalWasted = 0;
    for (const DuplicateGroup& group : m_duplicateGroups) {
        totalWasted += group.wastedSpace;
    }
    return totalWasted;
}

DuplicateDetector::DetectionStatistics DuplicateDetector::getStatistics() const
{
    QMutexLocker locker(&m_mutex);
    return m_statistics;
}

void DuplicateDetector::resetStatistics()
{
    QMutexLocker locker(&m_mutex);
    m_statistics = DetectionStatistics();
    LOG_DEBUG(LogCategories::DUPLICATE, "Statistics reset");
}

// DD-001: Size-Based Pre-filtering Implementation

QHash<qint64, QList<DuplicateDetector::FileInfo>> DuplicateDetector::groupFilesBySize(const QList<FileInfo>& files)
{
    LOG_DEBUG(LogCategories::DUPLICATE, QString("Grouping %1 files by size").arg(files.size()));
    
    QHash<qint64, QList<FileInfo>> sizeGroups;
    
    int processed = 0;
    for (const FileInfo& file : files) {
        if (m_cancelRequested) {
            LOG_INFO(LogCategories::DUPLICATE, "Size grouping cancelled");
            return QHash<qint64, QList<FileInfo>>();
        }
        
        sizeGroups[file.fileSize].append(file);
        processed++;
        
        // Update progress periodically
        if (processed % 1000 == 0) {
            updateProgress(DetectionProgress::SizeGrouping, processed, static_cast<int>(files.size()), file.filePath, file.fileSize);
        }
    }
    
    // Final progress update
    updateProgress(DetectionProgress::SizeGrouping, static_cast<int>(files.size()), static_cast<int>(files.size()));
    
    LOG_DEBUG(LogCategories::DUPLICATE, QString("Created %1 size groups").arg(sizeGroups.size()));
    
    // Update statistics
    {
        QMutexLocker locker(&m_mutex);
        
        // Count files with unique sizes (no duplicates possible)
        int filesWithUniqueSize = 0;
        int filesInSizeGroups = 0;
        
        for (auto it = sizeGroups.begin(); it != sizeGroups.end(); ++it) {
            if (it.value().size() == 1) {
                filesWithUniqueSize++;
            } else {
                filesInSizeGroups += static_cast<int>(it.value().size());
            }
        }
        
        m_statistics.filesWithUniqueSize = filesWithUniqueSize;
        m_statistics.filesInSizeGroups = filesInSizeGroups;
        
        LOG_INFO(LogCategories::DUPLICATE, QString("Size analysis - Unique sizes: %1, Potential duplicates: %2")
                 .arg(filesWithUniqueSize)
                 .arg(filesInSizeGroups));
    }
    
    return sizeGroups;
}

QList<QList<DuplicateDetector::FileInfo>> DuplicateDetector::getFilesWithDuplicateSizes(const QHash<qint64, QList<FileInfo>>& sizeGroups)
{
    QList<QList<FileInfo>> duplicateSizeGroups;
    
    for (auto it = sizeGroups.begin(); it != sizeGroups.end(); ++it) {
        if (it.value().size() > 1) {  // Only groups with potential duplicates
            duplicateSizeGroups.append(it.value());
        }
    }
    
    // Sort by group size (largest groups first for better progress feedback)
    std::sort(duplicateSizeGroups.begin(), duplicateSizeGroups.end(), 
              [](const QList<FileInfo>& a, const QList<FileInfo>& b) {
                  return a.size() > b.size();
              });
    
    return duplicateSizeGroups;
}

void DuplicateDetector::processSizeGroups(const QHash<qint64, QList<FileInfo>>& sizeGroups)
{
    // Get only groups with multiple files (potential duplicates)
    QList<QList<FileInfo>> duplicateSizeGroups = getFilesWithDuplicateSizes(sizeGroups);
    
    if (duplicateSizeGroups.isEmpty()) {
        LOG_INFO(LogCategories::DUPLICATE, "No potential duplicates found (all files have unique sizes)");
        QMutexLocker locker(&m_mutex);
        m_isDetecting = false;
        locker.unlock();
        emit detectionCompleted(0);
        return;
    }
    
    LOG_INFO(LogCategories::DUPLICATE, QString("Found %1 size groups with potential duplicates").arg(duplicateSizeGroups.size()));
    
    // If using Quick detection level, we only use size-based grouping
    if (m_options.level == DetectionLevel::Quick) {
        LOG_DEBUG(LogCategories::DUPLICATE, "Quick mode - creating groups based on size only");
        
        QList<DuplicateGroup> groups;
        for (const QList<FileInfo>& sizeGroup : duplicateSizeGroups) {
            if (sizeGroup.size() > 1) {
                DuplicateGroup group = createGroup(sizeGroup);
                group.hash = "(size-based)";  // No actual hash calculated
                groups.append(group);
                emit duplicateGroupFound(group);
            }
        }
        
        {
            QMutexLocker locker(&m_mutex);
            m_duplicateGroups = groups;
            m_statistics.duplicateGroupsFound = static_cast<int>(groups.size());
            m_statistics.detectionTime = QTime::fromMSecsSinceStartOfDay(m_detectionStartTime.msecsTo(QTime::currentTime()));
            m_isDetecting = false;
        }
        
        emit detectionCompleted(static_cast<int>(groups.size()));
        return;
    }
    
    // For Standard/Deep/Media modes, proceed with hash calculation
    LOG_DEBUG(LogCategories::DUPLICATE, "Proceeding to hash calculation phase");
    
    // Collect all files that need hash calculation
    QList<FileInfo> filesToHash;
    for (const QList<FileInfo>& sizeGroup : duplicateSizeGroups) {
        filesToHash.append(sizeGroup);
    }
    
    calculateHashesForFiles(filesToHash);
}

void DuplicateDetector::calculateHashesForFiles(const QList<FileInfo>& files)
{
    if (files.isEmpty()) {
        onAllSignaturesComplete();
        return;
    }

    LOG_INFO(LogCategories::DUPLICATE, QString("Starting signature calculation for %1 files using algorithm: %2")
             .arg(files.size())
             .arg(static_cast<int>(m_options.algorithmType)));

    updateProgress(DetectionProgress::HashCalculation, 0, static_cast<int>(files.size()));

    // Create algorithm instance
    auto algorithm = DetectionAlgorithmFactory::create(m_options.algorithmType);
    if (!algorithm) {
        LOG_ERROR(LogCategories::DUPLICATE, "Failed to create detection algorithm for async processing");
        onAllSignaturesComplete();
        return;
    }

    // Configure algorithm
    algorithm->setConfiguration(m_options.algorithmConfig);

    // Prepare pending signatures tracking
    {
        QMutexLocker locker(&m_mutex);
        m_pendingHashes.clear();
        for (const FileInfo& file : files) {
            m_pendingHashes[file.filePath] = file;
        }
        m_currentAlgorithm = std::move(algorithm);
    }

    // Calculate signatures synchronously in batches to avoid blocking
    QTimer::singleShot(0, this, [this, files]() {
        calculateSignaturesBatch(files, 0);
    });
}

void DuplicateDetector::calculateSignaturesBatch(const QList<FileInfo>& files, int startIndex)
{
    // PERFORMANCE FIX: Increase batch size dramatically for better throughput
    const int batchSize = 500; // Process 500 files at a time (was 5!)
    int processed = 0;

    // Process batch - minimize mutex contention
    for (int i = startIndex; i < files.size() && processed < batchSize; ++i, ++processed) {
        // Check cancellation without holding lock
        {
            QMutexLocker locker(&m_mutex);
            if (m_cancelRequested || !m_isDetecting || !m_currentAlgorithm) {
                return;
            }
        }

        const FileInfo& file = files[i];
        QByteArray signature;

        try {
            // Calculate hash WITHOUT holding mutex - this is the slow operation
            {
                QMutexLocker locker(&m_mutex);
                if (!m_currentAlgorithm) {
                    return;
                }
            }
            signature = m_currentAlgorithm->computeSignature(file.filePath);
        } catch (...) {
            LOG_WARNING(LogCategories::DUPLICATE, QString("Exception calculating signature for %1").arg(file.filePath));
            signature = QByteArray(); // Empty signature on error
        }

        // Update internal state - lock only for the update
        {
            QMutexLocker locker(&m_mutex);
            auto it = m_pendingHashes.find(file.filePath);
            if (it != m_pendingHashes.end()) {
                if (!signature.isEmpty()) {
                    QString hashStr = QString::fromUtf8(signature);
                    it.value().hash = hashStr;
                    m_statistics.hashCalculationsPerformed++;
                    
                    // PERFORMANCE: Build hash groups incrementally as hashes are computed
                    // This distributes the grouping work instead of doing it all at once
                    m_hashGroups[hashStr].append(it.value());
                } else {
                    LOG_WARNING(LogCategories::DUPLICATE, QString("Failed to calculate signature for %1").arg(file.filePath));
                    m_pendingHashes.erase(it);
                }
            }
        }

        // PERFORMANCE FIX: Update progress only occasionally, not for every file
        // Update every 100 files to reduce cross-thread signal overhead
        if (processed % 100 == 0 || processed == batchSize - 1) {
            int completed = startIndex + processed + 1;
            int total = static_cast<int>(files.size());
            updateProgress(DetectionProgress::HashCalculation, completed, total, file.filePath, file.fileSize);
        }
    }

    int nextIndex = startIndex + processed;

    // Continue with next batch or finish
    if (nextIndex < files.size()) {
        // PERFORMANCE FIX: Remove artificial delay - process immediately for maximum throughput
        // Background thread ensures UI stays responsive without needing delays
        QTimer::singleShot(0, this, [this, files, nextIndex]() {
            calculateSignaturesBatch(files, nextIndex);
        });
    } else {
        // All signatures calculated
        onAllSignaturesComplete();
    }
}


void DuplicateDetector::onAllSignaturesComplete()
{
    QMutexLocker locker(&m_mutex);
    
    if (m_cancelRequested || !m_isDetecting) {
        return;
    }
    
    Logger::instance()->info(LogCategories::DUPLICATE, "All signatures calculated, proceeding to duplicate grouping");
    
    // Collect all files with successful signatures
    QList<FileInfo> filesWithSignatures;
    for (auto it = m_pendingHashes.begin(); it != m_pendingHashes.end(); ++it) {
        if (!it.value().hash.isEmpty()) {
            filesWithSignatures.append(it.value());
        }
    }
    
    locker.unlock();
    
    processHashResults(filesWithSignatures);
}

void DuplicateDetector::processHashResults(const QList<FileInfo>& filesWithSignatures)
{
    if (filesWithSignatures.isEmpty()) {
        Logger::instance()->warning(LogCategories::DUPLICATE, "No files with valid signatures");
        QMutexLocker locker(&m_mutex);
        m_isDetecting = false;
        locker.unlock();
        emit detectionCompleted(0);
        return;
    }
    
    Logger::instance()->info(LogCategories::DUPLICATE, QString("Processing signature results for %1 files").arg(filesWithSignatures.size()));
    
    updateProgress(DetectionProgress::DuplicateGrouping, 0, static_cast<int>(filesWithSignatures.size()));
    
    // PERFORMANCE: Use pre-built hash groups instead of building from scratch
    // Groups were built incrementally during hash calculation
    QHash<QString, QList<FileInfo>> signatureGroups;
    {
        QMutexLocker locker(&m_mutex);
        signatureGroups = m_hashGroups;
        // Clear for next detection
        m_hashGroups.clear();
    }
    
    Logger::instance()->info(LogCategories::DUPLICATE, QString("Using %1 pre-built hash groups").arg(signatureGroups.size()));
    
    // Create duplicate groups (now much faster since grouping is already done)
    QList<DuplicateGroup> duplicateGroups = createDuplicateGroups(signatureGroups);
    
    if (!duplicateGroups.isEmpty()) {
        Logger::instance()->info(LogCategories::DUPLICATE, QString("Generating recommendations for %1 groups").arg(duplicateGroups.size()));
        updateProgress(DetectionProgress::GeneratingRecommendations, 0, static_cast<int>(duplicateGroups.size()));
        generateRecommendations(duplicateGroups);
    }
    
    // Finalize detection
    {
        QMutexLocker locker(&m_mutex);
        m_duplicateGroups = duplicateGroups;
        m_statistics.duplicateGroupsFound = static_cast<int>(duplicateGroups.size());
        
        int totalDuplicateFiles = 0;
        qint64 totalWastedSpace = 0;
        
        for (const DuplicateGroup& group : duplicateGroups) {
            totalDuplicateFiles += group.fileCount;
            totalWastedSpace += group.wastedSpace;
        }
        
        m_statistics.totalDuplicateFiles = totalDuplicateFiles;
        m_statistics.totalWastedSpace = totalWastedSpace;
        m_statistics.averageGroupSize = duplicateGroups.isEmpty() ? 0.0 : 
                                       static_cast<double>(totalDuplicateFiles) / static_cast<double>(duplicateGroups.size());
        
        m_statistics.detectionTime = QTime::fromMSecsSinceStartOfDay(m_detectionStartTime.msecsTo(QTime::currentTime()));
        m_isDetecting = false;
    }
    
    updateProgress(DetectionProgress::Complete, static_cast<int>(duplicateGroups.size()), static_cast<int>(duplicateGroups.size()));
    
    Logger::instance()->info(LogCategories::DUPLICATE, QString("Detection completed - %1 duplicate groups found").arg(duplicateGroups.size()));
    emit detectionCompleted(static_cast<int>(duplicateGroups.size()));
}

QHash<QString, QList<DuplicateDetector::FileInfo>> DuplicateDetector::groupFilesByHash(const QList<FileInfo>& files)
{
    QHash<QString, QList<FileInfo>> hashGroups;
    
    for (const FileInfo& file : files) {
        if (!file.hash.isEmpty()) {
            hashGroups[file.hash].append(file);
        }
    }
    
    return hashGroups;
}

QHash<QString, QList<DuplicateDetector::FileInfo>> DuplicateDetector::groupFilesBySignatureSimilarity(const QList<FileInfo>& files)
{
    QHash<QString, QList<FileInfo>> signatureGroups;
    
    QMutexLocker locker(&m_mutex);
    
    // For exact algorithms, use direct signature matching
    if (m_options.algorithmType == DetectionAlgorithmFactory::ExactHash || !m_currentAlgorithm) {
        locker.unlock();
        return groupFilesByHash(files);
    }
    
    // For similarity algorithms, use similarity comparison
    locker.unlock();
    
    for (const FileInfo& file : files) {
        if (file.hash.isEmpty()) continue;
        
        bool foundGroup = false;
        QByteArray fileSignature = file.hash.toUtf8();
        
        // Check against existing groups
        for (auto it = signatureGroups.begin(); it != signatureGroups.end(); ++it) {
            if (!it.value().isEmpty()) {
                QByteArray groupSignature = it.value().first().hash.toUtf8();
                
                QMutexLocker algorithmLocker(&m_mutex);
                if (m_currentAlgorithm) {
                    double similarity = m_currentAlgorithm->similarityScore(fileSignature, groupSignature);
                    algorithmLocker.unlock();
                    
                    if (similarity >= m_options.similarityThreshold) {
                        it.value().append(file);
                        foundGroup = true;
                        break;
                    }
                } else {
                    algorithmLocker.unlock();
                    break;
                }
            }
        }
        
        // Create new group if no similar group found
        if (!foundGroup) {
            signatureGroups[file.hash].append(file);
        }
    }
    
    return signatureGroups;
}

QList<DuplicateDetector::DuplicateGroup> DuplicateDetector::createDuplicateGroups(const QHash<QString, QList<FileInfo>>& hashGroups)
{
    QList<DuplicateGroup> groups;
    
    LOG_DEBUG(LogCategories::DUPLICATE, QString("Processing %1 unique hashes").arg(hashGroups.size()));
    
    int processed = 0;
    for (auto it = hashGroups.begin(); it != hashGroups.end(); ++it) {
        const QList<FileInfo>& files = it.value();
        
        LOG_DEBUG(LogCategories::DUPLICATE, QString("Hash: %1 has %2 files").arg(it.key()).arg(files.size()));
        
        if (files.size() > 1) {  // Only actual duplicates (multiple files with same signature)
            DuplicateGroup group = createGroup(files);
            group.hash = it.key();
            
            // Set algorithm information
            QMutexLocker locker(&m_mutex);
            if (m_currentAlgorithm) {
                group.algorithmUsed = m_currentAlgorithm->name();
                
                // Calculate average similarity score for the group
                if (m_options.algorithmType != DetectionAlgorithmFactory::ExactHash && files.size() > 1) {
                    double totalSimilarity = 0.0;
                    int comparisons = 0;
                    
                    // PERFORMANCE: For large groups, limit similarity calculations to avoid O(n²) explosion
                    // For groups > 100 files, sample only first 100 files for similarity
                    int maxFilesForSimilarity = qMin(100, files.size());
                    
                    for (int i = 0; i < maxFilesForSimilarity - 1; ++i) {
                        for (int j = i + 1; j < maxFilesForSimilarity; ++j) {
                            QByteArray sig1 = files[i].hash.toUtf8();
                            QByteArray sig2 = files[j].hash.toUtf8();
                            totalSimilarity += m_currentAlgorithm->similarityScore(sig1, sig2);
                            comparisons++;
                            
                            // Process events every 100 comparisons to keep UI responsive
                            if (comparisons % 100 == 0) {
                                locker.unlock();
                                QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
                                locker.relock();
                            }
                        }
                    }
                    
                    group.similarityScore = comparisons > 0 ? totalSimilarity / comparisons : 1.0;
                } else {
                    group.similarityScore = 1.0; // Exact match
                }
            } else {
                group.algorithmUsed = "Unknown";
                group.similarityScore = 1.0;
            }
            locker.unlock();
            
            groups.append(group);
            
            LOG_DEBUG(LogCategories::DUPLICATE, QString("Created duplicate group %1 with %2 files using %3 (similarity: %4)")
                     .arg(groups.size()).arg(files.size()).arg(group.algorithmUsed).arg(group.similarityScore));
            
            emit duplicateGroupFound(group);
        } else {
            LOG_DEBUG(LogCategories::DUPLICATE, "Skipping - only 1 file with this signature (not a duplicate)");
        }
        
        processed++;
        if (processed % 10 == 0) {
            updateProgress(DetectionProgress::DuplicateGrouping, processed, static_cast<int>(hashGroups.size()));
            // Process events to keep UI responsive
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        }
    }
    
    // Sort groups by wasted space (largest first)
    std::sort(groups.begin(), groups.end(), [](const DuplicateGroup& a, const DuplicateGroup& b) {
        return a.wastedSpace > b.wastedSpace;
    });
    
    return groups;
}

DuplicateDetector::DuplicateGroup DuplicateDetector::createGroup(const QList<FileInfo>& duplicateFiles)
{
    DuplicateGroup group;
    
    if (duplicateFiles.isEmpty()) {
        return group;
    }
    
    group.files = duplicateFiles;
    group.fileSize = duplicateFiles.first().fileSize;  // All files have same size
    group.fileCount = static_cast<int>(duplicateFiles.size());
    group.totalSize = group.fileSize * group.fileCount;
    group.wastedSpace = calculateWastedSpace(duplicateFiles);
    
    return group;
}

qint64 DuplicateDetector::calculateWastedSpace(const QList<FileInfo>& files)
{
    if (files.size() <= 1) {
        return 0;
    }
    
    // Wasted space = (number of files - 1) * file size
    // We keep one file, so the rest is wasted space
    return static_cast<qint64>(files.size() - 1) * files.first().fileSize;
}

// Smart Recommendations (Basic Implementation)

void DuplicateDetector::generateRecommendations(QList<DuplicateGroup>& groups)
{
    int processed = 0;
    for (DuplicateGroup& group : groups) {
        if (m_cancelRequested) {
            return;
        }
        
        QList<FileScore> scores = scoreFiles(group.files);
        group.recommendedAction = generateRecommendationText(scores);
        
        processed++;
        if (processed % 5 == 0) {
            updateProgress(DetectionProgress::GeneratingRecommendations, processed, static_cast<int>(groups.size()));
        }
    }
}

QList<DuplicateDetector::FileScore> DuplicateDetector::scoreFiles(const QList<FileInfo>& files)
{
    QList<FileScore> scores;
    
    for (const FileInfo& file : files) {
        FileScore score = calculateFileScore(file);
        scores.append(score);
    }
    
    // Sort by score (highest first)
    std::sort(scores.begin(), scores.end(), [](const FileScore& a, const FileScore& b) {
        return a.score > b.score;
    });
    
    return scores;
}

DuplicateDetector::FileScore DuplicateDetector::calculateFileScore(const FileInfo& file)
{
    FileScore score(file);
    
    // Basic scoring algorithm
    double locationScore = getLocationScore(file.filePath);
    double ageScore = getAgeScore(file.lastModified);
    double nameScore = getNameScore(file.fileName);
    double pathScore = getPathScore(file.filePath);
    
    // Weight the different factors
    score.score = (locationScore * 0.4) + (ageScore * 0.2) + (nameScore * 0.2) + (pathScore * 0.2);
    
    // Store individual factor scores
    score.factorScores[FileScore::LocationScore] = locationScore;
    score.factorScores[FileScore::AgeScore] = ageScore;
    score.factorScores[FileScore::NameScore] = nameScore;
    score.factorScores[FileScore::PathScore] = pathScore;
    
    // Generate reasoning
    QStringList reasons;
    if (locationScore > 70) reasons << "good location";
    if (ageScore > 70) reasons << "recent file";
    if (nameScore > 70) reasons << "clean filename";
    if (pathScore > 70) reasons << "organized path";
    
    score.reasoning = reasons.isEmpty() ? "Standard file" : reasons.join(", ");
    
    return score;
}

QString DuplicateDetector::generateRecommendationText(const QList<FileScore>& scores)
{
    if (scores.isEmpty()) {
        return "No recommendation";
    }
    
    const FileScore& best = scores.first();
    QFileInfo fileInfo(best.file.filePath);
    
    return QString("Keep: %1 (%2)").arg(fileInfo.fileName(), best.reasoning);
}

// Scoring Methods (Basic Implementation)

double DuplicateDetector::getLocationScore(const QString& filePath)
{
    // Simple location scoring based on common directory patterns
    QString lowerPath = filePath.toLower();
    
    if (lowerPath.contains("/documents/") || lowerPath.contains("/pictures/") || 
        lowerPath.contains("/music/") || lowerPath.contains("/videos/")) {
        return 90.0;  // High score for user directories
    }
    
    if (lowerPath.contains("/desktop/")) {
        return 75.0;  // Medium-high score for desktop
    }
    
    if (lowerPath.contains("/downloads/") || lowerPath.contains("/tmp/") || 
        lowerPath.contains("/temp/")) {
        return 30.0;  // Low score for temporary locations
    }
    
    return 50.0;  // Default score
}

double DuplicateDetector::getAgeScore(const QDateTime& lastModified)
{
    if (!lastModified.isValid()) {
        return 50.0;
    }
    
    // Score based on how recent the file is (newer = higher score)
    qint64 daysOld = lastModified.daysTo(QDateTime::currentDateTime());
    
    if (daysOld < 7) return 90.0;      // Very recent
    if (daysOld < 30) return 75.0;     // Recent
    if (daysOld < 365) return 60.0;    // Moderately old
    if (daysOld < 1095) return 40.0;   // Old (3 years)
    
    return 20.0;  // Very old
}

double DuplicateDetector::getNameScore(const QString& fileName)
{
    // Score based on filename quality
    QString lowerName = fileName.toLower();
    
    // Penalize obviously temporary or low-quality names
    if (lowerName.contains("copy") || lowerName.contains("temp") || 
        lowerName.contains("tmp") || lowerName.contains("backup")) {
        return 30.0;
    }
    
    // Penalize numbered duplicates (file (1).txt, file (2).txt)
    QRegularExpression numberedCopy(R"(\s\(\d+\))");
    if (lowerName.contains(numberedCopy)) {
        return 40.0;
    }
    
    // Reward clean, descriptive names
    if (lowerName.length() > 3 && lowerName.length() < 50 && 
        !lowerName.contains("untitled") && !lowerName.startsWith("new ")) {
        return 80.0;
    }
    
    return 60.0;  // Default score
}

double DuplicateDetector::getAccessScore(const QDateTime& lastAccessed)
{
    // Basic implementation - similar to age score but for access time
    return getAgeScore(lastAccessed);
}

double DuplicateDetector::getPathScore(const QString& filePath)
{
    // Score based on path organization and length
    int pathDepth = static_cast<int>(filePath.count('/'));
    
    // Prefer moderately organized paths (not too deep, not too shallow)
    if (pathDepth >= 3 && pathDepth <= 6) {
        return 80.0;
    }
    
    if (pathDepth < 3) {
        return 60.0;  // Too shallow
    }
    
    return 40.0;  // Too deep
}

// Utility Methods

bool DuplicateDetector::shouldIncludeFile(const FileInfo& file) const
{
    // Apply detection options filters
    if (m_options.skipEmptyFiles && file.fileSize == 0) {
        return false;
    }
    
    if (m_options.minimumFileSize > 0 && file.fileSize < m_options.minimumFileSize) {
        return false;
    }
    
    if (m_options.maximumFileSize > 0 && file.fileSize > m_options.maximumFileSize) {
        return false;
    }
    
    if (m_options.skipSystemFiles) {
        QString fileName = QFileInfo(file.filePath).fileName();
        if (fileName.startsWith('.') || fileName.startsWith("$")) {
            return false;
        }
    }
    
    return true;
}

void DuplicateDetector::updateProgress(DetectionProgress::Phase phase, int processed, int total, const QString& currentFile, qint64 currentFileSize)
{
    QMutexLocker locker(&m_mutex);

    m_progress.currentPhase = phase;
    m_progress.filesProcessed = processed;
    m_progress.totalFiles = total;
    m_progress.currentFile = currentFile;
    m_progress.currentFileSize = currentFileSize;
    m_progress.percentComplete = total > 0 ? (static_cast<double>(processed) / total) * 100.0 : 0.0;

    // Update phase-specific counters
    if (phase == DetectionProgress::SizeGrouping) {
        m_progress.sizeGroupsFound = static_cast<int>(m_sizeGroups.size());
    } else if (phase >= DetectionProgress::DuplicateGrouping) {
        m_progress.duplicateGroupsFound = static_cast<int>(m_duplicateGroups.size());

        // Calculate wasted space directly here to avoid recursive mutex lock
        qint64 totalWasted = 0;
        for (const DuplicateGroup& group : m_duplicateGroups) {
            totalWasted += group.wastedSpace;
        }
        m_progress.wastedSpaceFound = totalWasted;
    }

    DetectionProgress progressCopy = m_progress;
    locker.unlock();

    emit detectionProgress(progressCopy);
}

void DuplicateDetector::updateStatistics()
{
    // Statistics are updated throughout the detection process
    // This method can be used for final calculations if needed
}

// Utility Methods

DetectionAlgorithmFactory::AlgorithmType DuplicateDetector::convertDetectionLevel(DetectionLevel level)
{
    switch (level) {
        case DuplicateDetector::DetectionLevel::Quick:
            return DetectionAlgorithmFactory::QuickScan;
        case DuplicateDetector::DetectionLevel::Standard:
            return DetectionAlgorithmFactory::ExactHash;
        case DuplicateDetector::DetectionLevel::Deep:
            return DetectionAlgorithmFactory::DocumentSimilarity;
        case DuplicateDetector::DetectionLevel::Media:
            return DetectionAlgorithmFactory::PerceptualHash;
        default:
            return DetectionAlgorithmFactory::ExactHash;
    }
}

DuplicateDetector::DetectionLevel DuplicateDetector::convertAlgorithmType(DetectionAlgorithmFactory::AlgorithmType type)
{
    switch (type) {
        case DetectionAlgorithmFactory::QuickScan:
            return DetectionLevel::Quick;
        case DetectionAlgorithmFactory::ExactHash:
            return DetectionLevel::Standard;
        case DetectionAlgorithmFactory::DocumentSimilarity:
            return DetectionLevel::Deep;
        case DetectionAlgorithmFactory::PerceptualHash:
            return DetectionLevel::Media;
        default:
            return DuplicateDetector::DetectionLevel::Standard;
    }
}

// Static Methods

DuplicateDetector::FileInfo DuplicateDetector::FileInfo::fromScannerInfo(const FileScanner::FileInfo& scanInfo)
{
    FileInfo info;
    info.filePath = scanInfo.filePath;
    info.fileSize = scanInfo.fileSize;
    info.fileName = scanInfo.fileName;
    info.directory = scanInfo.directory;
    info.lastModified = scanInfo.lastModified;
    // lastAccessed and hash will be filled later if needed
    
    return info;
}