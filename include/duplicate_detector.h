#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QDateTime>
#include <QHash>
#include <QList>
#include <QMutex>
#include <QUuid>

#include "file_scanner.h"
#include "hash_calculator.h"

/**
 * @brief DuplicateDetector - Advanced duplicate file detection engine
 * 
 * Features:
 * - Size-based pre-filtering for performance optimization
 * - Hash-based exact duplicate detection
 * - Smart recommendations for file retention
 * - Memory-efficient processing for large file collections
 * - Progress reporting and cancellation support
 * - Multi-level detection algorithms (Quick/Standard/Deep)
 * 
 * Usage:
 * ```cpp
 * DuplicateDetector detector;
 * connect(&detector, &DuplicateDetector::duplicateGroupFound, this, &MyClass::onDuplicatesFound);
 * detector.findDuplicates(scannedFiles);
 * ```
 */
class DuplicateDetector : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Information about a single file for duplicate detection
     */
    struct FileInfo {
        QString filePath;           // Absolute file path
        qint64 fileSize;           // File size in bytes
        QString fileName;          // Just the filename
        QString directory;         // Directory path
        QDateTime lastModified;    // Last modification time
        QDateTime lastAccessed;    // Last access time (optional)
        QString hash;              // SHA-256 hash (calculated when needed)
        
        // Convenience constructor
        FileInfo() : fileSize(0) {}
        FileInfo(const QString& path, qint64 size) 
            : filePath(path), fileSize(size) {}
        
        // Convert from FileScanner::FileInfo
        static FileInfo fromScannerInfo(const FileScanner::FileInfo& scanInfo);
    };
    
    /**
     * @brief A group of duplicate files
     */
    struct DuplicateGroup {
        QString groupId;             // Unique identifier (UUID)
        QList<FileInfo> files;       // Files in this group
        qint64 fileSize;             // Size of each file (all identical)
        QString hash;                // SHA-256 hash (all identical)
        qint64 totalSize;            // Total size of all files
        qint64 wastedSpace;          // Space that can be saved (totalSize - fileSize)
        int fileCount;               // Number of files in group
        QString recommendedAction;   // "Keep newest", "Keep in Documents", etc.
        QDateTime detected;          // When duplicates were found
        
        // Convenience constructor
        DuplicateGroup() : fileSize(0), totalSize(0), wastedSpace(0), fileCount(0) {
            groupId = QUuid::createUuid().toString(QUuid::WithoutBraces);
            detected = QDateTime::currentDateTime();
        }
    };
    
    /**
     * @brief Detection algorithm levels
     */
    enum class DetectionLevel {
        Quick,      // Size-based only (fastest)
        Standard,   // Size + Hash (recommended)
        Deep,       // Size + Hash + Metadata analysis
        Media       // Specialized for images/videos with EXIF
    };
    
    /**
     * @brief Detection configuration options
     */
    struct DetectionOptions {
        DetectionLevel level = DetectionLevel::Standard;
        bool groupBySize = true;                    // Enable size-based pre-filtering
        bool analyzeMetadata = false;               // Compare file metadata
        bool fuzzyNameMatching = false;             // Enable fuzzy filename matching
        double similarityThreshold = 0.95;         // For near-duplicates (0.0-1.0)
        qint64 minimumFileSize = 0;                // Skip files smaller than this
        qint64 maximumFileSize = -1;               // Skip files larger than this (-1 = no limit)
        bool skipEmptyFiles = true;                 // Skip zero-byte files
        bool skipSystemFiles = true;                // Skip system/hidden files
    };
    
    /**
     * @brief Progress information during detection
     */
    struct DetectionProgress {
        enum Phase {
            SizeGrouping,               // Grouping files by size
            HashCalculation,            // Calculating hashes for potential duplicates  
            DuplicateGrouping,          // Creating final duplicate groups
            GeneratingRecommendations,  // Analyzing and generating recommendations
            Complete                    // Detection finished
        };
        
        Phase currentPhase = SizeGrouping;
        int filesProcessed = 0;
        int totalFiles = 0;
        int sizeGroupsFound = 0;
        int duplicateGroupsFound = 0;
        qint64 wastedSpaceFound = 0;
        QString currentFile;            // Currently processing file
        double percentComplete = 0.0;
        
        DetectionProgress() = default;
    };
    
    /**
     * @brief File scoring for smart recommendations
     */
    struct FileScore {
        FileInfo file;
        double score = 0.0;         // Higher = better to keep (0.0-100.0)
        QString reasoning;          // Why this file is recommended
        
        enum ScoreFactor {
            LocationScore = 0,      // File location quality
            AgeScore,              // Newer files preferred
            NameScore,             // Better filenames preferred
            AccessScore,           // Recently accessed preferred
            PathScore              // Shorter/cleaner paths preferred
        };
        
        QHash<ScoreFactor, double> factorScores;  // Individual factor scores
        
        FileScore() = default;
        FileScore(const FileInfo& f) : file(f) {}
    };

    explicit DuplicateDetector(QObject* parent = nullptr);
    ~DuplicateDetector();
    
    // Configuration
    void setOptions(const DetectionOptions& options);
    DetectionOptions getOptions() const;
    
    // Main detection interface
    void findDuplicates(const QList<FileInfo>& files);
    void findDuplicates(const QList<FileScanner::FileInfo>& scanResults);
    QList<DuplicateGroup> findDuplicatesSync(const QList<FileInfo>& files);
    
    // Operation control
    void cancelDetection();
    bool isDetecting() const;
    
    // Results access
    QList<DuplicateGroup> getDuplicateGroups() const;
    int getTotalDuplicateGroups() const;
    qint64 getTotalWastedSpace() const;
    
    // Statistics and analysis
    struct DetectionStatistics {
        int totalFilesProcessed = 0;
        int filesWithUniqueSize = 0;        // Files skipped (no possible duplicates)
        int filesInSizeGroups = 0;          // Files that need hash calculation
        int hashCalculationsPerformed = 0;
        int duplicateGroupsFound = 0;
        int totalDuplicateFiles = 0;
        qint64 totalWastedSpace = 0;
        QTime detectionTime;
        double averageGroupSize = 0.0;      // Average files per group
        
        DetectionStatistics() = default;
    };
    DetectionStatistics getStatistics() const;
    void resetStatistics();

signals:
    /**
     * @brief Emitted when detection starts
     * @param totalFiles Total number of files to process
     */
    void detectionStarted(int totalFiles);
    
    /**
     * @brief Emitted periodically during detection
     * @param progress Current progress information
     */
    void detectionProgress(const DetectionProgress& progress);
    
    /**
     * @brief Emitted when a duplicate group is found
     * @param group The duplicate group that was found
     */
    void duplicateGroupFound(const DuplicateGroup& group);
    
    /**
     * @brief Emitted when detection completes successfully
     * @param totalGroups Total number of duplicate groups found
     */
    void detectionCompleted(int totalGroups);
    
    /**
     * @brief Emitted when detection is cancelled
     */
    void detectionCancelled();
    
    /**
     * @brief Emitted when an error occurs during detection
     * @param error Error description
     */
    void detectionError(const QString& error);

private slots:
    void onHashCalculated(const HashCalculator::HashResult& result);
    void onAllHashesComplete();
    
private:
    // Phase 1: Size-based pre-filtering
    QHash<qint64, QList<FileInfo>> groupFilesBySize(const QList<FileInfo>& files);
    QList<QList<FileInfo>> getFilesWithDuplicateSizes(const QHash<qint64, QList<FileInfo>>& sizeGroups);
    void processSizeGroups(const QHash<qint64, QList<FileInfo>>& sizeGroups);
    
    // Phase 2: Hash calculation coordination
    void calculateHashesForFiles(const QList<FileInfo>& files);
    void processHashResults(const QList<FileInfo>& filesWithHashes);
    
    // Phase 3: Duplicate grouping
    QHash<QString, QList<FileInfo>> groupFilesByHash(const QList<FileInfo>& files);
    QList<DuplicateGroup> createDuplicateGroups(const QHash<QString, QList<FileInfo>>& hashGroups);
    DuplicateGroup createGroup(const QList<FileInfo>& duplicateFiles);
    
    // Phase 4: Smart recommendations
    void generateRecommendations(QList<DuplicateGroup>& groups);
    QList<FileScore> scoreFiles(const QList<FileInfo>& files);
    FileScore calculateFileScore(const FileInfo& file);
    QString generateRecommendationText(const QList<FileScore>& scores);
    
    // Scoring factors
    double getLocationScore(const QString& filePath);
    double getAgeScore(const QDateTime& lastModified);
    double getNameScore(const QString& fileName);
    double getAccessScore(const QDateTime& lastAccessed);
    double getPathScore(const QString& filePath);
    
    // Progress and statistics
    void updateProgress(DetectionProgress::Phase phase, int processed, int total, const QString& currentFile = QString());
    void updateStatistics();
    
    // Utility methods
    bool shouldIncludeFile(const FileInfo& file) const;
    qint64 calculateWastedSpace(const QList<FileInfo>& files);
    
    // Member variables
    DetectionOptions m_options;
    HashCalculator* m_hashCalculator;
    
    // Detection state
    QList<FileInfo> m_inputFiles;
    QList<DuplicateGroup> m_duplicateGroups;
    QHash<qint64, QList<FileInfo>> m_sizeGroups;
    QHash<QString, FileInfo> m_pendingHashes;  // Files waiting for hash calculation
    
    // Progress tracking
    DetectionProgress m_progress;
    DetectionStatistics m_statistics;
    bool m_isDetecting;
    bool m_cancelRequested;
    mutable QMutex m_mutex;
    
    // Timing
    QTime m_detectionStartTime;
};

// Q_DECLARE_METATYPE for use with Qt's signal/slot system
Q_DECLARE_METATYPE(DuplicateDetector::FileInfo)
Q_DECLARE_METATYPE(DuplicateDetector::DuplicateGroup)
Q_DECLARE_METATYPE(DuplicateDetector::DetectionProgress)
Q_DECLARE_METATYPE(DuplicateDetector::DetectionStatistics)