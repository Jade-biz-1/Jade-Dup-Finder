#include "scan_scope_preview_widget.h"
#include "core/logger.h"
#include "theme_manager.h"
#include <QtCore/QDir>
#include <QtCore/QDirIterator>
#include <QtCore/QFileInfo>
#include <QtCore/QRegularExpression>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QTreeWidgetItem>
#include <QtGui/QIcon>

ScanScopePreviewWidget::ScanScopePreviewWidget(QWidget* parent)
    : QWidget(parent)
    , m_layout(nullptr)
    , m_titleLabel(nullptr)
    , m_folderCountLabel(nullptr)
    , m_fileCountLabel(nullptr)
    , m_sizeLabel(nullptr)
    , m_statusLabel(nullptr)
    , m_pathsTree(nullptr)
    , m_updateTimer(new QTimer(this))
    , m_pendingMaxDepth(-1)
    , m_pendingIncludeHidden(false)
    , m_calculationPending(false)
{
    setupUI();
    
    // Configure debounce timer
    m_updateTimer->setSingleShot(true);
    m_updateTimer->setInterval(UPDATE_DELAY_MS);
    connect(m_updateTimer, &QTimer::timeout, this, &ScanScopePreviewWidget::performCalculation);
    
    // Register with ThemeManager for automatic theme updates
    ThemeManager::instance()->registerCustomWidget(this, "ScanScopePreviewWidget");
}

ScanScopePreviewWidget::~ScanScopePreviewWidget()
{
    // Qt handles cleanup
}

void ScanScopePreviewWidget::setupUI()
{
    m_layout = new QVBoxLayout(this);
    m_layout->setContentsMargins(8, 8, 8, 8);
    m_layout->setSpacing(8);
    
    // Title
    m_titleLabel = new QLabel(tr("📊 Scan Scope Preview"), this);
    m_titleLabel->setStyleSheet(
        "QLabel {"
        "    font-weight: bold;"
        "    font-size: 11pt;"
        "    padding: 4px;"
        "}"
    );
    
    // Statistics labels
    m_folderCountLabel = new QLabel(tr("Folders: -"), this);
    m_fileCountLabel = new QLabel(tr("Estimated Files: -"), this);
    m_sizeLabel = new QLabel(tr("Estimated Size: -"), this);
    
    QString labelStyle = 
        "QLabel {"
        "    padding: 4px 8px;"
        "    background: palette(base);"
        "    border: 1px solid palette(mid);"
        "    border-radius: 3px;"
        "    font-size: 10pt;"
        "}"
    ;
    
    m_folderCountLabel->setStyleSheet(labelStyle);
    m_fileCountLabel->setStyleSheet(labelStyle);
    m_sizeLabel->setStyleSheet(labelStyle);
    
    // Status label
    m_statusLabel = new QLabel(this);
    m_statusLabel->setWordWrap(true);
    m_statusLabel->setVisible(false);
    m_statusLabel->setStyleSheet(
        "QLabel {"
        "    color: palette(mid);"
        "    font-style: italic;"
        "    padding: 4px;"
        "}"
    );
    
    // Paths tree
    m_pathsTree = new QTreeWidget(this);
    m_pathsTree->setHeaderLabels(QStringList() << tr("Path") << tr("Status"));
    m_pathsTree->setAlternatingRowColors(true);
    m_pathsTree->setRootIsDecorated(false);
    m_pathsTree->setMaximumHeight(150);
    m_pathsTree->header()->setStretchLastSection(false);
    m_pathsTree->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    m_pathsTree->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    m_pathsTree->setStyleSheet(
        "QTreeWidget {"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    background: palette(base);"
        "    font-size: 9pt;"
        "}"
        "QTreeWidget::item {"
        "    padding: 2px;"
        "}"
    );
    
    // Add to layout
    m_layout->addWidget(m_titleLabel);
    m_layout->addWidget(m_folderCountLabel);
    m_layout->addWidget(m_fileCountLabel);
    m_layout->addWidget(m_sizeLabel);
    m_layout->addWidget(m_statusLabel);
    m_layout->addWidget(m_pathsTree);
    m_layout->addStretch();
    
    // Enforce minimum sizes for all controls
    ThemeManager::instance()->enforceMinimumSizes(this);
}

void ScanScopePreviewWidget::updatePreview(const QStringList& targetPaths,
                                           const QStringList& excludePatterns,
                                           const QStringList& excludeFolders,
                                           int maxDepth,
                                           bool includeHidden)
{
    // Store pending parameters
    m_pendingTargetPaths = targetPaths;
    m_pendingExcludePatterns = excludePatterns;
    m_pendingExcludeFolders = excludeFolders;
    m_pendingMaxDepth = maxDepth;
    m_pendingIncludeHidden = includeHidden;
    m_calculationPending = true;
    
    // Restart debounce timer
    m_updateTimer->stop();
    m_updateTimer->start();
    
    // Show calculating status
    m_statusLabel->setText(tr("Calculating..."));
    m_statusLabel->setVisible(true);
}

void ScanScopePreviewWidget::performCalculation()
{
    if (!m_calculationPending) {
        return;
    }
    
    m_calculationPending = false;
    emit calculationStarted();
    
    // Perform calculation
    calculateStats(m_pendingTargetPaths,
                  m_pendingExcludePatterns,
                  m_pendingExcludeFolders,
                  m_pendingMaxDepth,
                  m_pendingIncludeHidden);
    
    // Update display
    updateDisplay();
    
    emit calculationFinished();
    emit previewUpdated(m_currentStats);
}

void ScanScopePreviewWidget::calculateStats(const QStringList& targetPaths,
                                            const QStringList& excludePatterns,
                                            const QStringList& excludeFolders,
                                            int maxDepth,
                                            bool includeHidden)
{
    // Reset stats
    m_currentStats = ScopeStats();
    m_currentStats.includedPaths = targetPaths;
    m_currentStats.excludedPaths = excludeFolders;
    
    if (targetPaths.isEmpty()) {
        m_currentStats.calculationComplete = true;
        m_currentStats.errorMessage = tr("No paths selected");
        return;
    }
    
    // Count folders and estimate files
    int totalFolders = 0;
    int totalFiles = 0;
    qint64 totalSize = 0;
    int sampledFiles = 0;
    
    for (const QString& targetPath : targetPaths) {
        QDir dir(targetPath);
        if (!dir.exists()) {
            continue;
        }
        
        // Configure iterator flags
        QDirIterator::IteratorFlags flags = QDirIterator::Subdirectories;
        if (!includeHidden) {
            flags |= QDirIterator::FollowSymlinks;
        }
        
        QDir::Filters filters = QDir::Dirs | QDir::Files | QDir::NoDotAndDotDot;
        if (includeHidden) {
            filters |= QDir::Hidden;
        }
        
        QDirIterator it(targetPath, filters, flags);
        
        while (it.hasNext() && sampledFiles < MAX_SAMPLE_FILES) {
            QString path = it.next();
            QFileInfo info = it.fileInfo();
            
            // Check if path should be excluded
            if (shouldExcludePath(path, excludePatterns, excludeFolders)) {
                continue;
            }
            
            // Check depth limit
            if (maxDepth >= 0) {
                QString relativePath = path;
                relativePath.remove(targetPath);
                int depth = relativePath.count(QDir::separator());
                if (depth > maxDepth) {
                    continue;
                }
            }
            
            if (info.isDir()) {
                totalFolders++;
            } else if (info.isFile()) {
                totalFiles++;
                totalSize += info.size();
                sampledFiles++;
            }
        }
    }
    
    // If we hit the sample limit, estimate total
    if (sampledFiles >= MAX_SAMPLE_FILES) {
        // Rough estimation: assume similar file distribution
        double estimationFactor = 1.5; // Conservative multiplier
        m_currentStats.estimatedFileCount = static_cast<int>(totalFiles * estimationFactor);
        m_currentStats.estimatedSize = static_cast<qint64>(totalSize * estimationFactor);
        m_currentStats.folderCount = totalFolders;
    } else {
        m_currentStats.estimatedFileCount = totalFiles;
        m_currentStats.estimatedSize = totalSize;
        m_currentStats.folderCount = totalFolders;
    }
    
    m_currentStats.calculationComplete = true;
    
    Logger::instance()->log(Logger::Info, "ScanScopePreview",
        QString("Calculated scope: %1 folders, ~%2 files, ~%3")
            .arg(m_currentStats.folderCount)
            .arg(m_currentStats.estimatedFileCount)
            .arg(formatFileSize(m_currentStats.estimatedSize)));
}

void ScanScopePreviewWidget::updateDisplay()
{
    if (!m_currentStats.calculationComplete) {
        return;
    }
    
    // Update labels
    m_folderCountLabel->setText(tr("Folders: %1").arg(formatNumber(m_currentStats.folderCount)));
    m_fileCountLabel->setText(tr("Estimated Files: ~%1").arg(formatNumber(m_currentStats.estimatedFileCount)));
    m_sizeLabel->setText(tr("Estimated Size: ~%1").arg(formatFileSize(m_currentStats.estimatedSize)));
    
    // Update status
    if (!m_currentStats.errorMessage.isEmpty()) {
        m_statusLabel->setText(m_currentStats.errorMessage);
        // Apply theme-aware error styling
        m_statusLabel->setStyleSheet(ThemeManager::instance()->getStatusIndicatorStyle(ThemeManager::StatusType::Error) + 
                                   " font-style: italic; padding: 4px;");
        m_statusLabel->setVisible(true);
    } else {
        m_statusLabel->setVisible(false);
    }
    
    // Update paths tree
    m_pathsTree->clear();
    
    // Add included paths
    for (const QString& path : m_currentStats.includedPaths) {
        QTreeWidgetItem* item = new QTreeWidgetItem(m_pathsTree);
        item->setText(0, path);
        
        QDir dir(path);
        if (dir.exists()) {
            item->setText(1, tr("✓ Included"));
            // Use theme-aware success color
            QColor successColor = ThemeManager::instance()->currentTheme() == ThemeManager::Dark ? 
                                QColor("#4CAF50") : QColor("#28a745");
            item->setForeground(1, successColor);
        } else {
            item->setText(1, tr("✗ Not Found"));
            // Use theme-aware error color
            QColor errorColor = ThemeManager::instance()->currentTheme() == ThemeManager::Dark ? 
                              QColor("#F44336") : QColor("#dc3545");
            item->setForeground(1, errorColor);
        }
        
        item->setIcon(0, style()->standardIcon(QStyle::SP_DirIcon));
    }
    
    // Add excluded paths (if any)
    if (!m_currentStats.excludedPaths.isEmpty()) {
        for (const QString& path : m_currentStats.excludedPaths) {
            QTreeWidgetItem* item = new QTreeWidgetItem(m_pathsTree);
            item->setText(0, path);
            item->setText(1, tr("⊘ Excluded"));
            // Use theme-aware warning color
            QColor warningColor = ThemeManager::instance()->currentTheme() == ThemeManager::Dark ? 
                                QColor("#FF9800") : QColor("#ffc107");
            item->setForeground(1, warningColor);
            item->setIcon(0, style()->standardIcon(QStyle::SP_DirIcon));
        }
    }
}

ScanScopePreviewWidget::ScopeStats ScanScopePreviewWidget::getCurrentStats() const
{
    return m_currentStats;
}

void ScanScopePreviewWidget::clear()
{
    m_currentStats = ScopeStats();
    m_folderCountLabel->setText(tr("Folders: -"));
    m_fileCountLabel->setText(tr("Estimated Files: -"));
    m_sizeLabel->setText(tr("Estimated Size: -"));
    m_statusLabel->setVisible(false);
    m_pathsTree->clear();
    m_updateTimer->stop();
    m_calculationPending = false;
}

bool ScanScopePreviewWidget::shouldExcludePath(const QString& path,
                                               const QStringList& excludePatterns,
                                               const QStringList& excludeFolders) const
{
    // Check exclude folders
    for (const QString& excludeFolder : excludeFolders) {
        if (path.startsWith(excludeFolder)) {
            return true;
        }
    }
    
    // Check exclude patterns
    QFileInfo info(path);
    QString fileName = info.fileName();
    
    for (const QString& pattern : excludePatterns) {
        if (pattern.trimmed().isEmpty()) {
            continue;
        }
        
        if (matchesPattern(fileName, pattern)) {
            return true;
        }
    }
    
    return false;
}

bool ScanScopePreviewWidget::matchesPattern(const QString& path, const QString& pattern) const
{
    // Convert wildcard pattern to regex
    QString regexPattern = QRegularExpression::escape(pattern);
    regexPattern.replace("\\*", ".*");
    regexPattern.replace("\\?", ".");
    
    QRegularExpression regex("^" + regexPattern + "$", QRegularExpression::CaseInsensitiveOption);
    return regex.match(path).hasMatch();
}

QString ScanScopePreviewWidget::formatFileSize(qint64 bytes) const
{
    const qint64 KB = 1024;
    const qint64 MB = KB * 1024;
    const qint64 GB = MB * 1024;
    const qint64 TB = GB * 1024;
    
    if (bytes >= TB) {
        return QString("%1 TB").arg(bytes / static_cast<double>(TB), 0, 'f', 2);
    } else if (bytes >= GB) {
        return QString("%1 GB").arg(bytes / static_cast<double>(GB), 0, 'f', 2);
    } else if (bytes >= MB) {
        return QString("%1 MB").arg(bytes / static_cast<double>(MB), 0, 'f', 2);
    } else if (bytes >= KB) {
        return QString("%1 KB").arg(bytes / static_cast<double>(KB), 0, 'f', 2);
    } else {
        return QString("%1 bytes").arg(bytes);
    }
}

QString ScanScopePreviewWidget::formatNumber(int number) const
{
    // Add thousand separators
    QString str = QString::number(number);
    int pos = str.length() - 3;
    
    while (pos > 0) {
        str.insert(pos, ',');
        pos -= 3;
    }
    
    return str;
}
