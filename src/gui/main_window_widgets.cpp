#include "main_window.h"
#include "scan_history_manager.h"
#include "theme_manager.h"
#include <QtWidgets/QListWidgetItem>
#include <QtCore/QStandardPaths>
#include <QtCore/QStorageInfo>
#include <QtCore/QDateTime>

#include <QtGui/QPalette>

// ScanHistoryWidget Implementation
ScanHistoryWidget::ScanHistoryWidget(QWidget* parent)
    : QGroupBox(tr("Scan History"), parent)
    , m_layout(new QVBoxLayout(this))
    , m_historyList(new QListWidget(this))
    , m_viewAllButton(new QPushButton(tr("View All →"), this))
{
    m_layout->setContentsMargins(12, 20, 12, 12);
    m_layout->setSpacing(8);
    
    // Configure list widget
    m_historyList->setMaximumHeight(120);
    m_historyList->setAlternatingRowColors(true);
    m_historyList->setSelectionMode(QAbstractItemView::SingleSelection);
    
    // Configure view all button
    m_viewAllButton->setFixedHeight(24);
    m_viewAllButton->setStyleSheet(R"(
        QPushButton {
            text-align: right;
            border: none;
            color: palette(link);
            font-weight: bold;
        }
        QPushButton:hover {
            color: palette(link-visited);
            text-decoration: underline;
        }
    )");
    
    // Layout
    m_layout->addWidget(m_historyList);
    
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();
    buttonLayout->addWidget(m_viewAllButton);
    m_layout->addLayout(buttonLayout);
    
    // Connections
    connect(m_historyList, QOverload<QListWidgetItem*>::of(&QListWidget::itemDoubleClicked),
            [this](QListWidgetItem* item) {
                int row = m_historyList->row(item);
                onItemDoubleClicked(row);
            });
    connect(m_viewAllButton, &QPushButton::clicked, this, &ScanHistoryWidget::onViewAllClicked);
    
    // Don't add sample history - will be populated from real scans
    // addSampleHistory();
}

void ScanHistoryWidget::addScanResult(const ScanHistoryItem& item)
{
    m_historyItems.prepend(item); // Add to beginning
    
    // Keep only last 10 items in memory
    if (m_historyItems.size() > 10) {
        m_historyItems.removeLast();
    }
    
    updateHistoryDisplay();
}

void ScanHistoryWidget::clearHistory()
{
    m_historyItems.clear();
    updateHistoryDisplay();
}

QList<ScanHistoryWidget::ScanHistoryItem> ScanHistoryWidget::getHistory() const
{
    return m_historyItems;
}

void ScanHistoryWidget::refreshHistory()
{
    // Clear existing history
    m_historyItems.clear();
    
    // Load all scans from history manager
    QList<ScanHistoryManager::ScanRecord> scans = 
        ScanHistoryManager::instance()->getAllScans();
    
    // Convert to widget format
    for (const auto& record : scans) {
        ScanHistoryItem item;
        item.scanId = record.scanId;
        
        // Format date
        QDateTime now = QDateTime::currentDateTime();
        qint64 daysDiff = record.timestamp.daysTo(now);
        
        if (daysDiff == 0) {
            item.date = tr("Today, %1").arg(record.timestamp.toString("h:mm AP"));
        } else if (daysDiff == 1) {
            item.date = tr("Yesterday, %1").arg(record.timestamp.toString("h:mm AP"));
        } else if (daysDiff < 7) {
            item.date = record.timestamp.toString("dddd, h:mm AP");
        } else {
            item.date = record.timestamp.toString("MMM d, h:mm AP");
        }
        
        // Determine scan type from paths
        if (record.targetPaths.isEmpty()) {
            item.type = tr("Unknown");
        } else if (record.targetPaths.size() == 1) {
            QString path = record.targetPaths.first();
            if (path.contains("Download", Qt::CaseInsensitive)) {
                item.type = tr("Downloads");
            } else if (path.contains("Picture", Qt::CaseInsensitive) || path.contains("Photo", Qt::CaseInsensitive)) {
                item.type = tr("Photos");
            } else if (path.contains("Document", Qt::CaseInsensitive)) {
                item.type = tr("Documents");
            } else {
                item.type = tr("Custom");
            }
        } else {
            item.type = tr("Multiple Locations");
        }
        
        item.duplicateCount = record.duplicateGroups;
        item.spaceSaved = record.potentialSavings;
        
        m_historyItems.append(item);
    }
    
    // Update the display
    updateHistoryDisplay();
}

void ScanHistoryWidget::onItemDoubleClicked(int row)
{
    if (row >= 0 && row < m_historyItems.size()) {
        emit historyItemClicked(row);
    }
}

void ScanHistoryWidget::onViewAllClicked()
{
    emit viewAllRequested();
}

void ScanHistoryWidget::updateHistoryDisplay()
{
    m_historyList->clear();
    
    // Show only the 3 most recent items in the main window
    int itemsToShow = qMin(3, static_cast<int>(m_historyItems.size()));
    
    for (int i = 0; i < itemsToShow; ++i) {
        const ScanHistoryItem& item = m_historyItems[i];
        QString itemText = formatHistoryItem(item);
        
        QListWidgetItem* listItem = new QListWidgetItem(itemText, m_historyList);
        listItem->setData(Qt::UserRole, item.scanId);
    }
    
    // Show/hide the "View All" button based on whether there are more items
    m_viewAllButton->setVisible(m_historyItems.size() > 3);
}

QString ScanHistoryWidget::formatHistoryItem(const ScanHistoryItem& item) const
{
    QString spaceSaved = formatBytes(item.spaceSaved);
    return QString("%1 | %2 | %3 duplicates | %4")
           .arg(item.date)
           .arg(item.type)
           .arg(item.duplicateCount)
           .arg(spaceSaved);
}

QString ScanHistoryWidget::formatBytes(qint64 bytes) const
{
    const qint64 KB = 1024;
    const qint64 MB = KB * 1024;
    const qint64 GB = MB * 1024;
    
    if (bytes >= GB) {
        return QString("%1 GB").arg(QString::number(static_cast<double>(bytes) / double(GB), 'f', 1));
    } else if (bytes >= MB) {
        return QString("%1 MB").arg(QString::number(static_cast<double>(bytes) / double(MB), 'f', 1));
    } else if (bytes >= KB) {
        return QString("%1 KB").arg(QString::number(static_cast<double>(bytes) / double(KB), 'f', 1));
    } else {
        return QString("%1 B").arg(bytes);
    }
}

void ScanHistoryWidget::addSampleHistory()
{
    // Add some sample history items for demonstration
    ScanHistoryItem item1;
    item1.date = "Today, 2:30 PM";
    item1.type = "Downloads";
    item1.duplicateCount = 127;
    item1.spaceSaved = 2400000000; // 2.4 GB
    item1.scanId = "scan_001";
    
    ScanHistoryItem item2;
    item2.date = "Yesterday, 6:15 PM";
    item2.type = "Pictures";
    item2.duplicateCount = 89;
    item2.spaceSaved = 1800000000; // 1.8 GB
    item2.scanId = "scan_002";
    
    ScanHistoryItem item3;
    item3.date = "Oct 1, 3:45 PM";
    item3.type = "Full System";
    item3.duplicateCount = 341;
    item3.spaceSaved = 5100000000; // 5.1 GB
    item3.scanId = "scan_003";
    
    m_historyItems = { item1, item2, item3 };
    updateHistoryDisplay();
}

// SystemOverviewWidget Implementation
SystemOverviewWidget::SystemOverviewWidget(QWidget* parent)
    : QGroupBox(tr("System Overview"), parent)
    , m_layout(new QVBoxLayout(this))
    , m_diskSpaceLabel(new QLabel(this))
    , m_diskUsageBar(new QProgressBar(this))
    , m_availableSpaceLabel(new QLabel(this))
    , m_potentialSavingsLabel(new QLabel(this))
    , m_filesScannedLabel(new QLabel(this))
{
    m_layout->setContentsMargins(12, 20, 12, 12);
    m_layout->setSpacing(8);
    
    createStatsDisplay();
    
    // Initialize with default stats
    SystemStats defaultStats;
    defaultStats.totalDiskSpace = 0;
    defaultStats.availableDiskSpace = 0;
    defaultStats.potentialSavings = 0;
    defaultStats.filesScanned = 0;
    defaultStats.usagePercentage = 0.0;
    
    updateStats(defaultStats);
}

void SystemOverviewWidget::updateStats(const SystemStats& stats)
{
    m_currentStats = stats;
    updateDisplay();
}

SystemOverviewWidget::SystemStats SystemOverviewWidget::getCurrentStats() const
{
    return m_currentStats;
}

void SystemOverviewWidget::refreshStats()
{
    // Get current disk space info
    QString homePath = QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
    QStorageInfo storage(homePath);
    
    SystemStats stats = m_currentStats; // Keep existing values
    
    if (storage.isValid()) {
        stats.totalDiskSpace = storage.bytesTotal();
        stats.availableDiskSpace = storage.bytesAvailable();
        stats.usagePercentage = 100.0 - (double(storage.bytesAvailable()) / double(storage.bytesTotal()) * 100.0);
    }
    
    updateStats(stats);
}

void SystemOverviewWidget::createStatsDisplay()
{
    // Create two-column layout for stats
    QHBoxLayout* topLayout = new QHBoxLayout();
    QHBoxLayout* bottomLayout = new QHBoxLayout();
    
    // Disk space info - theme-aware styling
    QFont boldFont = m_diskSpaceLabel->font();
    boldFont.setBold(true);
    m_diskSpaceLabel->setFont(boldFont);
    
    m_diskUsageBar->setMaximum(100);
    m_diskUsageBar->setTextVisible(true);
    m_diskUsageBar->setFormat("%p% used");
    m_diskUsageBar->setFixedHeight(20);
    
    m_availableSpaceLabel->setWordWrap(true);
    
    // Statistics labels - theme-aware styling
    QFont savingsFont = m_potentialSavingsLabel->font();
    savingsFont.setBold(true);
    m_potentialSavingsLabel->setFont(savingsFont);
    // Theme-aware styling applied by ThemeManager for text color
    
    // Layout
    topLayout->addWidget(m_diskSpaceLabel);
    topLayout->addWidget(m_availableSpaceLabel);
    
    bottomLayout->addWidget(m_potentialSavingsLabel);
    bottomLayout->addWidget(m_filesScannedLabel);
    
    m_layout->addLayout(topLayout);
    m_layout->addWidget(m_diskUsageBar);
    m_layout->addLayout(bottomLayout);
}

void SystemOverviewWidget::updateDisplay()
{
    // Format disk space info
    QString totalSpace = formatBytes(m_currentStats.totalDiskSpace);
    QString availableSpace = formatBytes(m_currentStats.availableDiskSpace);
    double availablePercentage = 100.0 - m_currentStats.usagePercentage;
    
    m_diskSpaceLabel->setText(QString("Total Disk Space: %1").arg(totalSpace));
    m_availableSpaceLabel->setText(QString("Available: %1 (%2%)").arg(availableSpace).arg(QString::number(availablePercentage, 'f', 1)));
    
    // Update progress bar
    m_diskUsageBar->setValue(static_cast<int>(m_currentStats.usagePercentage));
    
    // Set color based on usage with better theme awareness
    QColor usageColor = getUsageColor(m_currentStats.usagePercentage);
    QString barStyle = QString(R"(
        QProgressBar {
            border: 1px solid palette(mid);
            border-radius: 3px;
            text-align: center;
            background: palette(base);
            color: palette(window-text);
        }
        QProgressBar::chunk {
            background-color: %1;
            border-radius: 2px;
        }
    )").arg(usageColor.name());
    
    m_diskUsageBar->setStyleSheet(barStyle);
    
    // Update other stats
    QString potentialSavings = formatBytes(m_currentStats.potentialSavings);
    m_potentialSavingsLabel->setText(QString("Potential Savings: %1").arg(potentialSavings));
    
    QString filesScannedText = QString("Files Scanned: %L1").arg(m_currentStats.filesScanned);
    m_filesScannedLabel->setText(filesScannedText);
}

QString SystemOverviewWidget::formatBytes(qint64 bytes) const
{
    const qint64 KB = 1024;
    const qint64 MB = KB * 1024;
    const qint64 GB = MB * 1024;
    const qint64 TB = GB * 1024;
    
    if (bytes >= TB) {
        return QString("%1 TB").arg(QString::number(static_cast<double>(bytes) / double(TB), 'f', 1));
    } else if (bytes >= GB) {
        return QString("%1 GB").arg(QString::number(static_cast<double>(bytes) / double(GB), 'f', 1));
    } else if (bytes >= MB) {
        return QString("%1 MB").arg(QString::number(static_cast<double>(bytes) / double(MB), 'f', 1));
    } else if (bytes >= KB) {
        return QString("%1 KB").arg(QString::number(static_cast<double>(bytes) / double(KB), 'f', 1));
    } else {
        return QString("%1 B").arg(bytes);
    }
}

QColor SystemOverviewWidget::getUsageColor(double percentage) const
{
    // Use theme-aware colors from ThemeManager
    ThemeData themeData = ThemeManager::instance()->getCurrentThemeData();
    
    if (percentage < 60.0) {
        // Success color - green
        return themeData.colors.success;
    } else if (percentage < 80.0) {
        // Warning color - orange
        return themeData.colors.warning;
    } else {
        // Error color - red
        return themeData.colors.error;
    }
}
