#include "scan_history_dialog.h"
#include "core/logger.h"
#include <QHeaderView>
#include <QMessageBox>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QDateTime>
#include <QDesktopServices>
#include <QUrl>
#include <QInputDialog>

ScanHistoryDialog::ScanHistoryDialog(QWidget* parent)
    : QDialog(parent)
    , m_mainLayout(nullptr)
    , m_filterWidget(nullptr)
    , m_searchEdit(nullptr)
    , m_typeFilter(nullptr)
    , m_dateFromEdit(nullptr)
    , m_dateToEdit(nullptr)
    , m_refreshButton(nullptr)
    , m_historyTable(nullptr)
    , m_statsLabel(nullptr)
    , m_buttonLayout(nullptr)
    , m_viewButton(nullptr)
    , m_deleteButton(nullptr)
    , m_exportButton(nullptr)
    , m_clearOldButton(nullptr)
    , m_closeButton(nullptr)
{
    setWindowTitle(tr("Scan History"));
    setMinimumSize(900, 600);
    resize(950, 650);
    
    setupUI();
    loadHistory();
    
    LOG_INFO(LogCategories::UI, "Scan History dialog created");
}

ScanHistoryDialog::~ScanHistoryDialog()
{
    LOG_DEBUG(LogCategories::UI, "Scan History dialog destroyed");
}

void ScanHistoryDialog::setupUI()
{
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setContentsMargins(15, 15, 15, 15);
    m_mainLayout->setSpacing(10);
    
    // Filter controls
    m_filterWidget = new QWidget(this);
    QHBoxLayout* filterLayout = new QHBoxLayout(m_filterWidget);
    filterLayout->setContentsMargins(0, 0, 0, 0);
    
    QLabel* searchLabel = new QLabel(tr("Search:"), this);
    m_searchEdit = new QLineEdit(this);
    m_searchEdit->setPlaceholderText(tr("Search by path or type..."));
    connect(m_searchEdit, &QLineEdit::textChanged, this, &ScanHistoryDialog::onSearchTextChanged);
    
    QLabel* typeLabel = new QLabel(tr("Type:"), this);
    m_typeFilter = new QComboBox(this);
    m_typeFilter->addItem(tr("All Types"), "all");
    m_typeFilter->addItem(tr("Downloads"), "downloads");
    m_typeFilter->addItem(tr("Photos"), "photos");
    m_typeFilter->addItem(tr("Documents"), "documents");
    m_typeFilter->addItem(tr("Custom"), "custom");
    connect(m_typeFilter, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &ScanHistoryDialog::onFilterChanged);
    
    QLabel* dateLabel = new QLabel(tr("Date:"), this);
    m_dateFromEdit = new QDateEdit(this);
    m_dateFromEdit->setCalendarPopup(true);
    m_dateFromEdit->setDate(QDate::currentDate().addMonths(-1));
    connect(m_dateFromEdit, &QDateEdit::dateChanged, this, &ScanHistoryDialog::onFilterChanged);
    
    QLabel* toLabel = new QLabel(tr("to"), this);
    m_dateToEdit = new QDateEdit(this);
    m_dateToEdit->setCalendarPopup(true);
    m_dateToEdit->setDate(QDate::currentDate());
    connect(m_dateToEdit, &QDateEdit::dateChanged, this, &ScanHistoryDialog::onFilterChanged);
    
    m_refreshButton = new QPushButton(tr("🔄 Refresh"), this);
    connect(m_refreshButton, &QPushButton::clicked, this, &ScanHistoryDialog::onRefreshClicked);
    
    filterLayout->addWidget(searchLabel);
    filterLayout->addWidget(m_searchEdit, 2);
    filterLayout->addWidget(typeLabel);
    filterLayout->addWidget(m_typeFilter, 1);
    filterLayout->addWidget(dateLabel);
    filterLayout->addWidget(m_dateFromEdit);
    filterLayout->addWidget(toLabel);
    filterLayout->addWidget(m_dateToEdit);
    filterLayout->addWidget(m_refreshButton);
    
    m_mainLayout->addWidget(m_filterWidget);
    
    // Table
    m_historyTable = new QTableWidget(this);
    m_historyTable->setColumnCount(6);
    m_historyTable->setHorizontalHeaderLabels({
        tr("Date/Time"), tr("Type"), tr("Locations"), 
        tr("Files Scanned"), tr("Duplicate Groups"), tr("Potential Savings")
    });
    
    m_historyTable->horizontalHeader()->setStretchLastSection(false);
    m_historyTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_historyTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    m_historyTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    m_historyTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    m_historyTable->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    m_historyTable->horizontalHeader()->setSectionResizeMode(5, QHeaderView::ResizeToContents);
    
    m_historyTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_historyTable->setSelectionMode(QAbstractItemView::SingleSelection);
    m_historyTable->setAlternatingRowColors(true);
    m_historyTable->setSortingEnabled(true);
    
    connect(m_historyTable, &QTableWidget::cellDoubleClicked, 
            this, &ScanHistoryDialog::onTableDoubleClicked);
    connect(m_historyTable, &QTableWidget::itemSelectionChanged,
            this, &ScanHistoryDialog::onSelectionChanged);
    
    m_mainLayout->addWidget(m_historyTable);
    
    // Stats
    m_statsLabel = new QLabel(this);
    m_statsLabel->setStyleSheet("QLabel { padding: 5px; background: palette(base); border: 1px solid palette(mid); border-radius: 3px; }");
    m_mainLayout->addWidget(m_statsLabel);
    
    // Action buttons
    m_buttonLayout = new QHBoxLayout();
    
    m_viewButton = new QPushButton(tr("View Results"), this);
    m_viewButton->setEnabled(false);
    connect(m_viewButton, &QPushButton::clicked, this, &ScanHistoryDialog::onViewClicked);
    
    m_deleteButton = new QPushButton(tr("Delete"), this);
    m_deleteButton->setEnabled(false);
    connect(m_deleteButton, &QPushButton::clicked, this, &ScanHistoryDialog::onDeleteClicked);
    
    m_exportButton = new QPushButton(tr("Export History..."), this);
    connect(m_exportButton, &QPushButton::clicked, this, &ScanHistoryDialog::onExportClicked);
    
    m_clearOldButton = new QPushButton(tr("Clear Old Scans..."), this);
    connect(m_clearOldButton, &QPushButton::clicked, this, &ScanHistoryDialog::onClearOldClicked);
    
    m_closeButton = new QPushButton(tr("Close"), this);
    connect(m_closeButton, &QPushButton::clicked, this, &QDialog::accept);
    
    m_buttonLayout->addWidget(m_viewButton);
    m_buttonLayout->addWidget(m_deleteButton);
    m_buttonLayout->addWidget(m_exportButton);
    m_buttonLayout->addWidget(m_clearOldButton);
    m_buttonLayout->addStretch();
    m_buttonLayout->addWidget(m_closeButton);
    
    m_mainLayout->addLayout(m_buttonLayout);
}

void ScanHistoryDialog::loadHistory()
{
    LOG_INFO(LogCategories::SYSTEM, "Loading scan history");
    
    m_allScans = ScanHistoryManager::instance()->getAllScans();
    m_filteredScans = m_allScans;
    
    LOG_INFO(LogCategories::SYSTEM, QString("Loaded %1 scans").arg(m_allScans.size()));
    
    applyFilters();
}

void ScanHistoryDialog::refreshHistory()
{
    LOG_INFO(LogCategories::UI, "Refreshing scan history");
    loadHistory();
}

void ScanHistoryDialog::applyFilters()
{
    m_historyTable->setRowCount(0);
    m_filteredScans.clear();
    
    QString searchText = m_searchEdit->text().toLower();
    QString typeFilter = m_typeFilter->currentData().toString();
    QDate dateFrom = m_dateFromEdit->date();
    QDate dateTo = m_dateToEdit->date();
    
    // Apply filters
    for (const auto& scan : m_allScans) {
        // Date filter
        if (scan.timestamp.date() < dateFrom || scan.timestamp.date() > dateTo) {
            continue;
        }
        
        // Type filter
        if (typeFilter != "all") {
            bool matchesType = false;
            for (const QString& path : scan.targetPaths) {
                if (typeFilter == "downloads" && path.contains("Download", Qt::CaseInsensitive)) {
                    matchesType = true;
                } else if (typeFilter == "photos" && (path.contains("Picture", Qt::CaseInsensitive) || path.contains("Photo", Qt::CaseInsensitive))) {
                    matchesType = true;
                } else if (typeFilter == "documents" && path.contains("Document", Qt::CaseInsensitive)) {
                    matchesType = true;
                } else if (typeFilter == "custom") {
                    matchesType = true;
                }
            }
            if (!matchesType) continue;
        }
        
        // Search filter
        if (!searchText.isEmpty()) {
            bool matchesSearch = false;
            for (const QString& path : scan.targetPaths) {
                if (path.toLower().contains(searchText)) {
                    matchesSearch = true;
                    break;
                }
            }
            if (!matchesSearch) continue;
        }
        
        m_filteredScans.append(scan);
    }
    
    // Populate table
    m_historyTable->setRowCount(m_filteredScans.size());
    
    for (int i = 0; i < m_filteredScans.size(); ++i) {
        const auto& scan = m_filteredScans[i];
        
        // Date/Time
        QTableWidgetItem* dateItem = new QTableWidgetItem(formatDateTime(scan.timestamp));
        dateItem->setData(Qt::UserRole, scan.scanId);
        m_historyTable->setItem(i, 0, dateItem);
        
        // Type
        QString type = tr("Custom");
        if (!scan.targetPaths.isEmpty()) {
            QString path = scan.targetPaths.first();
            if (path.contains("Download", Qt::CaseInsensitive)) type = tr("Downloads");
            else if (path.contains("Picture", Qt::CaseInsensitive) || path.contains("Photo", Qt::CaseInsensitive)) type = tr("Photos");
            else if (path.contains("Document", Qt::CaseInsensitive)) type = tr("Documents");
            else if (scan.targetPaths.size() > 1) type = tr("Multiple");
        }
        m_historyTable->setItem(i, 1, new QTableWidgetItem(type));
        
        // Locations
        QString locations = scan.targetPaths.join(", ");
        if (locations.length() > 50) {
            locations = locations.left(47) + "...";
        }
        m_historyTable->setItem(i, 2, new QTableWidgetItem(locations));
        
        // Files Scanned
        m_historyTable->setItem(i, 3, new QTableWidgetItem(QString::number(scan.filesScanned)));
        
        // Duplicate Groups
        m_historyTable->setItem(i, 4, new QTableWidgetItem(QString::number(scan.duplicateGroups)));
        
        // Potential Savings
        m_historyTable->setItem(i, 5, new QTableWidgetItem(formatFileSize(scan.potentialSavings)));
    }
    
    updateStats();
    
    LOG_INFO(LogCategories::SYSTEM, QString("Displaying %1 filtered scans").arg(m_filteredScans.size()));
}

void ScanHistoryDialog::updateStats()
{
    int totalScans = m_filteredScans.size();
    int totalGroups = 0;
    qint64 totalSavings = 0;
    
    for (const auto& scan : m_filteredScans) {
        totalGroups += scan.duplicateGroups;
        totalSavings += scan.potentialSavings;
    }
    
    m_statsLabel->setText(tr("Showing %1 scans | %2 total duplicate groups | %3 potential savings")
                         .arg(totalScans)
                         .arg(totalGroups)
                         .arg(formatFileSize(totalSavings)));
}

void ScanHistoryDialog::onViewClicked()
{
    int row = m_historyTable->currentRow();
    if (row >= 0 && row < m_filteredScans.size()) {
        QString scanId = m_filteredScans[row].scanId;
        LOG_INFO(LogCategories::UI, QString("User viewing scan: %1").arg(scanId));
        emit scanSelected(scanId);
        accept();
    }
}

void ScanHistoryDialog::onDeleteClicked()
{
    int row = m_historyTable->currentRow();
    if (row >= 0 && row < m_filteredScans.size()) {
        const auto& scan = m_filteredScans[row];
        
        QMessageBox::StandardButton reply = QMessageBox::question(this, tr("Delete Scan"),
            tr("Are you sure you want to delete this scan from history?\n\nDate: %1\nGroups: %2")
            .arg(formatDateTime(scan.timestamp))
            .arg(scan.duplicateGroups),
            QMessageBox::Yes | QMessageBox::No);
        
        if (reply == QMessageBox::Yes) {
            LOG_INFO(LogCategories::SYSTEM, QString("Deleting scan: %1").arg(scan.scanId));
            ScanHistoryManager::instance()->deleteScan(scan.scanId);
            emit scanDeleted(scan.scanId);
            refreshHistory();
        }
    }
}

void ScanHistoryDialog::onRefreshClicked()
{
    LOG_INFO(LogCategories::UI, "User clicked refresh in scan history");
    refreshHistory();
}

void ScanHistoryDialog::onExportClicked()
{
    LOG_INFO(LogCategories::UI, "User exporting scan history");
    
    QString fileName = QFileDialog::getSaveFileName(this, tr("Export Scan History"),
                                                   "scan_history.csv",
                                                   tr("CSV Files (*.csv);;All Files (*)"));
    
    if (fileName.isEmpty()) {
        return;
    }
    
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, tr("Export Error"),
                           tr("Could not open file for writing: %1").arg(file.errorString()));
        return;
    }
    
    QTextStream out(&file);
    
    // Write header
    out << "Date/Time,Type,Locations,Files Scanned,Duplicate Groups,Potential Savings (bytes)\n";
    
    // Write data
    for (const auto& scan : m_filteredScans) {
        out << formatDateTime(scan.timestamp) << ","
            << (scan.targetPaths.isEmpty() ? "Unknown" : scan.targetPaths.first()) << ","
            << scan.targetPaths.join(";") << ","
            << scan.filesScanned << ","
            << scan.duplicateGroups << ","
            << scan.potentialSavings << "\n";
    }
    
    file.close();
    
    LOG_INFO(LogCategories::EXPORT, QString("Exported %1 scans to: %2").arg(m_filteredScans.size()).arg(fileName));
    QMessageBox::information(this, tr("Export Complete"),
                           tr("Scan history exported successfully to:\n%1").arg(fileName));
}

void ScanHistoryDialog::onClearOldClicked()
{
    LOG_INFO(LogCategories::UI, "User clearing old scans");
    
    bool ok;
    int days = QInputDialog::getInt(this, tr("Clear Old Scans"),
                                   tr("Delete scans older than how many days?"),
                                   30, 1, 365, 1, &ok);
    
    if (ok) {
        QMessageBox::StandardButton reply = QMessageBox::question(this, tr("Confirm Clear"),
            tr("This will delete all scans older than %1 days. Continue?").arg(days),
            QMessageBox::Yes | QMessageBox::No);
        
        if (reply == QMessageBox::Yes) {
            LOG_INFO(LogCategories::SYSTEM, QString("Clearing scans older than %1 days").arg(days));
            ScanHistoryManager::instance()->clearOldScans(days);
            refreshHistory();
            QMessageBox::information(this, tr("Scans Cleared"),
                                   tr("Old scans have been removed from history."));
        }
    }
}

void ScanHistoryDialog::onSearchTextChanged(const QString& text)
{
    Q_UNUSED(text);
    applyFilters();
}

void ScanHistoryDialog::onFilterChanged()
{
    applyFilters();
}

void ScanHistoryDialog::onTableDoubleClicked(int row, int column)
{
    Q_UNUSED(column);
    if (row >= 0 && row < m_filteredScans.size()) {
        QString scanId = m_filteredScans[row].scanId;
        LOG_INFO(LogCategories::UI, QString("User double-clicked scan: %1").arg(scanId));
        emit scanSelected(scanId);
        accept();
    }
}

void ScanHistoryDialog::onSelectionChanged()
{
    bool hasSelection = m_historyTable->currentRow() >= 0;
    m_viewButton->setEnabled(hasSelection);
    m_deleteButton->setEnabled(hasSelection);
}

QString ScanHistoryDialog::formatFileSize(qint64 bytes) const
{
    const qint64 KB = 1024;
    const qint64 MB = KB * 1024;
    const qint64 GB = MB * 1024;
    const qint64 TB = GB * 1024;
    
    if (bytes >= TB) {
        return QString("%1 TB").arg(QString::number(static_cast<double>(bytes) / double(TB), 'f', 2));
    } else if (bytes >= GB) {
        return QString("%1 GB").arg(QString::number(static_cast<double>(bytes) / double(GB), 'f', 2));
    } else if (bytes >= MB) {
        return QString("%1 MB").arg(QString::number(static_cast<double>(bytes) / double(MB), 'f', 1));
    } else if (bytes >= KB) {
        return QString("%1 KB").arg(QString::number(static_cast<double>(bytes) / double(KB), 'f', 1));
    } else {
        return QString("%1 B").arg(bytes);
    }
}

QString ScanHistoryDialog::formatDateTime(const QDateTime& dt) const
{
    QDateTime now = QDateTime::currentDateTime();
    qint64 daysDiff = dt.daysTo(now);
    
    if (daysDiff == 0) {
        return tr("Today, %1").arg(dt.toString("h:mm AP"));
    } else if (daysDiff == 1) {
        return tr("Yesterday, %1").arg(dt.toString("h:mm AP"));
    } else if (daysDiff < 7) {
        return dt.toString("dddd, h:mm AP");
    } else {
        return dt.toString("MMM d, yyyy h:mm AP");
    }
}
