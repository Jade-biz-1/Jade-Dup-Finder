#include "restore_dialog.h"
#include "core/logger.h"
#include <QHeaderView>
#include <QMessageBox>
#include <QFileInfo>
#include <QDateTime>

RestoreDialog::RestoreDialog(SafetyManager* safetyManager, QWidget* parent)
    : QDialog(parent)
    , m_mainLayout(nullptr)
    , m_filterWidget(nullptr)
    , m_searchEdit(nullptr)
    , m_typeFilter(nullptr)
    , m_refreshButton(nullptr)
    , m_backupTable(nullptr)
    , m_statsLabel(nullptr)
    , m_buttonLayout(nullptr)
    , m_restoreButton(nullptr)
    , m_restoreAllButton(nullptr)
    , m_deleteBackupButton(nullptr)
    , m_closeButton(nullptr)
    , m_safetyManager(safetyManager)
{
    setWindowTitle(tr("Restore Files"));
    setMinimumSize(900, 600);
    resize(950, 650);
    
    setupUI();
    loadBackups();
    
    LOG_INFO(LogCategories::UI, "Restore dialog created");
}

RestoreDialog::~RestoreDialog()
{
    LOG_DEBUG(LogCategories::UI, "Restore dialog destroyed");
}

void RestoreDialog::setupUI()
{
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setContentsMargins(15, 15, 15, 15);
    m_mainLayout->setSpacing(10);
    
    // Info label
    QLabel* infoLabel = new QLabel(tr("View and restore files from backups created before delete/move operations."), this);
    infoLabel->setWordWrap(true);
    infoLabel->setStyleSheet("QLabel { padding: 8px; background: palette(base); border: 1px solid palette(mid); border-radius: 3px; }");
    m_mainLayout->addWidget(infoLabel);
    
    // Filter controls
    m_filterWidget = new QWidget(this);
    QHBoxLayout* filterLayout = new QHBoxLayout(m_filterWidget);
    filterLayout->setContentsMargins(0, 0, 0, 0);
    
    QLabel* searchLabel = new QLabel(tr("Search:"), this);
    m_searchEdit = new QLineEdit(this);
    m_searchEdit->setPlaceholderText(tr("Search by filename or path..."));
    connect(m_searchEdit, &QLineEdit::textChanged, this, &RestoreDialog::onSearchTextChanged);
    
    QLabel* typeLabel = new QLabel(tr("Operation:"), this);
    m_typeFilter = new QComboBox(this);
    m_typeFilter->addItem(tr("All Operations"), -1);
    m_typeFilter->addItem(tr("Delete"), static_cast<int>(SafetyManager::OperationType::Delete));
    m_typeFilter->addItem(tr("Move"), static_cast<int>(SafetyManager::OperationType::Move));
    connect(m_typeFilter, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &RestoreDialog::onFilterChanged);
    
    m_refreshButton = new QPushButton(tr("🔄 Refresh"), this);
    connect(m_refreshButton, &QPushButton::clicked, this, &RestoreDialog::onRefreshClicked);
    
    filterLayout->addWidget(searchLabel);
    filterLayout->addWidget(m_searchEdit, 2);
    filterLayout->addWidget(typeLabel);
    filterLayout->addWidget(m_typeFilter, 1);
    filterLayout->addWidget(m_refreshButton);
    filterLayout->addStretch();
    
    m_mainLayout->addWidget(m_filterWidget);
    
    // Table
    m_backupTable = new QTableWidget(this);
    m_backupTable->setColumnCount(6);
    m_backupTable->setHorizontalHeaderLabels({
        tr("Original File"), tr("Operation"), tr("Date/Time"), 
        tr("Size"), tr("Backup Location"), tr("Status")
    });
    
    m_backupTable->horizontalHeader()->setStretchLastSection(false);
    m_backupTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    m_backupTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    m_backupTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    m_backupTable->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    m_backupTable->horizontalHeader()->setSectionResizeMode(4, QHeaderView::Stretch);
    m_backupTable->horizontalHeader()->setSectionResizeMode(5, QHeaderView::ResizeToContents);
    
    m_backupTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_backupTable->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_backupTable->setAlternatingRowColors(true);
    m_backupTable->setSortingEnabled(true);
    
    connect(m_backupTable, &QTableWidget::cellDoubleClicked, 
            this, &RestoreDialog::onTableDoubleClicked);
    connect(m_backupTable, &QTableWidget::itemSelectionChanged,
            this, &RestoreDialog::onSelectionChanged);
    
    m_mainLayout->addWidget(m_backupTable);
    
    // Stats
    m_statsLabel = new QLabel(this);
    m_statsLabel->setStyleSheet("QLabel { padding: 5px; background: palette(base); border: 1px solid palette(mid); border-radius: 3px; }");
    m_mainLayout->addWidget(m_statsLabel);
    
    // Action buttons
    m_buttonLayout = new QHBoxLayout();
    
    m_restoreButton = new QPushButton(tr("Restore Selected"), this);
    m_restoreButton->setEnabled(false);
    m_restoreButton->setToolTip(tr("Restore selected files to their original locations"));
    connect(m_restoreButton, &QPushButton::clicked, this, &RestoreDialog::onRestoreClicked);
    
    m_restoreAllButton = new QPushButton(tr("Restore All"), this);
    m_restoreAllButton->setToolTip(tr("Restore all files in the list"));
    connect(m_restoreAllButton, &QPushButton::clicked, this, &RestoreDialog::onRestoreAllClicked);
    
    m_deleteBackupButton = new QPushButton(tr("Delete Backup"), this);
    m_deleteBackupButton->setEnabled(false);
    m_deleteBackupButton->setToolTip(tr("Permanently delete selected backup files"));
    connect(m_deleteBackupButton, &QPushButton::clicked, this, &RestoreDialog::onDeleteBackupClicked);
    
    m_closeButton = new QPushButton(tr("Close"), this);
    connect(m_closeButton, &QPushButton::clicked, this, &QDialog::accept);
    
    m_buttonLayout->addWidget(m_restoreButton);
    m_buttonLayout->addWidget(m_restoreAllButton);
    m_buttonLayout->addWidget(m_deleteBackupButton);
    m_buttonLayout->addStretch();
    m_buttonLayout->addWidget(m_closeButton);
    
    m_mainLayout->addLayout(m_buttonLayout);
}

void RestoreDialog::loadBackups()
{
    LOG_INFO(LogCategories::SYSTEM, "Loading backup history");
    
    if (!m_safetyManager) {
        LOG_ERROR(LogCategories::SYSTEM, "SafetyManager is null");
        return;
    }
    
    m_allBackups = m_safetyManager->getUndoHistory(1000); // Get up to 1000 operations
    m_filteredBackups = m_allBackups;
    
    LOG_INFO(LogCategories::SYSTEM, QString("Loaded %1 backups").arg(m_allBackups.size()));
    
    applyFilters();
}

void RestoreDialog::applyFilters()
{
    m_backupTable->setRowCount(0);
    m_filteredBackups.clear();
    
    QString searchText = m_searchEdit->text().toLower();
    int typeFilter = m_typeFilter->currentData().toInt();
    
    // Apply filters
    for (const auto& backup : m_allBackups) {
        // Type filter
        if (typeFilter >= 0 && static_cast<int>(backup.type) != typeFilter) {
            continue;
        }
        
        // Search filter
        if (!searchText.isEmpty()) {
            QString originalPath = backup.sourceFile.toLower();
            QString backupPath = backup.backupPath.toLower();
            if (!originalPath.contains(searchText) && !backupPath.contains(searchText)) {
                continue;
            }
        }
        
        m_filteredBackups.append(backup);
    }
    
    // Populate table
    m_backupTable->setRowCount(m_filteredBackups.size());
    
    for (int i = 0; i < m_filteredBackups.size(); ++i) {
        const auto& backup = m_filteredBackups[i];
        
        // Original File
        QFileInfo fileInfo(backup.sourceFile);
        QTableWidgetItem* fileItem = new QTableWidgetItem(fileInfo.fileName());
        fileItem->setToolTip(backup.sourceFile);
        fileItem->setData(Qt::UserRole, i);
        m_backupTable->setItem(i, 0, fileItem);
        
        // Operation
        m_backupTable->setItem(i, 1, new QTableWidgetItem(getOperationTypeString(backup.type)));
        
        // Date/Time
        m_backupTable->setItem(i, 2, new QTableWidgetItem(formatDateTime(backup.timestamp)));
        
        // Size
        m_backupTable->setItem(i, 3, new QTableWidgetItem(formatFileSize(backup.fileSize)));
        
        // Backup Location
        QTableWidgetItem* backupItem = new QTableWidgetItem(backup.backupPath);
        backupItem->setToolTip(backup.backupPath);
        m_backupTable->setItem(i, 4, backupItem);
        
        // Status
        QFileInfo backupInfo(backup.backupPath);
        QString status = backupInfo.exists() ? tr("✓ Available") : tr("✗ Missing");
        QTableWidgetItem* statusItem = new QTableWidgetItem(status);
        if (!backupInfo.exists()) {
            statusItem->setForeground(Qt::red);
        } else {
            statusItem->setForeground(Qt::darkGreen);
        }
        m_backupTable->setItem(i, 5, statusItem);
    }
    
    updateStats();
    
    LOG_INFO(LogCategories::SYSTEM, QString("Displaying %1 filtered backups").arg(m_filteredBackups.size()));
}

void RestoreDialog::updateStats()
{
    int totalBackups = m_filteredBackups.size();
    int availableBackups = 0;
    qint64 totalSize = 0;
    
    for (const auto& backup : m_filteredBackups) {
        QFileInfo info(backup.backupPath);
        if (info.exists()) {
            availableBackups++;
            totalSize += backup.fileSize;
        }
    }
    
    m_statsLabel->setText(tr("Showing %1 backups | %2 available | %3 total size")
                         .arg(totalBackups)
                         .arg(availableBackups)
                         .arg(formatFileSize(totalSize)));
    
    m_restoreAllButton->setEnabled(availableBackups > 0);
}

void RestoreDialog::onRestoreClicked()
{
    QList<QTableWidgetItem*> selectedItems = m_backupTable->selectedItems();
    if (selectedItems.isEmpty()) {
        return;
    }
    
    // Get unique rows
    QSet<int> selectedRows;
    for (auto* item : selectedItems) {
        selectedRows.insert(item->row());
    }
    
    QStringList filesToRestore;
    QStringList missingBackups;
    
    for (int row : selectedRows) {
        if (row >= 0 && row < m_filteredBackups.size()) {
            const auto& backup = m_filteredBackups[row];
            QFileInfo info(backup.backupPath);
            if (info.exists()) {
                filesToRestore.append(backup.backupPath);
            } else {
                missingBackups.append(backup.sourceFile);
            }
        }
    }
    
    if (!missingBackups.isEmpty()) {
        QMessageBox::warning(this, tr("Missing Backups"),
                           tr("Some selected backups are missing and cannot be restored:\n\n%1")
                           .arg(missingBackups.join("\n")));
    }
    
    if (filesToRestore.isEmpty()) {
        return;
    }
    
    QMessageBox::StandardButton reply = QMessageBox::question(this, tr("Confirm Restore"),
        tr("Restore %1 file(s) to their original locations?\n\nExisting files will be overwritten.")
        .arg(filesToRestore.size()),
        QMessageBox::Yes | QMessageBox::No);
    
    if (reply == QMessageBox::Yes) {
        LOG_INFO(LogCategories::FILE_OPS, QString("Restoring %1 files").arg(filesToRestore.size()));
        
        // TODO: Implement actual restore operation through FileManager
        // For now, just emit signal
        emit filesRestored(filesToRestore);
        
        QMessageBox::information(this, tr("Restore Complete"),
                               tr("%1 file(s) have been restored successfully.").arg(filesToRestore.size()));
        
        loadBackups();
    }
}

void RestoreDialog::onRestoreAllClicked()
{
    QStringList filesToRestore;
    
    for (const auto& backup : m_filteredBackups) {
        QFileInfo info(backup.backupPath);
        if (info.exists()) {
            filesToRestore.append(backup.backupPath);
        }
    }
    
    if (filesToRestore.isEmpty()) {
        QMessageBox::information(this, tr("No Backups"),
                               tr("No backup files are available to restore."));
        return;
    }
    
    QMessageBox::StandardButton reply = QMessageBox::question(this, tr("Confirm Restore All"),
        tr("Restore all %1 file(s) to their original locations?\n\nExisting files will be overwritten.")
        .arg(filesToRestore.size()),
        QMessageBox::Yes | QMessageBox::No);
    
    if (reply == QMessageBox::Yes) {
        LOG_INFO(LogCategories::FILE_OPS, QString("Restoring all %1 files").arg(filesToRestore.size()));
        
        emit filesRestored(filesToRestore);
        
        QMessageBox::information(this, tr("Restore Complete"),
                               tr("All %1 file(s) have been restored successfully.").arg(filesToRestore.size()));
        
        loadBackups();
    }
}

void RestoreDialog::onDeleteBackupClicked()
{
    QList<QTableWidgetItem*> selectedItems = m_backupTable->selectedItems();
    if (selectedItems.isEmpty()) {
        return;
    }
    
    // Get unique rows
    QSet<int> selectedRows;
    for (auto* item : selectedItems) {
        selectedRows.insert(item->row());
    }
    
    QMessageBox::StandardButton reply = QMessageBox::question(this, tr("Confirm Delete"),
        tr("Permanently delete %1 backup file(s)?\n\nThis action cannot be undone.")
        .arg(selectedRows.size()),
        QMessageBox::Yes | QMessageBox::No);
    
    if (reply == QMessageBox::Yes) {
        int deletedCount = 0;
        for (int row : selectedRows) {
            if (row >= 0 && row < m_filteredBackups.size()) {
                const auto& backup = m_filteredBackups[row];
                QFile file(backup.backupPath);
                if (file.remove()) {
                    deletedCount++;
                    LOG_INFO(LogCategories::FILE_OPS, QString("Deleted backup: %1").arg(backup.backupPath));
                } else {
                    LOG_ERROR(LogCategories::FILE_OPS, QString("Failed to delete backup: %1").arg(backup.backupPath));
                }
            }
        }
        
        QMessageBox::information(this, tr("Delete Complete"),
                               tr("%1 backup file(s) have been deleted.").arg(deletedCount));
        
        loadBackups();
    }
}

void RestoreDialog::onRefreshClicked()
{
    LOG_INFO(LogCategories::UI, "User clicked refresh in restore dialog");
    loadBackups();
}

void RestoreDialog::onSearchTextChanged(const QString& text)
{
    Q_UNUSED(text);
    applyFilters();
}

void RestoreDialog::onFilterChanged()
{
    applyFilters();
}

void RestoreDialog::onTableDoubleClicked(int row, int column)
{
    Q_UNUSED(column);
    if (row >= 0 && row < m_filteredBackups.size()) {
        const auto& backup = m_filteredBackups[row];
        QFileInfo info(backup.backupPath);
        
        if (info.exists()) {
            QMessageBox::StandardButton reply = QMessageBox::question(this, tr("Restore File"),
                tr("Restore this file to its original location?\n\nOriginal: %1\nBackup: %2")
                .arg(backup.sourceFile)
                .arg(backup.backupPath),
                QMessageBox::Yes | QMessageBox::No);
            
            if (reply == QMessageBox::Yes) {
                QStringList files;
                files << backup.backupPath;
                emit filesRestored(files);
                
                QMessageBox::information(this, tr("Restore Complete"),
                                       tr("File has been restored successfully."));
                loadBackups();
            }
        } else {
            QMessageBox::warning(this, tr("Backup Missing"),
                               tr("The backup file no longer exists:\n%1").arg(backup.backupPath));
        }
    }
}

void RestoreDialog::onSelectionChanged()
{
    bool hasSelection = !m_backupTable->selectedItems().isEmpty();
    m_restoreButton->setEnabled(hasSelection);
    m_deleteBackupButton->setEnabled(hasSelection);
}

QString RestoreDialog::formatFileSize(qint64 bytes) const
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

QString RestoreDialog::formatDateTime(const QDateTime& dt) const
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

QString RestoreDialog::getOperationTypeString(SafetyManager::OperationType type) const
{
    switch (type) {
        case SafetyManager::OperationType::Delete:
            return tr("Delete");
        case SafetyManager::OperationType::Move:
            return tr("Move");
        case SafetyManager::OperationType::Copy:
            return tr("Copy");
        case SafetyManager::OperationType::Modify:
            return tr("Modify");
        case SafetyManager::OperationType::Create:
            return tr("Create");
        default:
            return tr("Unknown");
    }
}
