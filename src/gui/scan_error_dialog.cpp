#include "scan_error_dialog.h"
#include "theme_manager.h"
#include <QApplication>
#include <QClipboard>
#include <QMessageBox>
#include <QHeaderView>

ScanErrorDialog::ScanErrorDialog(QWidget* parent)
    : QDialog(parent)
    , m_errorTable(nullptr)
    , m_summaryLabel(nullptr)
    , m_copyAllButton(nullptr)
    , m_copySelectedButton(nullptr)
    , m_closeButton(nullptr)
{
    setupUI();
    
    // Register with ThemeManager for automatic theme updates
    ThemeManager::instance()->registerDialog(this);
}

void ScanErrorDialog::setupUI() {
    setWindowTitle(tr("Scan Errors"));
    setModal(true);
    setMinimumSize(800, 400);
    resize(1000, 500);

    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(20, 20, 20, 20);

    // Summary label
    m_summaryLabel = new QLabel(tr("No errors to display"), this);
    QFont summaryFont = m_summaryLabel->font();
    summaryFont.setBold(true);
    m_summaryLabel->setFont(summaryFont);
    mainLayout->addWidget(m_summaryLabel);

    // Error table
    setupTable();
    mainLayout->addWidget(m_errorTable);

    // Buttons
    auto* buttonLayout = new QHBoxLayout();
    
    m_copyAllButton = new QPushButton(tr("Copy All"), this);
    m_copyAllButton->setEnabled(false);
    connect(m_copyAllButton, &QPushButton::clicked, this, &ScanErrorDialog::copyToClipboard);
    buttonLayout->addWidget(m_copyAllButton);

    m_copySelectedButton = new QPushButton(tr("Copy Selected"), this);
    m_copySelectedButton->setEnabled(false);
    connect(m_copySelectedButton, &QPushButton::clicked, this, &ScanErrorDialog::copySelectedToClipboard);
    buttonLayout->addWidget(m_copySelectedButton);

    buttonLayout->addStretch();

    m_closeButton = new QPushButton(tr("Close"), this);
    connect(m_closeButton, &QPushButton::clicked, this, &QDialog::accept);
    buttonLayout->addWidget(m_closeButton);

    mainLayout->addLayout(buttonLayout);
}

void ScanErrorDialog::setupTable() {
    m_errorTable = new QTableWidget(this);
    m_errorTable->setColumnCount(4);
    
    QStringList headers;
    headers << tr("Time") << tr("Type") << tr("File Path") << tr("Description");
    m_errorTable->setHorizontalHeaderLabels(headers);

    // Configure table
    m_errorTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_errorTable->setAlternatingRowColors(true);
    m_errorTable->setSortingEnabled(true);
    m_errorTable->verticalHeader()->setVisible(false);

    // Set column widths
    QHeaderView* header = m_errorTable->horizontalHeader();
    header->setStretchLastSection(true);
    header->resizeSection(0, 120); // Time
    header->resizeSection(1, 120); // Type
    header->resizeSection(2, 300); // File Path

    // Connect selection change
    connect(m_errorTable->selectionModel(), &QItemSelectionModel::selectionChanged,
            this, [this]() {
                bool hasSelection = m_errorTable->selectionModel()->hasSelection();
                m_copySelectedButton->setEnabled(hasSelection);
            });
}

void ScanErrorDialog::setErrors(const QList<FileScanner::ScanErrorInfo>& errors) {
    clearErrors();

    if (errors.isEmpty()) {
        m_summaryLabel->setText(tr("No errors to display"));
        return;
    }

    m_summaryLabel->setText(tr("Found %1 error(s) during scan:").arg(errors.size()));

    m_errorTable->setRowCount(errors.size());

    for (int i = 0; i < errors.size(); ++i) {
        const auto& error = errors[i];

        // Time
        auto* timeItem = new QTableWidgetItem(formatTimestamp(error.timestamp));
        timeItem->setFlags(timeItem->flags() & ~Qt::ItemIsEditable);
        m_errorTable->setItem(i, 0, timeItem);

        // Type
        auto* typeItem = new QTableWidgetItem(formatErrorType(error.errorType));
        typeItem->setFlags(typeItem->flags() & ~Qt::ItemIsEditable);
        m_errorTable->setItem(i, 1, typeItem);

        // File Path
        auto* pathItem = new QTableWidgetItem(error.filePath);
        pathItem->setFlags(pathItem->flags() & ~Qt::ItemIsEditable);
        pathItem->setToolTip(error.filePath); // Full path in tooltip
        m_errorTable->setItem(i, 2, pathItem);

        // Description
        auto* descItem = new QTableWidgetItem(error.errorMessage);
        descItem->setFlags(descItem->flags() & ~Qt::ItemIsEditable);
        descItem->setToolTip(error.errorMessage);
        m_errorTable->setItem(i, 3, descItem);
    }

    m_copyAllButton->setEnabled(true);
    m_errorTable->resizeRowsToContents();
}

void ScanErrorDialog::clearErrors() {
    m_errorTable->setRowCount(0);
    m_copyAllButton->setEnabled(false);
    m_copySelectedButton->setEnabled(false);
    m_summaryLabel->setText(tr("No errors to display"));
}

void ScanErrorDialog::copyToClipboard() {
    QString text = errorsToText(false);
    if (!text.isEmpty()) {
        QApplication::clipboard()->setText(text);
        QMessageBox::information(this, tr("Copied"), tr("All errors copied to clipboard."));
    }
}

void ScanErrorDialog::copySelectedToClipboard() {
    QString text = errorsToText(true);
    if (!text.isEmpty()) {
        QApplication::clipboard()->setText(text);
        QMessageBox::information(this, tr("Copied"), tr("Selected errors copied to clipboard."));
    }
}

QString ScanErrorDialog::formatErrorType(FileScanner::ScanError errorType) const {
    switch (errorType) {
        case FileScanner::ScanError::None:
            return tr("None");
        case FileScanner::ScanError::PermissionDenied:
            return tr("Permission Denied");
        case FileScanner::ScanError::FileSystemError:
            return tr("File System Error");
        case FileScanner::ScanError::NetworkTimeout:
            return tr("Network Timeout");
        case FileScanner::ScanError::DiskReadError:
            return tr("Disk Read Error");
        case FileScanner::ScanError::PathTooLong:
            return tr("Path Too Long");
        case FileScanner::ScanError::UnknownError:
        default:
            return tr("Unknown Error");
    }
}

QString ScanErrorDialog::formatTimestamp(const QDateTime& timestamp) const {
    return timestamp.toString("hh:mm:ss");
}

QString ScanErrorDialog::errorsToText(bool selectedOnly) const {
    QStringList lines;
    lines << tr("Scan Errors Report");
    lines << tr("Generated: %1").arg(QDateTime::currentDateTime().toString());
    lines << "";

    QList<int> rows;
    if (selectedOnly) {
        QModelIndexList selected = m_errorTable->selectionModel()->selectedRows();
        for (const QModelIndex& index : selected) {
            rows << index.row();
        }
    } else {
        for (int i = 0; i < m_errorTable->rowCount(); ++i) {
            rows << i;
        }
    }

    for (int row : rows) {
        QString time = m_errorTable->item(row, 0)->text();
        QString type = m_errorTable->item(row, 1)->text();
        QString path = m_errorTable->item(row, 2)->text();
        QString desc = m_errorTable->item(row, 3)->text();

        lines << tr("[%1] %2: %3").arg(time, type, desc);
        lines << tr("  Path: %1").arg(path);
        lines << "";
    }

    return lines.join("\n");
}