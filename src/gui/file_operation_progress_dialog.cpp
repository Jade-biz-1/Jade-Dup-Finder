#include "file_operation_progress_dialog.h"
#include "theme_manager.h"
#include <QGridLayout>
#include <QGroupBox>
#include <QApplication>
#include <QFileInfo>

FileOperationProgressDialog::FileOperationProgressDialog(QWidget* parent)
    : QDialog(parent)
    , m_titleLabel(nullptr)
    , m_operationTypeLabel(nullptr)
    , m_statusLabel(nullptr)
    , m_fileProgressBar(nullptr)
    , m_fileProgressLabel(nullptr)
    , m_byteProgressBar(nullptr)
    , m_byteProgressLabel(nullptr)
    , m_currentFileLabel(nullptr)
    , m_timeElapsedLabel(nullptr)
    , m_timeRemainingLabel(nullptr)
    , m_speedLabel(nullptr)
    , m_cancelButton(nullptr)
    , m_closeButton(nullptr)
    , m_operationQueue(nullptr)
    , m_updateTimer(new QTimer(this))
{
    setupUI();
    
    // Register with ThemeManager for automatic theme updates
    ThemeManager::instance()->registerDialog(this);
    connectSignals();
    resetDisplay();
}

void FileOperationProgressDialog::setupUI() {
    setWindowTitle(tr("File Operation Progress"));
    setModal(true);
    setMinimumSize(500, 350);
    resize(600, 400);

    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(20, 20, 20, 20);

    // Title and operation info
    m_titleLabel = new QLabel(tr("File Operation in Progress"), this);
    QFont titleFont = m_titleLabel->font();
    titleFont.setBold(true);
    titleFont.setPointSize(titleFont.pointSize() + 2);
    m_titleLabel->setFont(titleFont);
    m_titleLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(m_titleLabel);

    // Operation details group
    auto* detailsGroup = new QGroupBox(tr("Operation Details"), this);
    auto* detailsLayout = new QGridLayout(detailsGroup);
    
    detailsLayout->addWidget(new QLabel(tr("Type:"), this), 0, 0);
    m_operationTypeLabel = new QLabel(tr("Unknown"), this);
    detailsLayout->addWidget(m_operationTypeLabel, 0, 1);
    
    detailsLayout->addWidget(new QLabel(tr("Status:"), this), 1, 0);
    m_statusLabel = new QLabel(tr("Preparing..."), this);
    detailsLayout->addWidget(m_statusLabel, 1, 1);
    
    mainLayout->addWidget(detailsGroup);

    // Progress group
    auto* progressGroup = new QGroupBox(tr("Progress"), this);
    auto* progressLayout = new QGridLayout(progressGroup);
    
    // File progress
    progressLayout->addWidget(new QLabel(tr("Files:"), this), 0, 0);
    m_fileProgressBar = new QProgressBar(this);
    m_fileProgressBar->setRange(0, 100);
    progressLayout->addWidget(m_fileProgressBar, 0, 1);
    m_fileProgressLabel = new QLabel(tr("0 / 0"), this);
    progressLayout->addWidget(m_fileProgressLabel, 0, 2);
    
    // Byte progress
    progressLayout->addWidget(new QLabel(tr("Data:"), this), 1, 0);
    m_byteProgressBar = new QProgressBar(this);
    m_byteProgressBar->setRange(0, 100);
    progressLayout->addWidget(m_byteProgressBar, 1, 1);
    m_byteProgressLabel = new QLabel(tr("0 B / 0 B"), this);
    progressLayout->addWidget(m_byteProgressLabel, 1, 2);
    
    mainLayout->addWidget(progressGroup);

    // Current file and timing info
    auto* infoGroup = new QGroupBox(tr("Current Status"), this);
    auto* infoLayout = new QGridLayout(infoGroup);
    
    infoLayout->addWidget(new QLabel(tr("Current file:"), this), 0, 0);
    m_currentFileLabel = new QLabel(tr("None"), this);
    m_currentFileLabel->setWordWrap(true);
    // Apply monospace font for better path readability
    QFont monoFont("monospace");
    m_currentFileLabel->setFont(monoFont);
    infoLayout->addWidget(m_currentFileLabel, 0, 1);
    
    infoLayout->addWidget(new QLabel(tr("Time elapsed:"), this), 1, 0);
    m_timeElapsedLabel = new QLabel(tr("00:00:00"), this);
    infoLayout->addWidget(m_timeElapsedLabel, 1, 1);
    
    infoLayout->addWidget(new QLabel(tr("Time remaining:"), this), 2, 0);
    m_timeRemainingLabel = new QLabel(tr("Calculating..."), this);
    infoLayout->addWidget(m_timeRemainingLabel, 2, 1);
    
    infoLayout->addWidget(new QLabel(tr("Speed:"), this), 3, 0);
    m_speedLabel = new QLabel(tr("0 files/sec, 0 B/sec"), this);
    infoLayout->addWidget(m_speedLabel, 3, 1);
    
    mainLayout->addWidget(infoGroup);

    mainLayout->addStretch();

    // Buttons
    auto* buttonLayout = new QHBoxLayout();
    
    m_cancelButton = new QPushButton(tr("Cancel"), this);
    m_cancelButton->setEnabled(false);
    buttonLayout->addWidget(m_cancelButton);
    
    buttonLayout->addStretch();
    
    m_closeButton = new QPushButton(tr("Close"), this);
    m_closeButton->setEnabled(false);
    buttonLayout->addWidget(m_closeButton);
    
    mainLayout->addLayout(buttonLayout);
    
    // Enforce minimum sizes for all controls
    ThemeManager::instance()->enforceMinimumSizes(this);
}

void FileOperationProgressDialog::connectSignals() {
    connect(m_cancelButton, &QPushButton::clicked, this, &FileOperationProgressDialog::onCancelClicked);
    connect(m_closeButton, &QPushButton::clicked, this, &QDialog::accept);
    
    m_updateTimer->setInterval(100); // Update every 100ms
    connect(m_updateTimer, &QTimer::timeout, this, &FileOperationProgressDialog::updateProgress);
}

void FileOperationProgressDialog::setOperationQueue(FileOperationQueue* queue) {
    if (m_operationQueue) {
        disconnect(m_operationQueue, nullptr, this, nullptr);
    }
    
    m_operationQueue = queue;
    
    if (m_operationQueue) {
        connect(m_operationQueue, &FileOperationQueue::operationStarted,
                this, &FileOperationProgressDialog::onOperationStarted);
        connect(m_operationQueue, &FileOperationQueue::operationCompleted,
                this, &FileOperationProgressDialog::onOperationCompleted);
        connect(m_operationQueue, &FileOperationQueue::operationCancelled,
                this, &FileOperationProgressDialog::onOperationCancelled);
    }
}

void FileOperationProgressDialog::showForOperation(const QString& operationId) {
    m_currentOperationId = operationId;
    resetDisplay();
    
    if (m_operationQueue) {
        m_cancelButton->setEnabled(true);
        m_updateTimer->start();
    }
    
    show();
    raise();
    activateWindow();
}

void FileOperationProgressDialog::updateProgress() {
    if (!m_operationQueue || m_currentOperationId.isEmpty()) {
        return;
    }
    
    FileOperationQueue::OperationProgress progress = m_operationQueue->getCurrentOperationProgress();
    
    if (progress.operationId != m_currentOperationId) {
        return; // Not our operation
    }
    
    updateOperationInfo(progress);
}

void FileOperationProgressDialog::onCancelClicked() {
    if (!m_currentOperationId.isEmpty()) {
        emit cancelRequested(m_currentOperationId);
        m_cancelButton->setEnabled(false);
        m_statusLabel->setText(tr("Cancelling..."));
    }
}

void FileOperationProgressDialog::onOperationStarted(const QString& operationId) {
    if (operationId == m_currentOperationId) {
        m_statusLabel->setText(tr("In Progress"));
        m_cancelButton->setEnabled(true);
    }
}

void FileOperationProgressDialog::onOperationCompleted(const QString& operationId, bool success, const QString& errorMessage) {
    if (operationId != m_currentOperationId) {
        return;
    }
    
    m_updateTimer->stop();
    m_cancelButton->setEnabled(false);
    m_closeButton->setEnabled(true);
    
    if (success) {
        m_statusLabel->setText(tr("Completed Successfully"));
        m_titleLabel->setText(tr("Operation Completed"));
        
        // Set progress bars to 100%
        m_fileProgressBar->setValue(100);
        m_byteProgressBar->setValue(100);
        
        m_currentFileLabel->setText(tr("All files processed"));
        m_timeRemainingLabel->setText(tr("Complete"));
    } else {
        m_statusLabel->setText(tr("Failed: %1").arg(errorMessage));
        m_titleLabel->setText(tr("Operation Failed"));
        m_currentFileLabel->setText(tr("Error occurred"));
    }
}

void FileOperationProgressDialog::onOperationCancelled(const QString& operationId) {
    if (operationId != m_currentOperationId) {
        return;
    }
    
    m_updateTimer->stop();
    m_cancelButton->setEnabled(false);
    m_closeButton->setEnabled(true);
    
    m_statusLabel->setText(tr("Cancelled"));
    m_titleLabel->setText(tr("Operation Cancelled"));
    m_currentFileLabel->setText(tr("Operation was cancelled"));
    m_timeRemainingLabel->setText(tr("Cancelled"));
}

void FileOperationProgressDialog::resetDisplay() {
    m_operationTypeLabel->setText(tr("Unknown"));
    m_statusLabel->setText(tr("Preparing..."));
    
    m_fileProgressBar->setValue(0);
    m_fileProgressLabel->setText(tr("0 / 0"));
    
    m_byteProgressBar->setValue(0);
    m_byteProgressLabel->setText(tr("0 B / 0 B"));
    
    m_currentFileLabel->setText(tr("None"));
    m_timeElapsedLabel->setText(tr("00:00:00"));
    m_timeRemainingLabel->setText(tr("Calculating..."));
    m_speedLabel->setText(tr("0 files/sec, 0 B/sec"));
    
    m_cancelButton->setEnabled(false);
    m_closeButton->setEnabled(false);
}

void FileOperationProgressDialog::updateOperationInfo(const FileOperationQueue::OperationProgress& progress) {
    // Update operation type and status
    m_operationTypeLabel->setText(formatOperationType(progress.type));
    m_statusLabel->setText(formatOperationStatus(progress.status));
    
    // Update file progress
    if (progress.totalFiles > 0) {
        int filePercent = static_cast<int>((double)progress.filesProcessed / progress.totalFiles * 100);
        m_fileProgressBar->setValue(filePercent);
        m_fileProgressLabel->setText(tr("%1 / %2").arg(progress.filesProcessed).arg(progress.totalFiles));
    }
    
    // Update byte progress
    if (progress.totalBytes > 0) {
        int bytePercent = static_cast<int>((double)progress.bytesProcessed / progress.totalBytes * 100);
        m_byteProgressBar->setValue(bytePercent);
        m_byteProgressLabel->setText(tr("%1 / %2")
                                   .arg(formatFileSize(progress.bytesProcessed))
                                   .arg(formatFileSize(progress.totalBytes)));
    }
    
    // Update current file
    if (!progress.currentFile.isEmpty()) {
        QFileInfo fileInfo(progress.currentFile);
        m_currentFileLabel->setText(fileInfo.fileName());
        m_currentFileLabel->setToolTip(progress.currentFile);
    } else {
        m_currentFileLabel->setText(tr("Processing..."));
        m_currentFileLabel->setToolTip(QString());
    }
    
    // Update timing
    m_timeElapsedLabel->setText(formatTime(progress.elapsedTimeMs));
    
    if (progress.estimatedTimeRemainingMs >= 0) {
        m_timeRemainingLabel->setText(formatTime(progress.estimatedTimeRemainingMs));
    } else {
        m_timeRemainingLabel->setText(tr("Calculating..."));
    }
    
    // Update speed
    QString speedText = tr("%1 files/sec, %2")
                       .arg(progress.filesPerSecond, 0, 'f', 1)
                       .arg(formatSpeed(progress.bytesPerSecond));
    m_speedLabel->setText(speedText);
}

QString FileOperationProgressDialog::formatOperationType(FileOperationQueue::OperationType type) const {
    switch (type) {
        case FileOperationQueue::OperationType::Delete:
            return tr("Delete Files");
        case FileOperationQueue::OperationType::Move:
            return tr("Move Files");
        case FileOperationQueue::OperationType::Copy:
            return tr("Copy Files");
        case FileOperationQueue::OperationType::Trash:
            return tr("Move to Trash");
    }
    return tr("Unknown Operation");
}

QString FileOperationProgressDialog::formatOperationStatus(FileOperationQueue::OperationStatus status) const {
    switch (status) {
        case FileOperationQueue::OperationStatus::Pending:
            return tr("Pending");
        case FileOperationQueue::OperationStatus::InProgress:
            return tr("In Progress");
        case FileOperationQueue::OperationStatus::Completed:
            return tr("Completed");
        case FileOperationQueue::OperationStatus::Failed:
            return tr("Failed");
        case FileOperationQueue::OperationStatus::Cancelled:
            return tr("Cancelled");
    }
    return tr("Unknown Status");
}

QString FileOperationProgressDialog::formatFileSize(qint64 bytes) const {
    const qint64 KB = 1024;
    const qint64 MB = KB * 1024;
    const qint64 GB = MB * 1024;
    const qint64 TB = GB * 1024;
    
    if (bytes >= TB) {
        return tr("%1 TB").arg(bytes / (double)TB, 0, 'f', 2);
    } else if (bytes >= GB) {
        return tr("%1 GB").arg(bytes / (double)GB, 0, 'f', 2);
    } else if (bytes >= MB) {
        return tr("%1 MB").arg(bytes / (double)MB, 0, 'f', 1);
    } else if (bytes >= KB) {
        return tr("%1 KB").arg(bytes / (double)KB, 0, 'f', 1);
    } else {
        return tr("%1 B").arg(bytes);
    }
}

QString FileOperationProgressDialog::formatTime(qint64 milliseconds) const {
    qint64 seconds = milliseconds / 1000;
    qint64 minutes = seconds / 60;
    qint64 hours = minutes / 60;
    
    seconds %= 60;
    minutes %= 60;
    
    return tr("%1:%2:%3")
           .arg(hours, 2, 10, QChar('0'))
           .arg(minutes, 2, 10, QChar('0'))
           .arg(seconds, 2, 10, QChar('0'));
}

QString FileOperationProgressDialog::formatSpeed(double bytesPerSecond) const {
    const double KB = 1024.0;
    const double MB = KB * 1024.0;
    const double GB = MB * 1024.0;
    
    if (bytesPerSecond >= GB) {
        return tr("%1 GB/s").arg(bytesPerSecond / GB, 0, 'f', 2);
    } else if (bytesPerSecond >= MB) {
        return tr("%1 MB/s").arg(bytesPerSecond / MB, 0, 'f', 1);
    } else if (bytesPerSecond >= KB) {
        return tr("%1 KB/s").arg(bytesPerSecond / KB, 0, 'f', 1);
    } else {
        return tr("%1 B/s").arg(bytesPerSecond, 0, 'f', 0);
    }
}