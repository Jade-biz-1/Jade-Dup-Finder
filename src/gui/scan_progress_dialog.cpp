#include "scan_progress_dialog.h"
#include "theme_manager.h"
#include "operation_manager.h"
#include <QGridLayout>
#include <QFrame>
#include <QGroupBox>
#include <QListWidget>
#include <QListWidgetItem>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QDateTime>
#include <cmath>

ScanProgressDialog::ScanProgressDialog(QWidget* parent)
    : QDialog(parent)
    , m_overallProgress(nullptr)
    , m_throughputProgress(nullptr)
    , m_filesLabel(nullptr)
    , m_sizeLabel(nullptr)
    , m_currentFolderLabel(nullptr)
    , m_currentFileLabel(nullptr)
    , m_rateLabel(nullptr)
    , m_throughputLabel(nullptr)
    , m_etaLabel(nullptr)
    , m_elapsedLabel(nullptr)
    , m_statusLabel(nullptr)
    , m_pauseButton(nullptr)
    , m_cancelButton(nullptr)
    , m_errorsLabel(nullptr)
    , m_viewErrorsButton(nullptr)
    , m_operationTypeLabel(nullptr)
    , m_operationStatusLabel(nullptr)
    , m_averageFileSizeLabel(nullptr)
    , m_bytesPerSecondLabel(nullptr)
    , m_queueProgress(nullptr)
    , m_queueStatusLabel(nullptr)

    , m_queueGroup(nullptr)
    , m_lastErrorLabel(nullptr)
    , m_isPaused(false)
    , m_operationManager(nullptr)
{
    setupUI();
    
    // Register with ThemeManager for automatic theme updates
    ThemeManager::instance()->registerDialog(this);
    m_scanTimer.start();
}

void ScanProgressDialog::setupUI() {
    setWindowTitle(tr("Operation Progress"));
    setModal(true);
    setMinimumWidth(600);
    setMinimumHeight(450);

    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(20, 20, 20, 20);

    // Enhanced status section with operation type and visual status
    auto* statusLayout = new QHBoxLayout();
    
    m_statusLabel = new QLabel(tr("Initializing..."), this);
    QFont statusFont = m_statusLabel->font();
    statusFont.setPointSize(statusFont.pointSize() + 2);
    statusFont.setBold(true);
    m_statusLabel->setFont(statusFont);
    
    m_operationTypeLabel = new QLabel(tr("Scan Operation"), this);
    QFont typeFont = m_operationTypeLabel->font();
    typeFont.setPointSize(typeFont.pointSize() + 1);
    m_operationTypeLabel->setFont(typeFont);
    
    m_operationStatusLabel = new QLabel(getStatusIcon(OperationStatus::Initializing) + " " + getStatusText(OperationStatus::Initializing), this);
    m_operationStatusLabel->setStyleSheet(QString("color: %1; font-weight: bold;")
                                         .arg(getStatusColor(OperationStatus::Initializing).name()));
    
    statusLayout->addWidget(m_operationTypeLabel);
    statusLayout->addStretch();
    statusLayout->addWidget(m_operationStatusLabel);
    
    mainLayout->addWidget(m_statusLabel);
    mainLayout->addLayout(statusLayout);

    // Progress section
    createProgressSection(mainLayout);

    // Enhanced metrics section (Task 7.1)
    createEnhancedMetricsSection(mainLayout);

    // Details section
    createDetailsSection(mainLayout);

    // Queue section (Task 7.2)
    createQueueSection(mainLayout);

    // Spacer
    mainLayout->addStretch();

    // Buttons section
    createButtonSection(mainLayout);

    // Enforce minimum sizes for all controls
    ThemeManager::instance()->enforceMinimumSizes(this);
    
    setLayout(mainLayout);
}

void ScanProgressDialog::createProgressSection(QVBoxLayout* mainLayout) {
    auto* progressGroup = new QGroupBox(tr("Overall Progress"), this);
    auto* progressLayout = new QVBoxLayout(progressGroup);

    // Enhanced progress bar with custom styling
    m_overallProgress = new QProgressBar(this);
    m_overallProgress->setMinimum(0);
    m_overallProgress->setMaximum(100);
    m_overallProgress->setValue(0);
    m_overallProgress->setTextVisible(true);
    m_overallProgress->setFormat("%p% - %v/%m files");
    // Apply theme-aware progress bar styling
    m_overallProgress->setStyleSheet(ThemeManager::instance()->getProgressBarStyle(ThemeManager::ProgressType::Normal));
    progressLayout->addWidget(m_overallProgress);

    // T12: Throughput indicator bar
    auto* throughputLabel = new QLabel(tr("Performance:"), this);
    throughputLabel->setFont(QFont(throughputLabel->font().family(), throughputLabel->font().pointSize() - 1));
    progressLayout->addWidget(throughputLabel);
    
    m_throughputProgress = new QProgressBar(this);
    m_throughputProgress->setMinimum(0);
    m_throughputProgress->setMaximum(100);
    m_throughputProgress->setValue(0);
    m_throughputProgress->setTextVisible(true);
    m_throughputProgress->setFormat("Performance: %p%");
    m_throughputProgress->setMaximumHeight(15);
    // Apply theme-aware performance progress bar styling
    m_throughputProgress->setStyleSheet(ThemeManager::instance()->getProgressBarStyle(ThemeManager::ProgressType::Performance));
    progressLayout->addWidget(m_throughputProgress);

    // Progress details grid
    auto* detailsGrid = new QGridLayout();
    detailsGrid->setColumnStretch(1, 1);
    detailsGrid->setHorizontalSpacing(10);
    detailsGrid->setVerticalSpacing(5);

    // Files scanned
    auto* filesLabelText = new QLabel(tr("Files:"), this);
    filesLabelText->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_filesLabel = new QLabel(tr("0"), this);
    detailsGrid->addWidget(filesLabelText, 0, 0);
    detailsGrid->addWidget(m_filesLabel, 0, 1);

    // Data scanned
    auto* sizeLabelText = new QLabel(tr("Data:"), this);
    sizeLabelText->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_sizeLabel = new QLabel(tr("0 B"), this);
    detailsGrid->addWidget(sizeLabelText, 1, 0);
    detailsGrid->addWidget(m_sizeLabel, 1, 1);

    // Scan rate
    auto* rateLabelText = new QLabel(tr("Rate:"), this);
    rateLabelText->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_rateLabel = new QLabel(tr("0 files/sec"), this);
    detailsGrid->addWidget(rateLabelText, 2, 0);
    detailsGrid->addWidget(m_rateLabel, 2, 1);

    // Data throughput (T12)
    auto* throughputLabelText = new QLabel(tr("Throughput:"), this);
    throughputLabelText->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_throughputLabel = new QLabel(tr("0 MB/s"), this);
    detailsGrid->addWidget(throughputLabelText, 3, 0);
    detailsGrid->addWidget(m_throughputLabel, 3, 1);

    // Elapsed time (T12)
    auto* elapsedLabelText = new QLabel(tr("Elapsed:"), this);
    elapsedLabelText->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_elapsedLabel = new QLabel(tr("0s"), this);
    detailsGrid->addWidget(elapsedLabelText, 4, 0);
    detailsGrid->addWidget(m_elapsedLabel, 4, 1);

    // ETA
    auto* etaLabelText = new QLabel(tr("ETA:"), this);
    etaLabelText->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_etaLabel = new QLabel(tr("Calculating..."), this);
    detailsGrid->addWidget(etaLabelText, 5, 0);
    detailsGrid->addWidget(m_etaLabel, 5, 1);

    // Errors (Task 10)
    auto* errorsLabelText = new QLabel(tr("Errors:"), this);
    errorsLabelText->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_errorsLabel = new QLabel(tr("0"), this);
    detailsGrid->addWidget(errorsLabelText, 6, 0);
    detailsGrid->addWidget(m_errorsLabel, 6, 1);

    progressLayout->addLayout(detailsGrid);
    mainLayout->addWidget(progressGroup);
}

void ScanProgressDialog::createDetailsSection(QVBoxLayout* mainLayout) {
    auto* detailsGroup = new QGroupBox(tr("Current Activity"), this);
    auto* detailsLayout = new QVBoxLayout(detailsGroup);

    // Current folder
    auto* folderLayout = new QHBoxLayout();
    auto* folderLabelText = new QLabel(tr("Folder:"), this);
    folderLabelText->setMinimumWidth(60);
    m_currentFolderLabel = new QLabel(tr("—"), this);
    m_currentFolderLabel->setWordWrap(true);
    m_currentFolderLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    folderLayout->addWidget(folderLabelText);
    folderLayout->addWidget(m_currentFolderLabel, 1);
    detailsLayout->addLayout(folderLayout);

    // Current file
    auto* fileLayout = new QHBoxLayout();
    auto* fileLabelText = new QLabel(tr("File:"), this);
    fileLabelText->setMinimumWidth(60);
    m_currentFileLabel = new QLabel(tr("—"), this);
    m_currentFileLabel->setWordWrap(true);
    m_currentFileLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    fileLayout->addWidget(fileLabelText);
    fileLayout->addWidget(m_currentFileLabel, 1);
    detailsLayout->addLayout(fileLayout);

    mainLayout->addWidget(detailsGroup);
}

void ScanProgressDialog::createEnhancedMetricsSection(QVBoxLayout* mainLayout) {
    auto* metricsGroup = new QGroupBox(tr("Performance Metrics"), this);
    auto* metricsLayout = new QGridLayout(metricsGroup);
    metricsLayout->setColumnStretch(1, 1);
    metricsLayout->setHorizontalSpacing(10);
    metricsLayout->setVerticalSpacing(5);

    // Enhanced data processing rate
    auto* bytesRateLabelText = new QLabel(tr("Data Rate:"), this);
    bytesRateLabelText->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_bytesPerSecondLabel = new QLabel(tr("0 B/s"), this);
    metricsLayout->addWidget(bytesRateLabelText, 0, 0);
    metricsLayout->addWidget(m_bytesPerSecondLabel, 0, 1);

    // Average file size
    auto* avgSizeLabelText = new QLabel(tr("Avg File Size:"), this);
    avgSizeLabelText->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_averageFileSizeLabel = new QLabel(tr("0 B"), this);
    metricsLayout->addWidget(avgSizeLabelText, 1, 0);
    metricsLayout->addWidget(m_averageFileSizeLabel, 1, 1);

    // Last error display
    auto* errorLabelText = new QLabel(tr("Last Error:"), this);
    errorLabelText->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    m_lastErrorLabel = new QLabel(tr("None"), this);
    m_lastErrorLabel->setWordWrap(true);
    m_lastErrorLabel->setStyleSheet("color: #d32f2f;"); // Error color - will be themed
    metricsLayout->addWidget(errorLabelText, 2, 0);
    metricsLayout->addWidget(m_lastErrorLabel, 2, 1);

    mainLayout->addWidget(metricsGroup);
}

void ScanProgressDialog::createQueueSection(QVBoxLayout* mainLayout) {
    m_queueGroup = new QGroupBox(tr("Operation Queue"), this);
    auto* queueLayout = new QVBoxLayout(m_queueGroup);

    // Queue progress bar
    auto* queueProgressLayout = new QHBoxLayout();
    auto* queueProgressLabel = new QLabel(tr("Queue Progress:"), this);
    m_queueProgress = new QProgressBar(this);
    m_queueProgress->setMinimum(0);
    m_queueProgress->setMaximum(100);
    m_queueProgress->setValue(0);
    m_queueProgress->setTextVisible(true);
    m_queueProgress->setFormat("%p% - %v/%m operations");
    m_queueProgress->setStyleSheet(ThemeManager::instance()->getProgressBarStyle(ThemeManager::ProgressType::Queue));
    
    queueProgressLayout->addWidget(queueProgressLabel);
    queueProgressLayout->addWidget(m_queueProgress, 1);
    queueLayout->addLayout(queueProgressLayout);

    // Queue status
    m_queueStatusLabel = new QLabel(tr("No operations in queue"), this);
    queueLayout->addWidget(m_queueStatusLabel);

    // Operation queue list
    m_operationQueueList = new QListWidget(this);
    m_operationQueueList->setMaximumHeight(100);
    m_operationQueueList->setAlternatingRowColors(true);
    queueLayout->addWidget(m_operationQueueList);

    // Initially hide queue section if no operations
    m_queueGroup->setVisible(false);
    
    mainLayout->addWidget(m_queueGroup);
}

void ScanProgressDialog::createButtonSection(QVBoxLayout* mainLayout) {
    auto* buttonLayout = new QHBoxLayout();
    buttonLayout->addStretch();

    // Pause/Resume button
    m_pauseButton = new QPushButton(tr("Pause"), this);
    m_pauseButton->setMinimumWidth(100);
    connect(m_pauseButton, &QPushButton::clicked, this, [this]() {
        if (m_isPaused) {
            emit resumeRequested();
        } else {
            emit pauseRequested();
        }
    });
    buttonLayout->addWidget(m_pauseButton);

    // View Errors button (Task 10)
    m_viewErrorsButton = new QPushButton(tr("View Errors"), this);
    m_viewErrorsButton->setMinimumWidth(100);
    m_viewErrorsButton->setEnabled(false); // Initially disabled
    connect(m_viewErrorsButton, &QPushButton::clicked, this, &ScanProgressDialog::viewErrorsRequested);
    buttonLayout->addWidget(m_viewErrorsButton);

    // Cancel button
    m_cancelButton = new QPushButton(tr("Cancel"), this);
    m_cancelButton->setMinimumWidth(100);
    connect(m_cancelButton, &QPushButton::clicked, this, &ScanProgressDialog::cancelRequested);
    buttonLayout->addWidget(m_cancelButton);

    mainLayout->addLayout(buttonLayout);
}

void ScanProgressDialog::updateProgress(const ProgressInfo& info) {
    // Update files count
    if (info.totalFiles > 0) {
        m_filesLabel->setText(tr("%1 / %2")
            .arg(info.filesScanned)
            .arg(info.totalFiles));
        
        // Update progress bar with enhanced format
        int percentage = (info.filesScanned * 100) / info.totalFiles;
        m_overallProgress->setValue(percentage);
        m_overallProgress->setMaximum(info.totalFiles);
        m_overallProgress->setValue(info.filesScanned);
    } else {
        m_filesLabel->setText(QString::number(info.filesScanned));
        m_overallProgress->setMaximum(0); // Indeterminate progress
    }

    // Update data scanned
    m_sizeLabel->setText(formatBytes(info.bytesScanned));

    // Update scan rate
    if (info.filesPerSecond > 0) {
        m_rateLabel->setText(tr("%1 files/sec")
            .arg(info.filesPerSecond, 0, 'f', 1));
    } else {
        m_rateLabel->setText(tr("0 files/sec"));
    }

    // T12: Update data throughput
    qint64 elapsedMs = m_scanTimer.elapsed();
    if (elapsedMs > 0 && info.bytesScanned > 0) {
        double bytesPerSecond = (info.bytesScanned * 1000.0) / elapsedMs;
        double mbPerSecond = bytesPerSecond / (1024.0 * 1024.0);
        m_throughputLabel->setText(tr("%1 MB/s").arg(mbPerSecond, 0, 'f', 2));
        
        // Update throughput progress bar (scale to reasonable max of 100 MB/s)
        int throughputPercentage = qMin(100, static_cast<int>(mbPerSecond));
        m_throughputProgress->setValue(throughputPercentage);
    } else {
        m_throughputLabel->setText(tr("0 MB/s"));
        m_throughputProgress->setValue(0);
    }

    // T12: Update elapsed time
    if (elapsedMs > 0) {
        int elapsedSeconds = elapsedMs / 1000;
        m_elapsedLabel->setText(formatTime(elapsedSeconds));
    }

    // Update current folder
    if (!info.currentFolder.isEmpty()) {
        m_currentFolderLabel->setText(info.currentFolder);
    } else {
        m_currentFolderLabel->setText(tr("—"));
    }

    // Update current file
    if (!info.currentFile.isEmpty()) {
        m_currentFileLabel->setText(info.currentFile);
    } else {
        m_currentFileLabel->setText(tr("—"));
    }

    // Update ETA
    updateETA(info);

    // Update error count (Task 10)
    m_errorsLabel->setText(QString::number(info.errorsEncountered));
    m_viewErrorsButton->setEnabled(info.errorsEncountered > 0);

    // Update paused state if changed
    if (info.isPaused != m_isPaused) {
        setPaused(info.isPaused);
    }
    
    // T12: Update visual feedback
    updateVisualFeedback(info);
    
    // Task 7.1: Update enhanced metrics
    updateEnhancedMetrics(info);
    
    // Task 7.2: Update queue display if operations are queued
    if (!info.operationQueue.isEmpty()) {
        updateQueueDisplay(info.operationQueue);
        updateOperationQueueProgress(info);
    }
    
    // Task 7.1: Update operation status display
    updateOperationStatusDisplay(info.status);
}

void ScanProgressDialog::updateEnhancedMetrics(const ProgressInfo& info) {
    // Update bytes per second rate
    if (info.bytesPerSecond > 0) {
        m_bytesPerSecondLabel->setText(formatBytes(static_cast<qint64>(info.bytesPerSecond)) + "/s");
    } else {
        m_bytesPerSecondLabel->setText(tr("0 B/s"));
    }
    
    // Update average file size
    if (info.averageFileSize > 0) {
        m_averageFileSizeLabel->setText(formatBytes(static_cast<qint64>(info.averageFileSize)));
    } else {
        m_averageFileSizeLabel->setText(tr("0 B"));
    }
    
    // Update last error display
    if (!info.lastError.isEmpty()) {
        m_lastErrorLabel->setText(info.lastError);
        m_lastErrorLabel->setStyleSheet("color: " + getStatusColor(OperationStatus::Error).name() + ";");
    } else {
        m_lastErrorLabel->setText(tr("None"));
        m_lastErrorLabel->setStyleSheet("color: " + ThemeManager::instance()->getCurrentThemeData().colors.foreground.name() + ";");
    }
}

void ScanProgressDialog::updateOperationStatusDisplay(OperationStatus status) {
    QString statusText = getStatusIcon(status) + " " + getStatusText(status);
    m_operationStatusLabel->setText(statusText);
    m_operationStatusLabel->setStyleSheet(QString("color: %1; font-weight: bold;")
                                         .arg(getStatusColor(status).name()));
}

void ScanProgressDialog::updateQueueDisplay(const QList<QueuedOperation>& operations) {
    m_operationQueueList->clear();
    
    for (const auto& operation : operations) {
        QString itemText = QString("%1: %2 (%3)")
                          .arg(operation.operationType)
                          .arg(operation.description)
                          .arg(getStatusText(operation.status));
        
        auto* item = new QListWidgetItem(itemText);
        
        // Set item color based on status
        QColor statusColor = getStatusColor(operation.status);
        item->setForeground(QBrush(statusColor));
        
        // Add status icon
        item->setText(getStatusIcon(operation.status) + " " + itemText);
        
        m_operationQueueList->addItem(item);
    }
    
    // Show queue section if there are operations
    m_queueGroup->setVisible(!operations.isEmpty());
    
    // Update queue status label
    if (operations.isEmpty()) {
        m_queueStatusLabel->setText(tr("No operations in queue"));
    } else {
        int completed = 0;
        int running = 0;
        int pending = 0;
        
        for (const auto& op : operations) {
            switch (op.status) {
                case OperationStatus::Completed:
                    completed++;
                    break;
                case OperationStatus::Running:
                    running++;
                    break;
                case OperationStatus::Initializing:
                    pending++;
                    break;
                default:
                    break;
            }
        }
        
        m_queueStatusLabel->setText(tr("%1 operations: %2 completed, %3 running, %4 pending")
                                   .arg(operations.size())
                                   .arg(completed)
                                   .arg(running)
                                   .arg(pending));
    }
}

void ScanProgressDialog::updateOperationQueueProgress(const ProgressInfo& info) {
    if (info.totalOperationsInQueue > 0) {
        int completedOperations = 0;
        for (const auto& operation : info.operationQueue) {
            if (operation.status == OperationStatus::Completed) {
                completedOperations++;
            }
        }
        
        int queuePercentage = (completedOperations * 100) / info.totalOperationsInQueue;
        m_queueProgress->setValue(queuePercentage);
        m_queueProgress->setMaximum(info.totalOperationsInQueue);
        m_queueProgress->setValue(completedOperations);
    } else {
        m_queueProgress->setValue(0);
        m_queueProgress->setMaximum(1);
    }
}

QString ScanProgressDialog::getStatusIcon(OperationStatus status) const {
    switch (status) {
        case OperationStatus::Initializing:
            return "⏳";
        case OperationStatus::Running:
            return "▶️";
        case OperationStatus::Paused:
            return "⏸️";
        case OperationStatus::Completed:
            return "✅";
        case OperationStatus::Error:
            return "❌";
        case OperationStatus::Cancelled:
            return "🚫";
        default:
            return "❓";
    }
}

QString ScanProgressDialog::getStatusText(OperationStatus status) const {
    switch (status) {
        case OperationStatus::Initializing:
            return tr("Initializing");
        case OperationStatus::Running:
            return tr("Running");
        case OperationStatus::Paused:
            return tr("Paused");
        case OperationStatus::Completed:
            return tr("Completed");
        case OperationStatus::Error:
            return tr("Error");
        case OperationStatus::Cancelled:
            return tr("Cancelled");
        default:
            return tr("Unknown");
    }
}

QColor ScanProgressDialog::getStatusColor(OperationStatus status) const {
    // Get theme-aware colors from ThemeManager
    auto themeData = ThemeManager::instance()->getCurrentThemeData();
    
    switch (status) {
        case OperationStatus::Initializing:
            return themeData.colors.info;
        case OperationStatus::Running:
            return themeData.colors.accent;
        case OperationStatus::Paused:
            return themeData.colors.warning;
        case OperationStatus::Completed:
            return themeData.colors.success;
        case OperationStatus::Error:
            return themeData.colors.error;
        case OperationStatus::Cancelled:
            return themeData.colors.disabled;
        default:
            return themeData.colors.foreground;
    }
}

void ScanProgressDialog::updateOperationQueue(const QList<QueuedOperation>& operations) {
    updateQueueDisplay(operations);
}

void ScanProgressDialog::setOperationStatus(OperationStatus status) {
    updateOperationStatusDisplay(status);
}

void ScanProgressDialog::setOperationManager(OperationManager* manager) {
    if (m_operationManager) {
        // Disconnect old manager
        disconnect(m_operationManager, nullptr, this, nullptr);
    }
    
    m_operationManager = manager;
    
    if (m_operationManager) {
        // Connect to operation manager signals for automatic updates
        connect(m_operationManager, &OperationManager::progressInfoUpdated,
                this, &ScanProgressDialog::updateProgress);
        connect(m_operationManager, &OperationManager::operationStarted,
                this, [this](const QString& operationId) {
                    Q_UNUSED(operationId)
                    if (m_operationManager) {
                        updateProgress(m_operationManager->getProgressInfo());
                    }
                });
        connect(m_operationManager, &OperationManager::operationCompleted,
                this, [this](const QString& operationId) {
                    Q_UNUSED(operationId)
                    if (m_operationManager) {
                        updateProgress(m_operationManager->getProgressInfo());
                    }
                });
    }
}

void ScanProgressDialog::setPaused(bool paused) {
    m_isPaused = paused;
    updateButtonStates();

    if (paused) {
        m_statusLabel->setText(tr("Paused"));
        m_etaLabel->setText(tr("—"));
    } else {
        m_statusLabel->setText(tr("Scanning..."));
    }
}

bool ScanProgressDialog::isPaused() const {
    return m_isPaused;
}

void ScanProgressDialog::updateETA(const ProgressInfo& info) {
    if (m_isPaused) {
        m_etaLabel->setText(tr("—"));
        return;
    }

    if (info.secondsRemaining >= 0) {
        // Use provided ETA
        m_etaLabel->setText(formatTime(info.secondsRemaining));
    } else if (info.totalFiles > 0 && info.filesPerSecond > 0) {
        // T12: Enhanced ETA calculation with rate smoothing
        qint64 currentTime = m_scanTimer.elapsed();
        
        // Add current rate to history
        m_recentRates.append(info.filesPerSecond);
        m_recentTimestamps.append(currentTime);
        
        // Keep only recent samples
        while (m_recentRates.size() > MAX_RATE_SAMPLES) {
            m_recentRates.removeFirst();
            m_recentTimestamps.removeFirst();
        }
        
        // Calculate smoothed rate
        double smoothedRate = info.filesPerSecond;
        if (m_recentRates.size() >= 3) {
            // Use weighted average of recent rates
            double totalWeight = 0.0;
            double weightedSum = 0.0;
            
            for (int i = 0; i < m_recentRates.size(); ++i) {
                double weight = i + 1; // More recent samples get higher weight
                weightedSum += m_recentRates[i] * weight;
                totalWeight += weight;
            }
            
            if (totalWeight > 0) {
                smoothedRate = weightedSum / totalWeight;
            }
        }
        
        // Calculate ETA with smoothed rate
        int eta = calculateETA(info.filesScanned, info.totalFiles, smoothedRate);
        if (eta >= 0) {
            // Add confidence indicator for ETA
            QString etaText = formatTime(eta);
            if (m_recentRates.size() < 5) {
                etaText += tr(" (est.)");
            }
            m_etaLabel->setText(etaText);
        } else {
            m_etaLabel->setText(tr("Calculating..."));
        }
    } else {
        m_etaLabel->setText(tr("Unknown"));
    }
}

void ScanProgressDialog::updateButtonStates() {
    if (m_isPaused) {
        m_pauseButton->setText(tr("Resume"));
    } else {
        m_pauseButton->setText(tr("Pause"));
    }
}

int ScanProgressDialog::calculateETA(int filesScanned, int totalFiles, double filesPerSecond) {
    // Validate inputs
    if (filesScanned < 0 || totalFiles <= 0 || filesPerSecond <= 0.0) {
        return -1; // Cannot calculate
    }

    // Already complete
    if (filesScanned >= totalFiles) {
        return 0;
    }

    // Calculate remaining files
    int remainingFiles = totalFiles - filesScanned;

    // Calculate ETA in seconds
    double etaSeconds = remainingFiles / filesPerSecond;

    // Round to nearest second
    return static_cast<int>(std::round(etaSeconds));
}

QString ScanProgressDialog::formatTime(int seconds) {
    if (seconds < 0) {
        return tr("Unknown");
    }

    if (seconds == 0) {
        return tr("Complete");
    }

    if (seconds < 1) {
        return tr("< 1s");
    }

    int hours = seconds / 3600;
    int minutes = (seconds % 3600) / 60;
    int secs = seconds % 60;

    QStringList parts;

    if (hours > 0) {
        parts.append(tr("%1h").arg(hours));
    }

    if (minutes > 0 || hours > 0) {
        parts.append(tr("%1m").arg(minutes));
    }

    if (secs > 0 || (hours == 0 && minutes == 0)) {
        parts.append(tr("%1s").arg(secs));
    }

    return parts.join(" ");
}

QString ScanProgressDialog::formatBytes(qint64 bytes) {
    if (bytes < 0) {
        return tr("0 B");
    }

    const qint64 KB = 1024;
    const qint64 MB = KB * 1024;
    const qint64 GB = MB * 1024;
    const qint64 TB = GB * 1024;

    if (bytes >= TB) {
        return tr("%1 TB").arg(static_cast<double>(bytes) / static_cast<double>(TB), 0, 'f', 2);
    } else if (bytes >= GB) {
        return tr("%1 GB").arg(static_cast<double>(bytes) / static_cast<double>(GB), 0, 'f', 2);
    } else if (bytes >= MB) {
        return tr("%1 MB").arg(static_cast<double>(bytes) / static_cast<double>(MB), 0, 'f', 2);
    } else if (bytes >= KB) {
        return tr("%1 KB").arg(static_cast<double>(bytes) / static_cast<double>(KB), 0, 'f', 2);
    } else {
        return tr("%1 B").arg(bytes);
    }
}

// T12: Enhanced visual feedback based on scan performance
void ScanProgressDialog::updateVisualFeedback(const ProgressInfo& info) {
    // Update progress bar color based on performance
    QString progressStyle;
    
    // Apply theme-aware progress styling based on performance
    ThemeManager::ProgressType progressType;
    if (info.filesPerSecond > 50) {
        // High performance - success
        progressType = ThemeManager::ProgressType::Success;
    } else if (info.filesPerSecond > 20) {
        // Medium performance - normal/performance
        progressType = ThemeManager::ProgressType::Performance;
    } else if (info.filesPerSecond > 5) {
        // Low performance - warning
        progressType = ThemeManager::ProgressType::Warning;
    } else {
        // Very low performance - error
        progressType = ThemeManager::ProgressType::Error;
    }
    
    // Apply theme-aware styling
    m_overallProgress->setStyleSheet(ThemeManager::instance()->getProgressBarStyle(progressType));
    
    // Update status text with performance indicator
    if (!m_isPaused) {
        QString statusText = tr("Scanning...");
        if (info.filesPerSecond > 50) {
            statusText += tr(" (High Speed)");
        } else if (info.filesPerSecond > 20) {
            statusText += tr(" (Good Speed)");
        } else if (info.filesPerSecond > 5) {
            statusText += tr(" (Moderate Speed)");
        } else if (info.filesPerSecond > 0) {
            statusText += tr(" (Slow)");
        }
        m_statusLabel->setText(statusText);
    }
}
