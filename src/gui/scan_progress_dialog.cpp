#include "scan_progress_dialog.h"
#include <QGridLayout>
#include <QFrame>
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
    , m_isPaused(false)
{
    setupUI();
    m_scanTimer.start();
}

void ScanProgressDialog::setupUI() {
    setWindowTitle(tr("Scan Progress"));
    setModal(true);
    setMinimumWidth(500);
    setMinimumHeight(300);

    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(15);
    mainLayout->setContentsMargins(20, 20, 20, 20);

    // Status label at the top
    m_statusLabel = new QLabel(tr("Scanning..."), this);
    QFont statusFont = m_statusLabel->font();
    statusFont.setPointSize(statusFont.pointSize() + 2);
    statusFont.setBold(true);
    m_statusLabel->setFont(statusFont);
    mainLayout->addWidget(m_statusLabel);

    // Progress section
    createProgressSection(mainLayout);

    // Details section
    createDetailsSection(mainLayout);

    // Spacer
    mainLayout->addStretch();

    // Buttons section
    createButtonSection(mainLayout);

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
    m_overallProgress->setStyleSheet(
        "QProgressBar {"
        "    border: 2px solid grey;"
        "    border-radius: 5px;"
        "    text-align: center;"
        "    font-weight: bold;"
        "    height: 25px;"
        "}"
        "QProgressBar::chunk {"
        "    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
        "        stop:0 #4CAF50, stop:1 #45a049);"
        "    border-radius: 3px;"
        "}"
    );
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
    m_throughputProgress->setStyleSheet(
        "QProgressBar {"
        "    border: 1px solid grey;"
        "    border-radius: 3px;"
        "    text-align: center;"
        "    font-size: 10px;"
        "    height: 15px;"
        "}"
        "QProgressBar::chunk {"
        "    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
        "        stop:0 #2196F3, stop:1 #1976D2);"
        "    border-radius: 2px;"
        "}"
    );
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
    
    if (info.filesPerSecond > 50) {
        // High performance - green
        progressStyle = 
            "QProgressBar::chunk {"
            "    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "        stop:0 #4CAF50, stop:1 #45a049);"
            "    border-radius: 3px;"
            "}";
    } else if (info.filesPerSecond > 20) {
        // Medium performance - blue
        progressStyle = 
            "QProgressBar::chunk {"
            "    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "        stop:0 #2196F3, stop:1 #1976D2);"
            "    border-radius: 3px;"
            "}";
    } else if (info.filesPerSecond > 5) {
        // Low performance - orange
        progressStyle = 
            "QProgressBar::chunk {"
            "    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "        stop:0 #FF9800, stop:1 #F57C00);"
            "    border-radius: 3px;"
            "}";
    } else {
        // Very low performance - red
        progressStyle = 
            "QProgressBar::chunk {"
            "    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "        stop:0 #F44336, stop:1 #D32F2F);"
            "    border-radius: 3px;"
            "}";
    }
    
    // Apply the style while preserving the base style
    QString fullStyle = 
        "QProgressBar {"
        "    border: 2px solid grey;"
        "    border-radius: 5px;"
        "    text-align: center;"
        "    font-weight: bold;"
        "    height: 25px;"
        "}" + progressStyle;
    
    m_overallProgress->setStyleSheet(fullStyle);
    
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
