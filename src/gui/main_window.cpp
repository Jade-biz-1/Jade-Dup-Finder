#include "main_window.h"
#include "scan_dialog.h"
#include "results_window.h"
#include "file_scanner.h"
#include "app_config.h"
#include <QtWidgets/QApplication>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QListWidgetItem>
#include <QtCore/QSettings>
#include <QtCore/QStandardPaths>
#include <QtCore/QStorageInfo>
#include <QtCore/QDir>
#include <QtCore/QDebug>
#include <QtGui/QIcon>
#include <QtGui/QCloseEvent>
#include <QtGui/QPalette>

// Constructor
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_centralWidget(nullptr)
    , m_mainLayout(nullptr)
    , m_headerWidget(nullptr)
    , m_headerLayout(nullptr)
    , m_newScanButton(nullptr)
    , m_settingsButton(nullptr)
    , m_helpButton(nullptr)
    , m_notificationIndicator(nullptr)
    , m_planIndicator(nullptr)
    , m_quickActions(nullptr)
    , m_scanHistory(nullptr)
    , m_systemOverview(nullptr)
    , m_statusBar(nullptr)
    , m_progressBar(nullptr)
    , m_statusLabel(nullptr)
    , m_fileCountLabel(nullptr)
    , m_groupCountLabel(nullptr)
    , m_savingsLabel(nullptr)
    , m_fileScanner(nullptr)
    , m_duplicateDetector(nullptr)
    , m_hashCalculator(nullptr)
    , m_safetyManager(nullptr)
    , m_scanSetupDialog(nullptr)
    , m_resultsWindow(nullptr)
    , m_systemUpdateTimer(new QTimer(this))
{
    setWindowTitle(tr("DupFinder - Duplicate File Finder"));
    setMinimumSize(800, 600);
    resize(1024, 768);
    
    // Set application icon
    setWindowIcon(QIcon(":/icons/app-icon.png"));
    
    initializeUI();
    setupConnections();
    loadSettings();
    
    // Start system stats timer
    m_systemUpdateTimer->setInterval(30000); // Update every 30 seconds
    m_systemUpdateTimer->start();
    
    // Initial system stats update
    QTimer::singleShot(1000, this, &MainWindow::refreshSystemStats);
}

MainWindow::~MainWindow()
{
    saveSettings();
}

// Core component integration
void MainWindow::setFileScanner(FileScanner* scanner)
{
    m_fileScanner = scanner;
}

void MainWindow::setHashCalculator(HashCalculator* calculator)
{
    m_hashCalculator = calculator;
}

void MainWindow::setDuplicateDetector(DuplicateDetector* detector)
{
    m_duplicateDetector = detector;
}

void MainWindow::setSafetyManager(SafetyManager* manager)
{
    m_safetyManager = manager;
}

// Status management
void MainWindow::updateScanProgress(int percentage, const QString& status)
{
    if (m_progressBar) {
        m_progressBar->setValue(percentage);
        m_progressBar->setVisible(percentage > 0 && percentage < 100);
    }
    
    if (m_statusLabel) {
        m_statusLabel->setText(status);
    }
}

void MainWindow::showScanResults()
{
    qDebug() << "showScanResults called - opening results window";
    
    if (!m_resultsWindow) {
        m_resultsWindow = new ResultsWindow(this);
        
        // Connect results window signals
        connect(m_resultsWindow, &ResultsWindow::windowClosed,
                this, [this]() {
                    qDebug() << "Results window closed";
                    // Results window will be deleted when parent (this) is deleted
                });
        
        connect(m_resultsWindow, &ResultsWindow::fileOperationRequested,
                this, [this](const QString& operation, const QStringList& files) {
                    qDebug() << "File operation requested:" << operation << "on" << files.size() << "files";
                    // TODO: Forward to appropriate file operation handler
                });
        
        connect(m_resultsWindow, &ResultsWindow::resultsUpdated,
                this, [this](const ResultsWindow::ScanResults& results) {
                    qDebug() << "Results updated: " << results.duplicateGroups.size() << "groups";
                    
                    // Update main window stats
                    if (m_fileCountLabel) {
                        m_fileCountLabel->setText(tr("Files: %1").arg(results.totalFilesScanned));
                    }
                    if (m_groupCountLabel) {
                        m_groupCountLabel->setText(tr("Groups: %1").arg(results.duplicateGroups.size()));
                    }
                    if (m_savingsLabel) {
                        m_savingsLabel->setText(tr("Savings: %1").arg(formatFileSize(results.potentialSavings)));
                    }
                });
    }
    
    // Show the results window
    m_resultsWindow->show();
    m_resultsWindow->raise();
    m_resultsWindow->activateWindow();
}

void MainWindow::showError(const QString& title, const QString& message)
{
    QMessageBox::critical(this, title, message);
}

void MainWindow::showSuccess(const QString& title, const QString& message)
{
    QMessageBox::information(this, title, message);
}

// Public slots
void MainWindow::onNewScanRequested()
{
    LOG_INFO("User clicked 'New Scan' button");
    
    if (!m_scanSetupDialog) {
        LOG_DEBUG("Creating new ScanSetupDialog");
        m_scanSetupDialog = new ScanSetupDialog(this);
        
        // Connect scan configuration signal
        connect(m_scanSetupDialog, &ScanSetupDialog::scanConfigured,
                this, &MainWindow::handleScanConfiguration);
        
        // Connect preset saved signal
        connect(m_scanSetupDialog, &ScanSetupDialog::presetSaved,
                this, [this](const QString& name) {
                    LOG_INFO(QString("Preset saved: %1").arg(name));
                    // Preset is automatically saved by the dialog
                });
    }
    
    LOG_DEBUG("Showing scan setup dialog");
    m_scanSetupDialog->show();
    m_scanSetupDialog->raise();
    m_scanSetupDialog->activateWindow();
}

void MainWindow::onPresetSelected(const QString& preset)
{
    LOG_INFO(QString("User selected preset: %1").arg(preset));
    emit scanRequested(preset);
}

void MainWindow::onSettingsRequested()
{
    LOG_INFO("User clicked 'Settings' button");
    emit settingsRequested();
}

void MainWindow::onHelpRequested()
{
    LOG_INFO("User clicked 'Help' button");
    emit helpRequested();
}

void MainWindow::updateSystemInfo()
{
    LOG_DEBUG("Updating system information");
    refreshSystemStats();
}

void MainWindow::onScanHistoryItemClicked(int index)
{
    LOG_INFO(QString("User clicked history item: %1").arg(index));
    
    // Get the history item and load its results
    if (m_scanHistory && index >= 0) {
        QList<ScanHistoryWidget::ScanHistoryItem> history = m_scanHistory->getHistory();
        if (index < history.size()) {
            const auto& item = history[index];
            LOG_INFO(QString("Loading scan results for: %1").arg(item.scanId));
            
            // TODO: Load the actual scan results from storage
            // For now, show the results window
            showScanResults();
        }
    }
}

void MainWindow::onViewAllHistoryClicked()
{
    LOG_INFO("User clicked 'View All History'");
    
    // Show a dialog or window with full scan history
    QMessageBox::information(this, tr("Scan History"),
                           tr("Full scan history view will be implemented.\n"
                              "This will show all past scans with detailed information."));
}

// Protected methods
void MainWindow::closeEvent(QCloseEvent* event)
{
    saveSettings();
    emit applicationExit();
    event->accept();
}

void MainWindow::changeEvent(QEvent* event)
{
    if (event->type() == QEvent::WindowStateChange) {
        // Handle window state changes if needed
    }
    QMainWindow::changeEvent(event);
}

// Private slots
void MainWindow::initializeUI()
{
    // Create central widget and main layout
    m_centralWidget = new QWidget(this);
    setCentralWidget(m_centralWidget);
    
    m_mainLayout = new QVBoxLayout(m_centralWidget);
    m_mainLayout->setContentsMargins(12, 12, 12, 12);
    m_mainLayout->setSpacing(12);
    
    createHeaderWidget();
    createContentWidgets();
    createStatusBar();
    applyTheme();
}

void MainWindow::setupConnections()
{
    // Header button connections
    connect(m_newScanButton, &QPushButton::clicked, this, &MainWindow::onNewScanRequested);
    connect(m_settingsButton, &QPushButton::clicked, this, &MainWindow::onSettingsRequested);
    connect(m_helpButton, &QPushButton::clicked, this, &MainWindow::onHelpRequested);
    
    // Quick actions connections
    if (m_quickActions) {
        connect(m_quickActions, &QuickActionsWidget::presetSelected, this, &MainWindow::onPresetSelected);
    }
    
    // Scan history connections
    if (m_scanHistory) {
        connect(m_scanHistory, &ScanHistoryWidget::historyItemClicked, this, &MainWindow::onScanHistoryItemClicked);
        connect(m_scanHistory, &ScanHistoryWidget::viewAllRequested, this, &MainWindow::onViewAllHistoryClicked);
    }
    
    // FileScanner connections
    if (m_fileScanner) {
        connect(m_fileScanner, &FileScanner::scanStarted, this, [this]() {
            LOG_INFO("=== FileScanner: Scan Started ===");
            updateScanProgress(0, tr("Scanning..."));
            if (m_progressBar) m_progressBar->setVisible(true);
        });
        
        connect(m_fileScanner, &FileScanner::scanProgress, this, [this](int filesProcessed, int totalFiles, const QString& currentPath) {
            Q_UNUSED(totalFiles);
            QString status = tr("Scanning... %1 files found").arg(filesProcessed);
            updateScanProgress(50, status); // Show indeterminate progress
            if (m_fileCountLabel) {
                m_fileCountLabel->setText(tr("Files: %1").arg(filesProcessed));
            }
            LOG_DEBUG(QString("Scan progress: %1 files processed").arg(filesProcessed));
            LOG_FILE("Currently scanning", currentPath);
        });
        
        connect(m_fileScanner, &FileScanner::scanCompleted, this, [this]() {
            int filesFound = m_fileScanner->getTotalFilesFound();
            qint64 bytesScanned = m_fileScanner->getTotalBytesScanned();
            int errorsEncountered = m_fileScanner->getTotalErrorsEncountered();
            
            LOG_INFO("=== FileScanner: Scan Completed ===");
            LOG_INFO(QString("  - Files found: %1").arg(filesFound));
            LOG_INFO(QString("  - Bytes scanned: %1 (%2)").arg(bytesScanned).arg(formatFileSize(bytesScanned)));
            LOG_INFO(QString("  - Errors encountered: %1").arg(errorsEncountered));
            
            QString status = tr("Scan complete! Found %1 files (%2)")
                .arg(filesFound)
                .arg(formatFileSize(bytesScanned));
            updateScanProgress(100, status);
            
            if (m_fileCountLabel) {
                m_fileCountLabel->setText(tr("Files: %1").arg(filesFound));
            }
            
            // Re-enable quick actions
            if (m_quickActions) {
                m_quickActions->setEnabled(true);
                LOG_DEBUG("Quick actions re-enabled");
            }
            
            // Show success message
            showSuccess(tr("Scan Complete"), 
                       tr("Found %1 files totaling %2")
                       .arg(filesFound)
                       .arg(formatFileSize(bytesScanned)));
        });
        
        connect(m_fileScanner, &FileScanner::scanCancelled, this, [this]() {
            LOG_WARNING("=== FileScanner: Scan Cancelled by User ===");
            updateScanProgress(0, tr("Scan cancelled"));
            if (m_quickActions) {
                m_quickActions->setEnabled(true);
            }
        });
        
        connect(m_fileScanner, &FileScanner::scanError, this, [this](FileScanner::ScanError errorType, const QString& path, const QString& description) {
            Q_UNUSED(errorType);
            LOG_WARNING(QString("Scan error: %1 at %2").arg(description).arg(path));
            // Don't show individual errors to avoid spam, they're accumulated
        });
        
        connect(m_fileScanner, &FileScanner::scanErrorSummary, this, [this](int totalErrors, const QList<FileScanner::ScanErrorInfo>& errors) {
            if (totalErrors > 0) {
                LOG_WARNING(QString("=== Scan completed with %1 error(s) ===").arg(totalErrors));
                for (int i = 0; i < qMin(5, errors.size()); ++i) {
                    LOG_WARNING(QString("  Error %1: %2 - %3").arg(i+1).arg(errors[i].filePath).arg(errors[i].errorMessage));
                }
                if (errors.size() > 5) {
                    LOG_WARNING(QString("  ... and %1 more errors").arg(errors.size() - 5));
                }
                // Optionally show a warning to the user
                QString message = tr("Scan completed with %1 error(s). Some files or directories could not be accessed.").arg(totalErrors);
                updateScanProgress(100, message);
            }
        });
    }
    
    // System update timer
    connect(m_systemUpdateTimer, &QTimer::timeout, this, &MainWindow::refreshSystemStats);
}

void MainWindow::updatePlanIndicator()
{
    // This will be updated based on license status
    if (m_planIndicator) {
        m_planIndicator->setText(tr("👤 Free Plan"));
        // Use system colors for better theme compatibility
        m_planIndicator->setStyleSheet("QLabel { color: palette(mid); font-weight: bold; }");
    }
}

void MainWindow::refreshSystemStats()
{
    if (!m_systemOverview)
        return;
        
    SystemOverviewWidget::SystemStats stats;
    
    // Get disk space info for home directory
    QString homePath = QStandardPaths::writableLocation(QStandardPaths::HomeLocation);
    QStorageInfo storage(homePath);
    
    if (storage.isValid()) {
        stats.totalDiskSpace = storage.bytesTotal();
        stats.availableDiskSpace = storage.bytesAvailable();
        stats.usagePercentage = 100.0 - (double(storage.bytesAvailable()) / double(storage.bytesTotal()) * 100.0);
    }
    
    // Placeholder values - will be updated with real data later
    stats.potentialSavings = 0; // Will be calculated from scan results
    stats.filesScanned = 0;     // Will be updated from scan history
    
    m_systemOverview->updateStats(stats);
}

// Private methods
void MainWindow::createHeaderWidget()
{
    m_headerWidget = new QWidget(this);
    m_headerLayout = new QHBoxLayout(m_headerWidget);
    m_headerLayout->setContentsMargins(0, 0, 0, 0);
    m_headerLayout->setSpacing(8);
    
    // Left side buttons
    m_newScanButton = new QPushButton(tr("📁 New Scan"), this);
    m_newScanButton->setFixedSize(120, 32);
    m_newScanButton->setStyleSheet("QPushButton { font-weight: bold; }");
    
    m_settingsButton = new QPushButton(tr("⚙️ Settings"), this);
    m_settingsButton->setFixedSize(120, 32);
    
    m_helpButton = new QPushButton(tr("❓ Help"), this);
    m_helpButton->setFixedSize(120, 32);
    
    // Test button for results window (temporary)
    QPushButton* testResultsButton = new QPushButton(tr("🔍 View Results"), this);
    testResultsButton->setFixedSize(120, 32);
    testResultsButton->setToolTip(tr("Open Results Window (Test)"));
    connect(testResultsButton, &QPushButton::clicked, this, &MainWindow::showScanResults);
    
    // Spacer
    m_headerLayout->addWidget(m_newScanButton);
    m_headerLayout->addWidget(m_settingsButton);
    m_headerLayout->addWidget(m_helpButton);
    m_headerLayout->addWidget(testResultsButton);
    m_headerLayout->addStretch();
    
    // Right side indicators
    m_notificationIndicator = new QLabel(tr("🔔"), this);
    m_notificationIndicator->setFixedSize(24, 24);
    m_notificationIndicator->setAlignment(Qt::AlignCenter);
    
    m_planIndicator = new QLabel(tr("👤 Free Plan"), this);
    // Style will be set in updatePlanIndicator()
    
    m_headerLayout->addWidget(m_notificationIndicator);
    m_headerLayout->addWidget(m_planIndicator);
    
    // Add header to main layout
    m_mainLayout->addWidget(m_headerWidget);
    
    // Add separator line
    QFrame* separator = new QFrame(this);
    separator->setFrameShape(QFrame::HLine);
    separator->setFrameShadow(QFrame::Sunken);
    m_mainLayout->addWidget(separator);
}

void MainWindow::createContentWidgets()
{
    // Create content widgets
    m_quickActions = new QuickActionsWidget(this);
    m_scanHistory = new ScanHistoryWidget(this);
    m_systemOverview = new SystemOverviewWidget(this);
    
    // Add to main layout
    m_mainLayout->addWidget(m_quickActions);
    m_mainLayout->addWidget(m_scanHistory);
    m_mainLayout->addWidget(m_systemOverview);
    
    // Add stretch to push everything up
    m_mainLayout->addStretch();
}

void MainWindow::createStatusBar()
{
    m_statusBar = statusBar();
    
    // Status label
    m_statusLabel = new QLabel(tr("Ready"));
    m_statusBar->addWidget(m_statusLabel);
    
    m_statusBar->addWidget(new QLabel("|"));
    
    // File count
    m_fileCountLabel = new QLabel(tr("Files: 0"));
    m_statusBar->addWidget(m_fileCountLabel);
    
    m_statusBar->addWidget(new QLabel("|"));
    
    // Group count
    m_groupCountLabel = new QLabel(tr("Groups: 0"));
    m_statusBar->addWidget(m_groupCountLabel);
    
    m_statusBar->addWidget(new QLabel("|"));
    
    // Potential savings
    m_savingsLabel = new QLabel(tr("Potential savings: 0 MB"));
    m_statusBar->addWidget(m_savingsLabel);
    
    // Progress bar (initially hidden)
    m_progressBar = new QProgressBar(this);
    m_progressBar->setVisible(false);
    m_progressBar->setMaximumWidth(200);
    m_statusBar->addPermanentWidget(m_progressBar);
    
    updatePlanIndicator();
}

void MainWindow::applyTheme()
{
    // Apply theme-aware styling using system palette colors
    QString styleSheet = QString(R"(
        QGroupBox {
            font-weight: bold;
            border: 1px solid palette(mid);
            border-radius: 5px;
            margin-top: 8px;
            padding-top: 8px;
            color: palette(window-text);
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 8px 0 8px;
            color: palette(window-text);
        }
        
        QPushButton {
            border: 1px solid palette(mid);
            border-radius: 4px;
            padding: 6px 12px;
            background: palette(button);
            color: palette(button-text);
        }
        
        QPushButton:hover {
            background: palette(light);
            border-color: palette(highlight);
        }
        
        QPushButton:pressed {
            background: palette(midlight);
        }
        
        QLabel {
            color: palette(window-text);
        }
        
        QListWidget {
            background: palette(base);
            color: palette(text);
            border: 1px solid palette(mid);
            border-radius: 3px;
        }
        
        QListWidget::item:selected {
            background: palette(highlight);
            color: palette(highlighted-text);
        }
        
        QProgressBar {
            border: 1px solid palette(mid);
            border-radius: 3px;
            text-align: center;
            color: palette(window-text);
        }
        
        QProgressBar::chunk {
            border-radius: 2px;
        }
    )");
    
    setStyleSheet(styleSheet);
}

void MainWindow::loadSettings()
{
    QSettings settings;
    settings.beginGroup("MainWindow");
    
    // Restore window geometry
    QByteArray geometry = settings.value("geometry").toByteArray();
    if (!geometry.isEmpty()) {
        restoreGeometry(geometry);
    }
    
    // Restore window state
    QByteArray state = settings.value("state").toByteArray();
    if (!state.isEmpty()) {
        restoreState(state);
    }
    
    settings.endGroup();
}

void MainWindow::saveSettings()
{
    QSettings settings;
    settings.beginGroup("MainWindow");
    settings.setValue("geometry", saveGeometry());
    settings.setValue("state", saveState());
    settings.endGroup();
}

QString MainWindow::formatFileSize(qint64 bytes) const
{
    if (bytes < 1024) {
        return tr("%1 B").arg(bytes);
    } else if (bytes < 1024 * 1024) {
        return tr("%1 KB").arg(QString::number(static_cast<double>(bytes) / 1024.0, 'f', 1));
    } else if (bytes < 1024 * 1024 * 1024) {
        return tr("%1 MB").arg(QString::number(static_cast<double>(bytes) / (1024.0 * 1024.0), 'f', 1));
    } else {
        return tr("%1 GB").arg(QString::number(static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), 'f', 2));
    }
}

// QuickActionsWidget Implementation
QuickActionsWidget::QuickActionsWidget(QWidget* parent)
    : QGroupBox(tr("Quick Actions"), parent)
    , m_layout(new QGridLayout(this))
    , m_quickScanButton(nullptr)
    , m_downloadsButton(nullptr)
    , m_photoButton(nullptr)
    , m_documentsButton(nullptr)
    , m_fullSystemButton(nullptr)
    , m_customButton(nullptr)
{
    createButtons();
    setupButtonStyles();
}

void QuickActionsWidget::setEnabled(bool enabled)
{
    QGroupBox::setEnabled(enabled);
    
    // Update visual state of all buttons
    const QList<QPushButton*> buttons = {
        m_quickScanButton, m_downloadsButton, m_photoButton,
        m_documentsButton, m_fullSystemButton, m_customButton
    };
    
    for (QPushButton* button : buttons) {
        if (button) {
            button->setEnabled(enabled);
        }
    }
}

void QuickActionsWidget::createButtons()
{
    m_layout->setContentsMargins(12, 20, 12, 12);
    m_layout->setSpacing(8);
    
    // Create buttons with icons and text
    m_quickScanButton = new QPushButton(tr("🚀 Start Quick Scan"), this);
    m_quickScanButton->setFixedSize(180, 60);
    
    m_downloadsButton = new QPushButton(tr("📂 Downloads Cleanup"), this);
    m_downloadsButton->setFixedSize(180, 60);
    
    m_photoButton = new QPushButton(tr("📸 Photo Cleanup"), this);
    m_photoButton->setFixedSize(180, 60);
    
    m_documentsButton = new QPushButton(tr("📄 Documents"), this);
    m_documentsButton->setFixedSize(180, 60);
    
    m_fullSystemButton = new QPushButton(tr("🖥️ Full System Scan"), this);
    m_fullSystemButton->setFixedSize(180, 60);
    
    m_customButton = new QPushButton(tr("⭐ Custom Preset"), this);
    m_customButton->setFixedSize(180, 60);
    
    // Arrange in 2x3 grid
    m_layout->addWidget(m_quickScanButton, 0, 0);
    m_layout->addWidget(m_downloadsButton, 0, 1);
    m_layout->addWidget(m_photoButton, 0, 2);
    m_layout->addWidget(m_documentsButton, 1, 0);
    m_layout->addWidget(m_fullSystemButton, 1, 1);
    m_layout->addWidget(m_customButton, 1, 2);
    
    // Connect signals
    connect(m_quickScanButton, &QPushButton::clicked, this, &QuickActionsWidget::onQuickScanClicked);
    connect(m_downloadsButton, &QPushButton::clicked, this, &QuickActionsWidget::onDownloadsCleanupClicked);
    connect(m_photoButton, &QPushButton::clicked, this, &QuickActionsWidget::onPhotoCleanupClicked);
    connect(m_documentsButton, &QPushButton::clicked, this, &QuickActionsWidget::onDocumentsClicked);
    connect(m_fullSystemButton, &QPushButton::clicked, this, &QuickActionsWidget::onFullSystemClicked);
    connect(m_customButton, &QPushButton::clicked, this, &QuickActionsWidget::onCustomPresetClicked);
}

void QuickActionsWidget::setupButtonStyles()
{
    // Use theme-aware styling for better compatibility
    QString buttonStyle = R"(
        QPushButton {
            font-weight: bold;
            text-align: center;
            border: 2px solid palette(mid);
            border-radius: 8px;
            background: palette(button);
            color: palette(button-text);
            padding: 8px;
        }
        
        QPushButton:hover {
            background: palette(light);
            border-color: palette(highlight);
        }
        
        QPushButton:pressed {
            background: palette(midlight);
            border-color: palette(dark);
        }
        
        QPushButton:focus {
            border-color: palette(highlight);
        }
    )";
    
    const QList<QPushButton*> buttons = {
        m_quickScanButton, m_downloadsButton, m_photoButton,
        m_documentsButton, m_fullSystemButton, m_customButton
    };
    
    for (QPushButton* button : buttons) {
        if (button) {
            button->setStyleSheet(buttonStyle);
        }
    }
}

// Preset button slots
void QuickActionsWidget::onQuickScanClicked() { emit presetSelected("quick"); }
void QuickActionsWidget::onDownloadsCleanupClicked() { emit presetSelected("downloads"); }
void QuickActionsWidget::onPhotoCleanupClicked() { emit presetSelected("photos"); }
void QuickActionsWidget::onDocumentsClicked() { emit presetSelected("documents"); }
void QuickActionsWidget::onFullSystemClicked() { emit presetSelected("fullsystem"); }
void QuickActionsWidget::onCustomPresetClicked() { emit presetSelected("custom"); }

// Main Window scan configuration handler
void MainWindow::handleScanConfiguration()
{
    LOG_INFO("=== Starting New Scan ===");
    
    // Get the configuration from the dialog
    if (!m_scanSetupDialog) {
        LOG_ERROR("Scan dialog not initialized");
        return;
    }
    
    ScanSetupDialog::ScanConfiguration config = m_scanSetupDialog->getCurrentConfiguration();
    
    LOG_INFO(QString("Scan Configuration:"));
    LOG_INFO(QString("  - Target paths (%1): %2").arg(config.targetPaths.size()).arg(config.targetPaths.join(", ")));
    LOG_INFO(QString("  - Excluded folders: %1").arg(config.excludeFolders.join(", ")));
    LOG_INFO(QString("  - Detection mode: %1").arg(static_cast<int>(config.detectionMode)));
    LOG_INFO(QString("  - Minimum file size: %1 MB").arg(config.minimumFileSize));
    LOG_INFO(QString("  - Include hidden: %1").arg(config.includeHidden ? "Yes" : "No"));
    LOG_INFO(QString("  - Follow symlinks: %1").arg(config.followSymlinks ? "Yes" : "No"));
    
    // Pass configuration to the FileScanner
    if (m_fileScanner) {
        // Convert ScanSetupDialog configuration to FileScanner::ScanOptions
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths = config.targetPaths;
        scanOptions.minimumFileSize = config.minimumFileSize * 1024 * 1024; // Convert MB to bytes
        scanOptions.includeHiddenFiles = config.includeHidden;
        scanOptions.scanSystemDirectories = config.includeSystem;
        scanOptions.followSymlinks = config.followSymlinks;
        
        // Parse exclude patterns
        if (!config.excludePatterns.isEmpty()) {
            scanOptions.excludePatterns = config.excludePatterns;
            LOG_DEBUG(QString("  - Exclude patterns: %1").arg(scanOptions.excludePatterns.join(", ")));
        }
        
        LOG_INFO(QString("Initiating FileScanner with %1 target paths").arg(scanOptions.targetPaths.size()));
        LOG_DEBUG(QString("  - Min file size (bytes): %1").arg(scanOptions.minimumFileSize));
        
        // Start the scan
        m_fileScanner->startScan(scanOptions);
        
        // Update UI to show scanning state
        updateScanProgress(0, tr("Scanning..."));
        LOG_INFO("Scan initiated successfully");
    } else {
        LOG_ERROR("FileScanner not initialized!");
        showError(tr("Error"), tr("File scanner is not initialized. Please restart the application."));
    }
    
    // Update UI to show scanning state
    if (m_quickActions) {
        m_quickActions->setEnabled(false);
        LOG_DEBUG("Quick actions disabled during scan");
    }
}

