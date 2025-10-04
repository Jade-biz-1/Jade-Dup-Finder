#include "main_window.h"
#include "scan_dialog.h"
#include "results_window.h"
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
    if (!m_scanSetupDialog) {
        m_scanSetupDialog = new ScanSetupDialog(this);
        
        // Connect scan configuration signal
        connect(m_scanSetupDialog, &ScanSetupDialog::scanConfigured,
                this, &MainWindow::handleScanConfiguration);
        
        // Connect preset saved signal
        connect(m_scanSetupDialog, &ScanSetupDialog::presetSaved,
                this, [this](const QString& name) {
                    qDebug() << "Preset saved:" << name;
                    // TODO: Update quick actions with new preset
                });
    }
    
    m_scanSetupDialog->show();
    m_scanSetupDialog->raise();
    m_scanSetupDialog->activateWindow();
}

void MainWindow::onPresetSelected(const QString& preset)
{
    qDebug() << "Preset selected:" << preset;
    emit scanRequested(preset);
}

void MainWindow::onSettingsRequested()
{
    emit settingsRequested();
}

void MainWindow::onHelpRequested()
{
    emit helpRequested();
}

void MainWindow::updateSystemInfo()
{
    refreshSystemStats();
}

void MainWindow::onScanHistoryItemClicked(int index)
{
    qDebug() << "History item clicked:" << index;
}

void MainWindow::onViewAllHistoryClicked()
{
    qDebug() << "View all history clicked";
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
    // Get the configuration from the dialog
    if (!m_scanSetupDialog) {
        return;
    }
    
    ScanSetupDialog::ScanConfiguration config = m_scanSetupDialog->getCurrentConfiguration();
    
    qDebug() << "Starting scan with configuration:";
    qDebug() << "Target paths:" << config.targetPaths;
    qDebug() << "Excluded folders:" << config.excludeFolders;
    qDebug() << "Detection mode:" << static_cast<int>(config.detectionMode);
    qDebug() << "Minimum file size:" << config.minimumFileSize;
    
    // TODO: Pass configuration to the core scanning engine
    if (m_fileScanner) {
        // This would be the actual implementation:
        // m_fileScanner->startScan(config);
        
        // For now, just show a message
        updateScanProgress(0, tr("Scan configured. Ready to start..."));
        
        // Simulate starting the scan
        QTimer::singleShot(1000, [this]() {
            updateScanProgress(10, tr("Initializing scan..."));
        });
    }
    
    // Update UI to show scanning state
    if (m_quickActions) {
        m_quickActions->setEnabled(false);
    }
}

