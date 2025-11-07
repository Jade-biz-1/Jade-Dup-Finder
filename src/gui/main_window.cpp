// Project headers
#include "main_window.h"
#include "scan_dialog.h"
#include "results_window.h"
#include "settings_dialog.h"
#include "scan_history_dialog.h"
#include "restore_dialog.h"
#include "safety_features_dialog.h"
#include "about_dialog.h"
#include "file_scanner.h"
#include "theme_manager.h"
#include "app_config.h"
#include "scan_history_manager.h"
#include "../core/window_state_manager.h"
#include "logger.h"

// Qt headers
#include <QtWidgets/QApplication>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QListWidgetItem>
#include <QtCore/QSettings>
#include <QtCore/QStandardPaths>
#include <QtCore/QStorageInfo>
#include <QtCore/QDir>
#include <QtCore/QUuid>
#include <QtConcurrent/QtConcurrent>
#include <QtGui/QIcon>
#include <QtGui/QCloseEvent>
#include <QtGui/QPalette>
#include <QtGui/QShortcut>
#include <QtGui/QKeySequence>

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
    , m_detectorThread(nullptr)
    , m_hashCalculator(nullptr)
    , m_safetyManager(nullptr)
    , m_fileManager(nullptr)
    , m_scanSetupDialog(nullptr)
    , m_resultsWindow(nullptr)
    , m_settingsDialog(nullptr)
    , m_scanProgressDialog(nullptr)
    , m_scanErrorDialog(nullptr)
    , m_safetyFeaturesDialog(nullptr)  // T17
    , m_aboutDialog(nullptr)  // Section 1.5.2
    , m_systemUpdateTimer(new QTimer(this))
{
    setWindowTitle(tr("DupFinder - Duplicate File Finder"));
    setMinimumSize(800, 600);
    resize(1024, 768);
    
    // Set application icon
    setWindowIcon(QIcon(":/icons/app-icon.png"));
    
    initializeUI();
    setupConnections();
    setupKeyboardShortcuts();
    loadSettings();
    
    // Register with WindowStateManager for automatic state saving/restoring
    WindowStateManager::instance()->registerWindow(this, "MainWindow");
    
    // Start system stats timer
    m_systemUpdateTimer->setInterval(30000); // Update every 30 seconds
    m_systemUpdateTimer->start();
    
    // Initial system stats update
    QTimer::singleShot(1000, this, &MainWindow::refreshSystemStats);
    
    // DISABLED: Theme compliance validation produces too many warnings
    // These are style recommendations, not errors, and don't affect functionality
    // Uncomment below if you want to see detailed theme compliance reports:
    /*
    QTimer::singleShot(2000, []() {
        // Run validation in background thread (QFuture intentionally not stored)
        [[maybe_unused]] auto future = QtConcurrent::run([]() {
            ThemeManager::instance()->performThemeComplianceTest();
        });
    });
    */
}

MainWindow::~MainWindow()
{
    // Clean up detector thread if it exists
    if (m_detectorThread) {
        // Stop the thread gracefully
        m_detectorThread->quit();
        m_detectorThread->wait();
        delete m_detectorThread;
        m_detectorThread = nullptr;
    }

    saveSettings();
}

// Core component integration
void MainWindow::setFileScanner(FileScanner* scanner)
{
    // Avoid duplicate setup if the same scanner is set
    if (m_fileScanner == scanner) {
        return;
    }

    // Disconnect from previous scanner if any
    if (m_fileScanner) {
        disconnect(m_fileScanner, nullptr, this, nullptr);
    }

    m_fileScanner = scanner;
    
    // Set up connections now that we have the scanner
    if (m_fileScanner) {
        LOG_DEBUG(LogCategories::UI, "Setting up FileScanner connections");
        
        connect(m_fileScanner, &FileScanner::scanStarted, this, [this]() {
            LOG_INFO(LogCategories::UI, "=== FileScanner: Scan Started ===");
            updateScanProgress(0, tr("Scanning..."));
            if (m_progressBar) m_progressBar->setVisible(true);
            
            // Show scan progress dialog
            if (!m_scanProgressDialog) {
                m_scanProgressDialog = new ScanProgressDialog(this);
                
                // Register with WindowStateManager
                WindowStateManager::instance()->registerDialog(m_scanProgressDialog, "ScanProgressDialog");
                
                // Register with ThemeManager for automatic theme updates
                ThemeManager::instance()->registerDialog(m_scanProgressDialog);
                
                // Connect pause/resume buttons to FileScanner
                connect(m_scanProgressDialog, &ScanProgressDialog::pauseRequested, 
                        m_fileScanner, &FileScanner::pauseScan, Qt::UniqueConnection);
                connect(m_scanProgressDialog, &ScanProgressDialog::resumeRequested, 
                        m_fileScanner, &FileScanner::resumeScan, Qt::UniqueConnection);
                connect(m_scanProgressDialog, &ScanProgressDialog::cancelRequested, 
                        m_fileScanner, &FileScanner::cancelScan, Qt::UniqueConnection);
                
                // Connect View Errors button (Task 10)
                connect(m_scanProgressDialog, &ScanProgressDialog::viewErrorsRequested,
                        this, [this]() {
                            if (!m_scanErrorDialog) {
                                m_scanErrorDialog = new ScanErrorDialog(this);
                                
                                // Register with WindowStateManager
                                WindowStateManager::instance()->registerDialog(m_scanErrorDialog, "ScanErrorDialog");
                                
                                // Register with ThemeManager for automatic theme updates
                                ThemeManager::instance()->registerDialog(m_scanErrorDialog);
                            }
                            
                            // Get current errors from FileScanner
                            if (m_fileScanner) {
                                QList<FileScanner::ScanErrorInfo> errors = m_fileScanner->getScanErrors();
                                m_scanErrorDialog->setErrors(errors);
                            }
                            
                            m_scanErrorDialog->show();
                            m_scanErrorDialog->raise();
                            m_scanErrorDialog->activateWindow();
                        });
                
                // Connect FileScanner pause/resume signals to dialog
                connect(m_fileScanner, &FileScanner::scanPaused, 
                        this, [this]() {
                            if (m_scanProgressDialog) {
                                m_scanProgressDialog->setPaused(true);
                            }
                        });
                connect(m_fileScanner, &FileScanner::scanResumed, 
                        this, [this]() {
                            if (m_scanProgressDialog) {
                                m_scanProgressDialog->setPaused(false);
                            }
                        });
            }
            
            // Ensure the dialog is properly shown and brought to front
            m_scanProgressDialog->show();
            m_scanProgressDialog->raise();
            m_scanProgressDialog->activateWindow();
            
            // Force the dialog to be visible and on top
            m_scanProgressDialog->setWindowState(m_scanProgressDialog->windowState() & ~Qt::WindowMinimized);
            m_scanProgressDialog->setWindowState(m_scanProgressDialog->windowState() | Qt::WindowActive);
            
            LOG_DEBUG(LogCategories::UI, "Scan progress dialog shown and activated");
        });
        
        connect(m_fileScanner, &FileScanner::scanProgress, this, [this](int filesProcessed, int totalFiles, const QString& currentPath) {
            Q_UNUSED(totalFiles);
            QString status = tr("Scanning... %1 files found").arg(filesProcessed);
            updateScanProgress(50, status); // Show indeterminate progress
            if (m_fileCountLabel) {
                m_fileCountLabel->setText(tr("Files: %1").arg(filesProcessed));
            }
            LOG_DEBUG(LogCategories::UI, QString("Scan progress: %1 files processed").arg(filesProcessed));
            LOG_DEBUG(LogCategories::SCAN, QString("File: %1 - %2").arg(currentPath).arg("Currently scanning"));
        });
        
        bool connected = connect(m_fileScanner, &FileScanner::scanCompleted, this, &MainWindow::onScanCompleted, Qt::UniqueConnection);
        LOG_DEBUG(LogCategories::UI, QString("FileScanner::scanCompleted connection result: %1").arg(connected ? "success" : "failed"));
        
        // Connect detailed progress to scan progress dialog
        connect(m_fileScanner, &FileScanner::detailedProgress, this, [this](const FileScanner::ScanProgress& progress) {
            if (m_scanProgressDialog) {
                ScanProgressDialog::ProgressInfo info;
                // PERFORMANCE FIX: Set operation type and status to show "Running" instead of "Initializing"
                info.operationType = tr("File Scan");
                info.status = ScanProgressDialog::OperationStatus::Running;
                info.filesScanned = progress.filesScanned;
                info.bytesScanned = progress.bytesScanned;
                info.currentFolder = progress.currentFolder;
                info.currentFile = progress.currentFile;
                info.filesPerSecond = progress.filesPerSecond;
                info.isPaused = progress.isPaused;
                info.errorsEncountered = m_fileScanner->getTotalErrorsEncountered(); // Task 10

                m_scanProgressDialog->updateProgress(info);
            }
        });
        
        LOG_DEBUG(LogCategories::UI, "FileScanner connections complete");
    }
}

void MainWindow::setHashCalculator(HashCalculator* calculator)
{
    m_hashCalculator = calculator;
}

void MainWindow::setDuplicateDetector(DuplicateDetector* detector)
{
    m_duplicateDetector = detector;

    // Set up connections now that we have the detector
    if (m_duplicateDetector) {
        LOG_DEBUG(LogCategories::UI, "Setting up DuplicateDetector on background thread");

        // Create a background thread for the detector
        m_detectorThread = new QThread(this);
        LOG_DEBUG(LogCategories::UI, QString("Created detector thread: %1").arg(reinterpret_cast<quintptr>(m_detectorThread)));

        // Move detector to the background thread
        m_duplicateDetector->moveToThread(m_detectorThread);
        LOG_DEBUG(LogCategories::UI, "DuplicateDetector moved to background thread");

        // Connect signals with Qt::QueuedConnection for thread-safe communication
        // Qt::QueuedConnection ensures signals are processed in the receiver's thread
        connect(m_duplicateDetector, &DuplicateDetector::detectionStarted,
                this, &MainWindow::onDuplicateDetectionStarted, Qt::QueuedConnection);

        connect(m_duplicateDetector, &DuplicateDetector::detectionProgress,
                this, &MainWindow::onDuplicateDetectionProgress, Qt::QueuedConnection);

        connect(m_duplicateDetector, &DuplicateDetector::detectionCompleted,
                this, &MainWindow::onDuplicateDetectionCompleted, Qt::QueuedConnection);

        connect(m_duplicateDetector, &DuplicateDetector::detectionError,
                this, &MainWindow::onDuplicateDetectionError, Qt::QueuedConnection);

        // Clean up thread when detector is done or on application exit
        connect(m_detectorThread, &QThread::finished, m_detectorThread, &QObject::deleteLater);

        // Start the thread
        m_detectorThread->start();
        LOG_DEBUG(LogCategories::UI, "Detector thread started");
        LOG_DEBUG(LogCategories::UI, "DuplicateDetector connections complete");
    }
}

void MainWindow::setSafetyManager(SafetyManager* manager)
{
    m_safetyManager = manager;
}

void MainWindow::setFileManager(FileManager* manager)
{
    m_fileManager = manager;
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
    LOG_DEBUG(LogCategories::UI, "showScanResults called - opening results window");
    
    if (!m_resultsWindow) {
        m_resultsWindow = new ResultsWindow(this);
        
        // Register with WindowStateManager
        WindowStateManager::instance()->registerWindow(m_resultsWindow, "ResultsWindow");
        
        // Set FileManager reference
        if (m_fileManager) {
            m_resultsWindow->setFileManager(m_fileManager);
        }
        
        // Connect results window signals
        connect(m_resultsWindow, &ResultsWindow::windowClosed,
                this, [this]() {
                    LOG_DEBUG(LogCategories::UI, "Results window closed");
                    // Results window will be deleted when parent (this) is deleted
                });
        
        connect(m_resultsWindow, &ResultsWindow::resultsUpdated,
                this, [this](const ResultsWindow::ScanResults& results) {
                    LOG_DEBUG(LogCategories::UI, QString("Results updated: %1 groups").arg(results.duplicateGroups.size()));
                    
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
    LOG_INFO(LogCategories::UI, "User clicked 'New Scan' button");
    
    if (!m_scanSetupDialog) {
        LOG_DEBUG(LogCategories::UI, "Creating new ScanSetupDialog");
        m_scanSetupDialog = new ScanSetupDialog(this);
        
        // Register with WindowStateManager
        WindowStateManager::instance()->registerDialog(m_scanSetupDialog, "ScanSetupDialog");
        
        // Connect scan configuration signal
        connect(m_scanSetupDialog, &ScanSetupDialog::scanConfigured,
                this, &MainWindow::handleScanConfiguration);
        
        // Connect preset saved signal
        connect(m_scanSetupDialog, &ScanSetupDialog::presetSaved,
                this, [this](const QString& name) {
                    LOG_INFO(LogCategories::UI, QString("Preset saved: %1").arg(name));
                    // Preset is automatically saved by the dialog
                });
    }
    
    LOG_DEBUG(LogCategories::UI, "Showing scan setup dialog");
    m_scanSetupDialog->show();
    m_scanSetupDialog->raise();
    m_scanSetupDialog->activateWindow();
}

void MainWindow::onPresetSelected(const QString& preset)
{
    LOG_DEBUG(LogCategories::UI, QString("onPresetSelected called with preset: %1").arg(preset));
    LOG_INFO(LogCategories::UI, QString("User selected preset: %1").arg(preset));
    
    // Create scan dialog if needed
    if (!m_scanSetupDialog) {
        LOG_DEBUG(LogCategories::UI, "Creating new ScanSetupDialog for preset");
        m_scanSetupDialog = new ScanSetupDialog(this);
        
        // Register with WindowStateManager
        WindowStateManager::instance()->registerDialog(m_scanSetupDialog, "ScanSetupDialog");
        
        // Connect scan configuration signal
        connect(m_scanSetupDialog, &ScanSetupDialog::scanConfigured,
                this, &MainWindow::handleScanConfiguration);
        
        // Connect preset saved signal
        connect(m_scanSetupDialog, &ScanSetupDialog::presetSaved,
                this, [this](const QString& name) {
                    LOG_INFO(LogCategories::UI, QString("Preset saved: %1").arg(name));
                });
        
        // Re-enable quick actions when dialog closes (if not scanning)
        connect(m_scanSetupDialog, &QDialog::finished, this, [this](int result) {
            Q_UNUSED(result);
            // Only re-enable if we're not currently scanning
            if (m_quickActions && m_fileScanner && !m_fileScanner->isScanning()) {
                LOG_DEBUG(LogCategories::UI, "Dialog closed, re-enabling quick actions");
                m_quickActions->setEnabled(true);
            }
        });
    }
    
    // Load the preset configuration
    m_scanSetupDialog->loadPreset(preset);
    
    // Show the dialog
    LOG_DEBUG(LogCategories::UI, "Showing scan setup dialog with preset");
    m_scanSetupDialog->show();
    m_scanSetupDialog->raise();
    m_scanSetupDialog->activateWindow();
}

void MainWindow::onSettingsRequested()
{
    LOG_INFO(LogCategories::UI, "User clicked 'Settings' button");
    
    if (!m_settingsDialog) {
        m_settingsDialog = new SettingsDialog(this);
        
        // Register with WindowStateManager
        WindowStateManager::instance()->registerDialog(m_settingsDialog, "SettingsDialog");
        
        connect(m_settingsDialog, &SettingsDialog::settingsChanged,
                this, [this]() {
                    LOG_INFO(LogCategories::UI, "Settings changed, reloading configuration");
                    // Reload settings in application
                    loadSettings();
                });
    }
    
    m_settingsDialog->show();
    m_settingsDialog->raise();
    m_settingsDialog->activateWindow();
}

// T17: Safety Features UI
void MainWindow::onSafetyFeaturesRequested()
{
    LOG_INFO(LogCategories::UI, "User requested Safety Features dialog");
    
    if (!m_safetyFeaturesDialog) {
        m_safetyFeaturesDialog = new SafetyFeaturesDialog(m_safetyManager, this);
        
        // Register with WindowStateManager
        WindowStateManager::instance()->registerDialog(m_safetyFeaturesDialog, "SafetyFeaturesDialog");
        
        // Connect signals
        connect(m_safetyFeaturesDialog, &SafetyFeaturesDialog::protectionRulesChanged,
                this, [this]() {
                    LOG_INFO(LogCategories::UI, "Protection rules changed");
                    statusBar()->showMessage(tr("Protection rules updated"), 3000);
                });
        
        connect(m_safetyFeaturesDialog, &SafetyFeaturesDialog::safetySettingsChanged,
                this, [this]() {
                    LOG_INFO(LogCategories::UI, "Safety settings changed");
                    statusBar()->showMessage(tr("Safety settings updated"), 3000);
                });
    }
    
    m_safetyFeaturesDialog->show();
    m_safetyFeaturesDialog->raise();
    m_safetyFeaturesDialog->activateWindow();
}

void MainWindow::onHelpRequested()
{
    LOG_INFO(LogCategories::UI, "User clicked 'Help' button");
    
    QString helpText = tr(
        "<h2>DupFinder - Duplicate File Finder</h2>"
        "<p><b>Quick Start:</b></p>"
        "<ol>"
        "<li>Click 'New Scan' to configure a scan</li>"
        "<li>Select folders to scan</li>"
        "<li>Configure scan options (file size, types, etc.)</li>"
        "<li>Click 'Start Scan' to begin</li>"
        "<li>Review duplicate groups in results</li>"
        "<li>Select files to delete or move</li>"
        "</ol>"
        "<p><b>Quick Actions:</b></p>"
        "<ul>"
        "<li><b>Quick Scan:</b> Scan common locations (Home, Downloads, Documents)</li>"
        "<li><b>Downloads Cleanup:</b> Find duplicates in Downloads folder</li>"
        "<li><b>Photo Cleanup:</b> Find duplicate photos in Pictures folder</li>"
        "<li><b>Documents:</b> Scan document folders</li>"
        "<li><b>Full System:</b> Comprehensive system scan</li>"
        "</ul>"
        "<p><b>Keyboard Shortcuts:</b></p>"
        "<ul>"
        "<li><b>Ctrl+N:</b> New Scan</li>"
        "<li><b>Ctrl+O:</b> View Scan History</li>"
        "<li><b>Ctrl+S:</b> Export Results (when results window is open)</li>"
        "<li><b>Ctrl+Z:</b> Undo/Restore Files</li>"
        "<li><b>Ctrl+,:</b> Settings</li>"
        "<li><b>Ctrl+Shift+S:</b> Safety Features</li>"
        "<li><b>Ctrl+Shift+A:</b> About DupFinder</li>"
        "<li><b>Ctrl+Q:</b> Quit Application</li>"
        "<li><b>F1:</b> Help</li>"
        "<li><b>F5 / Ctrl+R:</b> Refresh System Stats</li>"
        "<li><b>Escape:</b> Cancel Operation/Close Dialog</li>"
        "<li><b>Ctrl+1:</b> Quick Scan</li>"
        "<li><b>Ctrl+2:</b> Downloads Cleanup</li>"
        "<li><b>Ctrl+3:</b> Photo Cleanup</li>"
        "<li><b>Ctrl+4:</b> Documents Scan</li>"
        "<li><b>Ctrl+5:</b> Full System Scan</li>"
        "<li><b>Ctrl+6:</b> Custom Scan</li>"
        "</ul>"
        "<p><b>Safety Features:</b></p>"
        "<ul>"
        "<li>Automatic backups before deletion</li>"
        "<li>Protected system files</li>"
        "<li>Undo functionality</li>"
        "</ul>"
        "<p>For more information and full version details, click 'About' button below.</p>"
        "<p>Visit: <a href='https://dupfinder.org/docs'>dupfinder.org/docs</a></p>"
    );
    
    QMessageBox msgBox(this);
    msgBox.setWindowTitle(tr("DupFinder Help"));
    msgBox.setText(helpText);
    msgBox.setTextFormat(Qt::RichText);
    msgBox.setIcon(QMessageBox::Information);
    
    QPushButton* aboutButton = msgBox.addButton(tr("About"), QMessageBox::ActionRole);
    QPushButton* okButton = msgBox.addButton(QMessageBox::Ok);
    msgBox.setDefaultButton(okButton);
    
    msgBox.exec();
    
    if (msgBox.clickedButton() == aboutButton) {
        onAboutRequested();
    }
}

void MainWindow::onAboutRequested()
{
    LOG_INFO(LogCategories::UI, "User requested About dialog");
    
    if (!m_aboutDialog) {
        m_aboutDialog = new AboutDialog(this);
        
        // Register with WindowStateManager
        WindowStateManager::instance()->registerDialog(m_aboutDialog, "AboutDialog");
    }
    
    m_aboutDialog->show();
    m_aboutDialog->raise();
    m_aboutDialog->activateWindow();
}

void MainWindow::onRestoreRequested()
{
    LOG_INFO(LogCategories::UI, "User clicked 'Restore' button");
    
    if (!m_safetyManager) {
        QMessageBox::warning(this, tr("Restore Files"),
                           tr("Safety manager not initialized. Cannot access backups."));
        return;
    }
    
    // Create and show restore dialog
    RestoreDialog* restoreDialog = new RestoreDialog(m_safetyManager, this);
    
    // Register with WindowStateManager
    WindowStateManager::instance()->registerDialog(restoreDialog, "RestoreDialog");
    
    // Register with ThemeManager for automatic theme updates
    ThemeManager::instance()->registerDialog(restoreDialog);
    
    // Connect filesRestored signal
    connect(restoreDialog, &RestoreDialog::filesRestored,
            this, [this](const QStringList& backupPaths) {
                LOG_INFO(LogCategories::FILE_OPS, QString("Restoring %1 files from backups").arg(backupPaths.size()));
                
                int successCount = 0;
                int failCount = 0;
                QStringList failedFiles;
                
                for (const QString& backupPath : backupPaths) {
                    // Use SafetyManager to restore from backup
                    bool success = m_safetyManager->restoreFromBackup(backupPath);
                    if (success) {
                        successCount++;
                        LOG_INFO(LogCategories::FILE_OPS, QString("Restored: %1").arg(backupPath));
                    } else {
                        failCount++;
                        failedFiles.append(backupPath);
                        LOG_ERROR(LogCategories::FILE_OPS, QString("Failed to restore: %1").arg(backupPath));
                    }
                }
                
                // Show result message
                if (failCount == 0) {
                    QMessageBox::information(this, tr("Restore Complete"),
                                           tr("Successfully restored %1 file(s).").arg(successCount));
                } else {
                    QString message = tr("Restored %1 file(s) successfully.\n%2 file(s) failed to restore.")
                                        .arg(successCount).arg(failCount);
                    if (!failedFiles.isEmpty()) {
                        message += tr("\n\nFailed files:\n%1").arg(failedFiles.join("\n"));
                    }
                    QMessageBox::warning(this, tr("Restore Completed with Errors"), message);
                }
            });
    
    restoreDialog->setAttribute(Qt::WA_DeleteOnClose);
    restoreDialog->show();
}

void MainWindow::updateSystemInfo()
{
    LOG_DEBUG(LogCategories::UI, "Updating system information");
    refreshSystemStats();
}

void MainWindow::onScanHistoryItemClicked(int index)
{
    LOG_INFO(LogCategories::UI, QString("User clicked history item: %1").arg(index));
    
    // Get the history item and load its results
    if (m_scanHistory && index >= 0) {
        QList<ScanHistoryWidget::ScanHistoryItem> history = m_scanHistory->getHistory();
        if (index < history.size()) {
            const auto& item = history[index];
            LOG_INFO(LogCategories::UI, QString("Loading scan results for: %1").arg(item.scanId));
            
            // Load scan from history manager
            ScanHistoryManager::ScanRecord record = 
                ScanHistoryManager::instance()->loadScan(item.scanId);
            
            if (record.isValid()) {
                LOG_INFO(LogCategories::UI, QString("Loaded scan with %1 groups").arg(record.groups.size()));
                
                // Create results window if needed
                if (!m_resultsWindow) {
                    m_resultsWindow = new ResultsWindow(this);
                    
                    // Register with WindowStateManager
                    WindowStateManager::instance()->registerWindow(m_resultsWindow, "ResultsWindow");
                    
                    // Set FileManager reference
                    if (m_fileManager) {
                        m_resultsWindow->setFileManager(m_fileManager);
                    }
                }
                
                // Display the loaded results
                m_resultsWindow->displayDuplicateGroups(record.groups);
                m_resultsWindow->show();
                m_resultsWindow->raise();
                m_resultsWindow->activateWindow();
                
                // Update stats
                if (m_fileCountLabel) {
                    m_fileCountLabel->setText(tr("Files: %1").arg(record.filesScanned));
                }
                if (m_groupCountLabel) {
                    m_groupCountLabel->setText(tr("Groups: %1").arg(record.duplicateGroups));
                }
                if (m_savingsLabel) {
                    m_savingsLabel->setText(tr("Savings: %1").arg(formatFileSize(record.potentialSavings)));
                }
            } else {
                LOG_ERROR(LogCategories::UI, "Failed to load scan from history");
                QMessageBox::warning(this, tr("Load Error"),
                    tr("Could not load scan results. The scan may have been deleted."));
            }
        }
    }
}

void MainWindow::onViewAllHistoryClicked()
{
    LOG_INFO(LogCategories::UI, "User clicked 'View All History'");
    
    // Create and show scan history dialog
    ScanHistoryDialog* historyDialog = new ScanHistoryDialog(this);
    
    // Register with WindowStateManager
    WindowStateManager::instance()->registerDialog(historyDialog, "ScanHistoryDialog");
    
    // Register with ThemeManager for automatic theme updates
    ThemeManager::instance()->registerDialog(historyDialog);
    
    // Connect signals
    connect(historyDialog, &ScanHistoryDialog::scanSelected,
            this, [this, historyDialog](const QString& scanId) {
                LOG_INFO(LogCategories::UI, QString("Loading scan from history: %1").arg(scanId));
                
                // Load scan from history manager
                ScanHistoryManager::ScanRecord record = ScanHistoryManager::instance()->loadScan(scanId);
                
                if (record.isValid()) {
                    // Show results in results window
                    if (!m_resultsWindow) {
                        m_resultsWindow = new ResultsWindow(this);
                        
                        // Register with WindowStateManager
                        WindowStateManager::instance()->registerWindow(m_resultsWindow, "ResultsWindow");
                        
                        if (m_fileManager) {
                            m_resultsWindow->setFileManager(m_fileManager);
                        }
                        
                        // Connect results window close event to show scan history dialog again
                        connect(m_resultsWindow, &ResultsWindow::windowClosed,
                                this, [historyDialog]() {
                                    LOG_INFO(LogCategories::UI, "Results window closed, showing scan history dialog again");
                                    if (historyDialog && !historyDialog->isVisible()) {
                                        historyDialog->show();
                                        historyDialog->raise();
                                        historyDialog->activateWindow();
                                    }
                                });
                    }
                    
                    m_resultsWindow->displayDuplicateGroups(record.groups);
                    m_resultsWindow->show();
                    m_resultsWindow->raise();
                    m_resultsWindow->activateWindow();
                } else {
                    QMessageBox::warning(this, tr("Load Error"),
                                       tr("Could not load scan results from history."));
                }
            });
    
    connect(historyDialog, &ScanHistoryDialog::scanDeleted,
            this, [this](const QString& scanId) {
                LOG_INFO(LogCategories::UI, QString("Scan deleted from history: %1").arg(scanId));
                // Refresh the history widget
                if (m_scanHistory) {
                    m_scanHistory->refreshHistory();
                }
            });
    
    historyDialog->setAttribute(Qt::WA_DeleteOnClose);
    historyDialog->show();
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
        LOG_DEBUG(LogCategories::UI, "Connecting QuickActionsWidget signals");
        bool connected = connect(m_quickActions, &QuickActionsWidget::presetSelected, this, &MainWindow::onPresetSelected);
        LOG_DEBUG(LogCategories::UI, QString("QuickActionsWidget connection result: %1").arg(connected ? "success" : "failed"));
    } else {
        LOG_WARNING(LogCategories::UI, "m_quickActions is NULL!");
    }
    
    // Scan history connections
    if (m_scanHistory) {
        connect(m_scanHistory, &ScanHistoryWidget::historyItemClicked, this, &MainWindow::onScanHistoryItemClicked);
        connect(m_scanHistory, &ScanHistoryWidget::viewAllRequested, this, &MainWindow::onViewAllHistoryClicked);
    }
    
    // FileScanner connections are set up in setFileScanner() method
    
    // System update timer
    connect(m_systemUpdateTimer, &QTimer::timeout, this, &MainWindow::refreshSystemStats);
    
    // DuplicateDetector connections are set up in setDuplicateDetector() method
    
    // Theme manager connections
    connect(ThemeManager::instance(), &ThemeManager::themeChanged,
            this, &MainWindow::applyTheme);
}

void MainWindow::setupKeyboardShortcuts()
{
    // Ctrl+N - New Scan
    QShortcut* newScanShortcut = new QShortcut(QKeySequence::New, this);
    connect(newScanShortcut, &QShortcut::activated, this, &MainWindow::onNewScanRequested);
    
    // Ctrl+O - Open Results (View History)
    QShortcut* openShortcut = new QShortcut(QKeySequence::Open, this);
    connect(openShortcut, &QShortcut::activated, this, &MainWindow::onViewAllHistoryClicked);
    
    // Ctrl+S - Export Results (if results window is open)
    QShortcut* saveShortcut = new QShortcut(QKeySequence::Save, this);
    connect(saveShortcut, &QShortcut::activated, this, [this]() {
        if (m_resultsWindow && m_resultsWindow->isVisible()) {
            // Trigger export in results window
            m_resultsWindow->exportResults();
        }
    });
    
    // Ctrl+, (Comma) - Settings
    QShortcut* settingsShortcut = new QShortcut(QKeySequence::Preferences, this);
    connect(settingsShortcut, &QShortcut::activated, this, &MainWindow::onSettingsRequested);
    
    // F1 - Help
    QShortcut* helpShortcut = new QShortcut(QKeySequence::HelpContents, this);
    connect(helpShortcut, &QShortcut::activated, this, &MainWindow::onHelpRequested);
    
    // Ctrl+Q - Quit
    QShortcut* quitShortcut = new QShortcut(QKeySequence::Quit, this);
    connect(quitShortcut, &QShortcut::activated, this, &QMainWindow::close);
    
    // Ctrl+R - Refresh System Stats
    QShortcut* refreshShortcut = new QShortcut(QKeySequence::Refresh, this);
    connect(refreshShortcut, &QShortcut::activated, this, &MainWindow::refreshSystemStats);
    
    // F5 - Refresh (alternative)
    QShortcut* f5Shortcut = new QShortcut(QKeySequence(Qt::Key_F5), this);
    connect(f5Shortcut, &QShortcut::activated, this, &MainWindow::refreshSystemStats);
    
    // Ctrl+1 through Ctrl+6 - Quick action presets
    QShortcut* quickScanShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_1), this);
    connect(quickScanShortcut, &QShortcut::activated, this, [this]() {
        onPresetSelected("quick");
    });
    
    QShortcut* downloadsShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_2), this);
    connect(downloadsShortcut, &QShortcut::activated, this, [this]() {
        onPresetSelected("downloads");
    });
    
    QShortcut* photosShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_3), this);
    connect(photosShortcut, &QShortcut::activated, this, [this]() {
        onPresetSelected("photos");
    });
    
    QShortcut* documentsShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_4), this);
    connect(documentsShortcut, &QShortcut::activated, this, [this]() {
        onPresetSelected("documents");
    });
    
    QShortcut* fullSystemShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_5), this);
    connect(fullSystemShortcut, &QShortcut::activated, this, [this]() {
        onPresetSelected("fullsystem");
    });
    
    QShortcut* customShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_6), this);
    connect(customShortcut, &QShortcut::activated, this, [this]() {
        onPresetSelected("custom");
    });
    
    // T19: Additional keyboard shortcuts
    
    // Ctrl+Z - Undo/Restore Files
    QShortcut* undoShortcut = new QShortcut(QKeySequence::Undo, this);
    connect(undoShortcut, &QShortcut::activated, this, &MainWindow::onRestoreRequested);
    
    // Ctrl+Shift+S - Safety Features
    QShortcut* safetyShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_S), this);
    connect(safetyShortcut, &QShortcut::activated, this, &MainWindow::onSafetyFeaturesRequested);
    
    // Ctrl+Shift+A - About Dialog (Section 1.5.2)
    QShortcut* aboutShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_A), this);
    connect(aboutShortcut, &QShortcut::activated, this, &MainWindow::onAboutRequested);
    
    // Escape - Cancel current operation or close active dialog
    QShortcut* escapeShortcut = new QShortcut(QKeySequence(Qt::Key_Escape), this);
    connect(escapeShortcut, &QShortcut::activated, this, [this]() {
        // Cancel scan if in progress
        if (m_fileScanner && m_fileScanner->isScanning()) {
            m_fileScanner->cancelScan();
        }
        // Close scan progress dialog if open
        if (m_scanProgressDialog && m_scanProgressDialog->isVisible()) {
            m_scanProgressDialog->close();
        }
    });
    
    // P3 UI Enhancement Shortcuts (Task 31)
    
    // Ctrl+Shift+F - Advanced Filter
    QShortcut* advancedFilterShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_F), this);
    connect(advancedFilterShortcut, &QShortcut::activated, this, [this]() {
        if (m_resultsWindow && m_resultsWindow->isVisible()) {
            m_resultsWindow->showAdvancedFilterDialog();
        }
    });
    
    // Ctrl+E - View Scan Errors
    QShortcut* viewErrorsShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_E), this);
    connect(viewErrorsShortcut, &QShortcut::activated, this, [this]() {
        if (m_scanErrorDialog && m_fileScanner) {
            QList<FileScanner::ScanErrorInfo> errors = m_fileScanner->getScanErrors();
            if (!errors.isEmpty()) {
                m_scanErrorDialog->setErrors(errors);
                m_scanErrorDialog->show();
                m_scanErrorDialog->raise();
                m_scanErrorDialog->activateWindow();
            }
        }
    });
    
    // Ctrl+P - Pause/Resume Scan
    QShortcut* pauseResumeShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_P), this);
    connect(pauseResumeShortcut, &QShortcut::activated, this, [this]() {
        if (m_fileScanner && m_fileScanner->isScanning()) {
            if (m_fileScanner->isPaused()) {
                m_fileScanner->resumeScan();
            } else {
                m_fileScanner->pauseScan();
            }
        }
    });
    
    // Ctrl+Shift+S - Smart Selection
    QShortcut* smartSelectionShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_S), this);
    connect(smartSelectionShortcut, &QShortcut::activated, this, [this]() {
        if (m_resultsWindow && m_resultsWindow->isVisible()) {
            m_resultsWindow->showSmartSelectionDialog();
        }
    });
    
    // Ctrl+Shift+H - Operation History
    // TODO(Phase3-Feature): Implement operation history dialog
    // Track all file operations (delete, move, restore) with timestamps
    // Allow filtering and searching operation history
    // Priority: LOW - Nice to have for audit trail
    QShortcut* operationHistoryShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_H), this);
    connect(operationHistoryShortcut, &QShortcut::activated, this, [this]() {
        LOG_DEBUG(LogCategories::UI, "Operation history shortcut activated (not yet implemented)");
        QMessageBox::information(this, tr("Coming Soon"),
            tr("Operation History feature will be available in a future update."));
    });
}

void MainWindow::updatePlanIndicator()
{
    // This will be updated based on license status
    if (m_planIndicator) {
        m_planIndicator->setText(tr("👤 Free Plan"));
        // Apply theme-aware styling using QFont
        QFont planFont = m_planIndicator->font();
        planFont.setBold(true);
        m_planIndicator->setFont(planFont);
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
    
    // Left side buttons - use theme-aware sizing
    QSize buttonMinSize = ThemeManager::instance()->getMinimumControlSize(ThemeManager::ControlType::Button);
    
    m_newScanButton = new QPushButton(tr("📁 New Scan"), this);
    m_newScanButton->setMinimumSize(buttonMinSize.width() + 20, buttonMinSize.height());
    // Apply theme-aware styling using QFont
    QFont boldFont = m_newScanButton->font();
    boldFont.setBold(true);
    m_newScanButton->setFont(boldFont);
    m_newScanButton->setToolTip(tr("Start a new scan (Ctrl+N)"));
    
    m_settingsButton = new QPushButton(tr("⚙️ Settings"), this);
    m_settingsButton->setMinimumSize(buttonMinSize.width() + 20, buttonMinSize.height());
    m_settingsButton->setToolTip(tr("Configure application settings (Ctrl+,)"));
    
    m_helpButton = new QPushButton(tr("❓ Help"), this);
    m_helpButton->setMinimumSize(buttonMinSize.width() + 20, buttonMinSize.height());
    m_helpButton->setToolTip(tr("View help and keyboard shortcuts (F1)"));
    
    // Restore button
    QPushButton* restoreButton = new QPushButton(tr("🔄 Restore"), this);
    restoreButton->setMinimumSize(buttonMinSize.width() + 20, buttonMinSize.height());
    restoreButton->setToolTip(tr("Restore files from backups"));
    connect(restoreButton, &QPushButton::clicked, this, &MainWindow::onRestoreRequested);
    
    // T17: Safety Features button
    QPushButton* safetyButton = new QPushButton(tr("🛡️ Safety"), this);
    safetyButton->setMinimumSize(buttonMinSize.width() + 20, buttonMinSize.height());
    safetyButton->setToolTip(tr("Configure file protection and safety features"));
    connect(safetyButton, &QPushButton::clicked, this, &MainWindow::onSafetyFeaturesRequested);
    
    // View Results button - allows access to results window
    QPushButton* testResultsButton = new QPushButton(tr("🔍 View Results"), this);
    // Use minimum size instead of fixed size to allow text to fit
    testResultsButton->setMinimumSize(buttonMinSize.width() + 20, buttonMinSize.height());
    testResultsButton->setToolTip(tr("Open Results Window"));
    connect(testResultsButton, &QPushButton::clicked, this, &MainWindow::showScanResults);
    
    // Spacer
    m_headerLayout->addWidget(m_newScanButton);
    m_headerLayout->addWidget(m_settingsButton);
    m_headerLayout->addWidget(m_helpButton);
    m_headerLayout->addWidget(restoreButton);
    m_headerLayout->addWidget(safetyButton);  // T17
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
    
    // Add minimal stretch to prevent excessive empty space
    m_mainLayout->addStretch(1);
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
    // Apply theme using ThemeManager
    ThemeManager::instance()->applyToWidget(this);
    
    // Update specific child widgets that need theme refresh
    if (m_quickActions) m_quickActions->update();
    if (m_scanHistory) m_scanHistory->update();
    if (m_systemOverview) m_systemOverview->update();
    
    // Update the main window itself
    update();
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
    m_quickScanButton->setToolTip(tr("Scan common locations: Home, Downloads, Documents (Ctrl+1)"));
    
    m_downloadsButton = new QPushButton(tr("📂 Downloads Cleanup"), this);
    m_downloadsButton->setFixedSize(180, 60);
    m_downloadsButton->setToolTip(tr("Find duplicates in Downloads folder (Ctrl+2)"));
    
    m_photoButton = new QPushButton(tr("📸 Photo Cleanup"), this);
    m_photoButton->setFixedSize(180, 60);
    m_photoButton->setToolTip(tr("Find duplicate photos in Pictures folder (Ctrl+3)"));
    
    m_documentsButton = new QPushButton(tr("📄 Documents"), this);
    m_documentsButton->setFixedSize(180, 60);
    m_documentsButton->setToolTip(tr("Scan Documents folder for duplicates (Ctrl+4)"));
    
    m_fullSystemButton = new QPushButton(tr("🖥️ Full System Scan"), this);
    m_fullSystemButton->setFixedSize(180, 60);
    m_fullSystemButton->setToolTip(tr("Comprehensive scan of entire home directory (Ctrl+5)"));
    
    m_customButton = new QPushButton(tr("⭐ Custom Preset"), this);
    m_customButton->setFixedSize(180, 60);
    m_customButton->setToolTip(tr("Configure custom scan settings (Ctrl+6)"));
    
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
void QuickActionsWidget::onQuickScanClicked() { 
    LOG_DEBUG(LogCategories::UI, "Quick Scan button clicked");
    emit presetSelected("quick"); 
}
void QuickActionsWidget::onDownloadsCleanupClicked() { 
    LOG_DEBUG(LogCategories::UI, "Downloads button clicked");
    emit presetSelected("downloads"); 
}
void QuickActionsWidget::onPhotoCleanupClicked() { 
    LOG_DEBUG(LogCategories::UI, "Photo button clicked");
    emit presetSelected("photos"); 
}
void QuickActionsWidget::onDocumentsClicked() { 
    LOG_DEBUG(LogCategories::UI, "Documents button clicked");
    emit presetSelected("documents"); 
}
void QuickActionsWidget::onFullSystemClicked() { 
    LOG_DEBUG(LogCategories::UI, "Full System button clicked");
    emit presetSelected("fullsystem"); 
}
void QuickActionsWidget::onCustomPresetClicked() { 
    LOG_DEBUG(LogCategories::UI, "Custom button clicked");
    emit presetSelected("custom"); 
}

// Main Window scan configuration handler
void MainWindow::handleScanConfiguration(const ScanSetupDialog::ScanConfiguration& config)
{
    LOG_INFO(LogCategories::UI, "=== Starting New Scan ===");
    
    LOG_INFO(LogCategories::UI, QString("Scan Configuration:"));
    LOG_INFO(LogCategories::UI, QString("  - Target paths (%1): %2").arg(config.targetPaths.size()).arg(config.targetPaths.join(", ")));
    LOG_INFO(LogCategories::UI, QString("  - Excluded folders: %1").arg(config.excludeFolders.join(", ")));
    LOG_INFO(LogCategories::UI, QString("  - Detection mode: %1").arg(static_cast<int>(config.detectionMode)));
    LOG_DEBUG(LogCategories::CONFIG, QString("config.minimumFileSize (from dialog): %1 bytes").arg(config.minimumFileSize));
    LOG_INFO(LogCategories::UI, QString("  - Minimum file size: %1 bytes (from dialog)").arg(config.minimumFileSize));
    LOG_INFO(LogCategories::UI, QString("  - Include hidden: %1").arg(config.includeHidden ? "Yes" : "No"));
    LOG_INFO(LogCategories::UI, QString("  - Follow symlinks: %1").arg(config.followSymlinks ? "Yes" : "No"));
    
    // Store scan configuration for duplicate detection
    m_currentScanConfig = config;
    
    // Pass configuration to the FileScanner
    if (m_fileScanner) {
        // Convert ScanSetupDialog configuration to FileScanner::ScanOptions
        FileScanner::ScanOptions scanOptions;
        scanOptions.targetPaths = config.targetPaths;
        scanOptions.minimumFileSize = config.minimumFileSize; // Already in bytes from dialog
        scanOptions.includeHiddenFiles = config.includeHidden;
        scanOptions.scanSystemDirectories = config.includeSystem;
        scanOptions.followSymlinks = config.followSymlinks;
        
        // Parse exclude patterns
        if (!config.excludePatterns.isEmpty()) {
            scanOptions.excludePatterns = config.excludePatterns;
            LOG_DEBUG(LogCategories::UI, QString("  - Exclude patterns: %1").arg(scanOptions.excludePatterns.join(", ")));
        }
        
        LOG_INFO(LogCategories::UI, QString("Initiating FileScanner with %1 target paths").arg(scanOptions.targetPaths.size()));
        LOG_DEBUG(LogCategories::UI, QString("  - Min file size (bytes): %1").arg(scanOptions.minimumFileSize));
        
        // Start the scan
        m_fileScanner->startScan(scanOptions);
        
        // Update UI to show scanning state
        updateScanProgress(0, tr("Scanning..."));
        LOG_INFO(LogCategories::UI, "Scan initiated successfully");
    } else {
        LOG_ERROR(LogCategories::UI, "FileScanner not initialized!");
        showError(tr("Error"), tr("File scanner is not initialized. Please restart the application."));
    }
    
    // Update UI to show scanning state
    if (m_quickActions) {
        m_quickActions->setEnabled(false);
        LOG_DEBUG(LogCategories::UI, "Quick actions disabled during scan");
    }
}


// Duplicate detection handlers
void MainWindow::onScanCompleted()
{
    LOG_DEBUG(LogCategories::UI, "MainWindow::onScanCompleted() called");
    int filesFound = m_fileScanner->getTotalFilesFound();
    qint64 bytesScanned = m_fileScanner->getTotalBytesScanned();
    int errorsEncountered = m_fileScanner->getTotalErrorsEncountered();
    
    LOG_INFO(LogCategories::SCAN, QString("Scan completed - Files found: %1").arg(filesFound));
    LOG_INFO(LogCategories::SCAN, QString("Bytes scanned: %1, Errors: %2").arg(bytesScanned).arg(errorsEncountered));
    
    LOG_INFO(LogCategories::UI, "=== FileScanner: Scan Completed ===");
    LOG_INFO(LogCategories::UI, QString("  - Files found: %1").arg(filesFound));
    LOG_INFO(LogCategories::UI, QString("  - Bytes scanned: %1 (%2)").arg(bytesScanned).arg(formatFileSize(bytesScanned)));
    LOG_INFO(LogCategories::UI, QString("  - Errors encountered: %1").arg(errorsEncountered));
    
    QString status = tr("Scan complete! Found %1 files (%2)")
        .arg(filesFound)
        .arg(formatFileSize(bytesScanned));
    updateScanProgress(100, status);
    
    if (m_fileCountLabel) {
        m_fileCountLabel->setText(tr("Files: %1").arg(filesFound));
    }
    
    // Keep scan progress dialog open - it will be used for duplicate detection
    // Don't hide it here, let duplicate detection reuse it
    
    // Cache scan results for duplicate detection
    m_lastScanResults = m_fileScanner->getScannedFiles();
    
    // Convert FileScanner::FileInfo to DuplicateDetector::FileInfo
    QList<DuplicateDetector::FileInfo> detectorFiles;
    detectorFiles.reserve(m_lastScanResults.size());
    
    for (const auto& scanFile : m_lastScanResults) {
        detectorFiles.append(DuplicateDetector::FileInfo::fromScannerInfo(scanFile));
    }
    
    LOG_INFO(LogCategories::UI, QString("=== Starting Duplicate Detection ==="));
    LOG_INFO(LogCategories::UI, QString("  - Files to analyze: %1").arg(detectorFiles.size()));
    
    // Start duplicate detection if detector is available
    LOG_DEBUG(LogCategories::DUPLICATE, QString("Checking duplicate detector - exists: %1, files: %2")
              .arg(m_duplicateDetector ? "YES" : "NO")
              .arg(detectorFiles.size()));
    
    if (m_duplicateDetector && !detectorFiles.isEmpty()) {
        // Configure detection options based on scan configuration
        DuplicateDetector::DetectionOptions detectionOptions = convertScanConfigToDetectionOptions(m_currentScanConfig);
        m_duplicateDetector->setOptions(detectionOptions);

        LOG_INFO(LogCategories::DUPLICATE, QString("Starting duplicate detection with %1 files using algorithm: %2")
                 .arg(detectorFiles.size())
                 .arg(static_cast<int>(detectionOptions.algorithmType)));

        // Invoke findDuplicates on the background thread using Qt::QueuedConnection
        // This ensures the method executes on the detector's thread, not the main thread
        QMetaObject::invokeMethod(m_duplicateDetector, "findDuplicates",
                                   Qt::QueuedConnection,
                                   Q_ARG(QList<DuplicateDetector::FileInfo>, detectorFiles));
    } else {
        if (!m_duplicateDetector) {
            LOG_ERROR(LogCategories::DUPLICATE, "DuplicateDetector is NULL!");
            LOG_ERROR(LogCategories::UI, "DuplicateDetector not initialized!");
            showError(tr("Error"), tr("Duplicate detector is not initialized. Please restart the application."));
        } else {
            LOG_WARNING(LogCategories::DUPLICATE, "No files to analyze (detectorFiles is empty)");
            LOG_WARNING(LogCategories::UI, "No files to analyze for duplicates");
            showSuccess(tr("Scan Complete"), 
                       tr("Found %1 files totaling %2, but no files to analyze for duplicates")
                       .arg(filesFound)
                       .arg(formatFileSize(bytesScanned)));
        }
        
        // Re-enable quick actions
        if (m_quickActions) {
            m_quickActions->setEnabled(true);
            LOG_DEBUG(LogCategories::UI, "Quick actions re-enabled");
        }
    }
}

void MainWindow::onDuplicateDetectionStarted(int totalFiles)
{
    LOG_INFO(LogCategories::UI, QString("=== Duplicate Detection Started ==="));
    LOG_INFO(LogCategories::UI, QString("  - Total files to process: %1").arg(totalFiles));

    // Show the progress dialog if it exists
    if (m_scanProgressDialog) {
        m_scanProgressDialog->show();
        m_scanProgressDialog->raise();
        m_scanProgressDialog->activateWindow();

        // Initialize progress info
        ScanProgressDialog::ProgressInfo progressInfo;
        progressInfo.operationType = "duplicate_detection";
        progressInfo.status = ScanProgressDialog::OperationStatus::Running;
        progressInfo.totalFiles = totalFiles;
        progressInfo.filesScanned = 0;

        m_scanProgressDialog->updateProgress(progressInfo);
        LOG_DEBUG(LogCategories::UI, "Progress dialog shown for duplicate detection");
    }

    updateScanProgress(0, tr("Detecting duplicates..."));
}

void MainWindow::onDuplicateDetectionProgress(const DuplicateDetector::DetectionProgress& progress)
{
    QString phaseText;
    switch (progress.currentPhase) {
        case DuplicateDetector::DetectionProgress::SizeGrouping:
            phaseText = tr("Grouping by size");
            break;
        case DuplicateDetector::DetectionProgress::HashCalculation:
            phaseText = tr("Calculating hashes");
            break;
        case DuplicateDetector::DetectionProgress::DuplicateGrouping:
            phaseText = tr("Finding duplicates");
            break;
        case DuplicateDetector::DetectionProgress::GeneratingRecommendations:
            phaseText = tr("Generating recommendations");
            break;
        case DuplicateDetector::DetectionProgress::Complete:
            phaseText = tr("Complete");
            break;
    }

    QString status = tr("Detecting duplicates: %1 (%2/%3 files)")
        .arg(phaseText)
        .arg(progress.filesProcessed)
        .arg(progress.totalFiles);

    updateScanProgress(static_cast<int>(progress.percentComplete), status);

    // Update the detailed progress dialog if it exists
    if (m_scanProgressDialog) {
        ScanProgressDialog::ProgressInfo progressInfo;
        progressInfo.operationType = "duplicate_detection";
        progressInfo.status = ScanProgressDialog::OperationStatus::Running;
        progressInfo.totalFiles = progress.totalFiles;
        progressInfo.filesScanned = progress.filesProcessed;

        // Add phase information to the operation type
        if (!progress.currentFile.isEmpty()) {
            progressInfo.currentFile = QFileInfo(progress.currentFile).fileName();
            progressInfo.currentFolder = QFileInfo(progress.currentFile).dir().path();
        }

        // Set file size for current file if available
        if (progress.currentFileSize > 0) {
            progressInfo.bytesScanned = progress.currentFileSize;
        }

        m_scanProgressDialog->updateProgress(progressInfo);
    }

    LOG_DEBUG(LogCategories::UI, QString("Detection progress: %1% - %2").arg(progress.percentComplete, 0, 'f', 1).arg(phaseText));
}

void MainWindow::onDuplicateDetectionCompleted(int totalGroups)
{
    LOG_INFO(LogCategories::UI, "=== Duplicate Detection Completed ===");
    LOG_INFO(LogCategories::UI, QString("  - Duplicate groups found: %1").arg(totalGroups));
    
    // Get results from detector
    QList<DuplicateDetector::DuplicateGroup> groups = m_duplicateDetector->getDuplicateGroups();
    qint64 totalWastedSpace = m_duplicateDetector->getTotalWastedSpace();
    
    LOG_INFO(LogCategories::UI, QString("  - Total wasted space: %1").arg(formatFileSize(totalWastedSpace)));
    
    // Update status
    QString status = tr("Detection complete! Found %1 duplicate groups")
        .arg(totalGroups);
    updateScanProgress(100, status);
    
    if (m_groupCountLabel) {
        m_groupCountLabel->setText(tr("Groups: %1").arg(totalGroups));
    }
    
    if (m_savingsLabel) {
        m_savingsLabel->setText(tr("Savings: %1").arg(formatFileSize(totalWastedSpace)));
    }
    
    // Re-enable quick actions
    if (m_quickActions) {
        m_quickActions->setEnabled(true);
        LOG_DEBUG(LogCategories::UI, "Quick actions re-enabled");
    }
    
    // Save scan to history
    saveScanToHistory(groups);
    
    // Show results if duplicates were found
    if (totalGroups > 0) {
        showSuccess(tr("Detection Complete"), 
                   tr("Found %1 duplicate groups with potential savings of %2")
                   .arg(totalGroups)
                   .arg(formatFileSize(totalWastedSpace)));
        
        // Create results window if needed
        if (!m_resultsWindow) {
            m_resultsWindow = new ResultsWindow(this);
            
            // Register with WindowStateManager
            WindowStateManager::instance()->registerWindow(m_resultsWindow, "ResultsWindow");
            
            // Set FileManager reference
            if (m_fileManager) {
                m_resultsWindow->setFileManager(m_fileManager);
            }
            
            // Connect results window signals
            connect(m_resultsWindow, &ResultsWindow::windowClosed,
                    this, [this]() {
                        LOG_DEBUG(LogCategories::UI, "Results window closed");
                    });
            
            connect(m_resultsWindow, &ResultsWindow::fileOperationRequested,
                    this, [this](const QString& operation, const QStringList& files) {
                        LOG_INFO(LogCategories::UI, QString("File operation requested: %1 on %2 files").arg(operation).arg(files.size()));
                    });
            
            connect(m_resultsWindow, &ResultsWindow::resultsUpdated,
                    this, [this](const ResultsWindow::ScanResults& results) {
                        LOG_DEBUG(LogCategories::UI, QString("Results updated: %1 groups").arg(results.duplicateGroups.size()));
                        
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
        
        // Pass results to ResultsWindow and show it
        LOG_DEBUG(LogCategories::UI, QString("About to display %1 groups in ResultsWindow").arg(groups.size()));
        LOG_INFO(LogCategories::UI, QString("Displaying %1 duplicate groups in ResultsWindow").arg(groups.size()));
        m_resultsWindow->displayDuplicateGroups(groups);
        LOG_DEBUG(LogCategories::UI, "Showing results window");
        m_resultsWindow->show();
        m_resultsWindow->raise();
        m_resultsWindow->activateWindow();
        LOG_DEBUG(LogCategories::UI, "Results window displayed");
    } else {
        showSuccess(tr("Detection Complete"),
                   tr("No duplicate files found. Your files are unique!"));
    }

    // Hide the scan progress dialog now that detection is complete
    if (m_scanProgressDialog) {
        m_scanProgressDialog->hide();
        LOG_DEBUG(LogCategories::UI, "Progress dialog hidden after detection completion");
    }
}

void MainWindow::onDuplicateDetectionError(const QString& error)
{
    LOG_ERROR(LogCategories::UI, QString("=== Duplicate Detection Error ==="));
    LOG_ERROR(LogCategories::UI, QString("  - Error: %1").arg(error));
    
    updateScanProgress(0, tr("Detection failed"));
    
    // Re-enable quick actions
    if (m_quickActions) {
        m_quickActions->setEnabled(true);
        LOG_DEBUG(LogCategories::UI, "Quick actions re-enabled");
    }
    
    showError(tr("Detection Error"),
             tr("An error occurred during duplicate detection:\n%1").arg(error));

    // Hide the scan progress dialog on error
    if (m_scanProgressDialog) {
        m_scanProgressDialog->hide();
        LOG_DEBUG(LogCategories::UI, "Progress dialog hidden after detection error");
    }
}

void MainWindow::saveScanToHistory(const QList<DuplicateDetector::DuplicateGroup>& groups)
{
    LOG_INFO(LogCategories::UI, "Saving scan to history");
    
    // Create scan record
    ScanHistoryManager::ScanRecord record;
    record.scanId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    record.timestamp = QDateTime::currentDateTime();
    record.filesScanned = m_lastScanResults.size();
    record.duplicateGroups = groups.size();
    record.potentialSavings = calculatePotentialSavings(groups);
    record.groups = groups;
    
    // Get target paths from last scan configuration
    // Note: Using generic label since scan configuration paths are not persisted yet
    record.targetPaths << tr("Recent Scan");
    
    // Save to history manager
    ScanHistoryManager::instance()->saveScan(record);
    
    // Refresh history widget
    if (m_scanHistory) {
        m_scanHistory->refreshHistory();
    }
    
    LOG_INFO(LogCategories::UI, QString("Scan saved to history: %1 groups, %2 potential savings")
             .arg(record.duplicateGroups)
             .arg(formatFileSize(record.potentialSavings)));
}

qint64 MainWindow::calculatePotentialSavings(const QList<DuplicateDetector::DuplicateGroup>& groups)
{
    qint64 totalSavings = 0;
    
    for (const auto& group : groups) {
        // Potential savings is the wasted space (total size minus one file to keep)
        totalSavings += group.wastedSpace;
    }
    
    return totalSavings;
}

DuplicateDetector::DetectionOptions MainWindow::convertScanConfigToDetectionOptions(const ScanSetupDialog::ScanConfiguration& config)
{
    DuplicateDetector::DetectionOptions options;
    
    // Convert detection mode to algorithm type
    switch (config.detectionMode) {
        case ScanSetupDialog::DetectionMode::ExactHash:
            options.algorithmType = DetectionAlgorithmFactory::ExactHash;
            options.level = DuplicateDetector::DetectionLevel::Standard;
            break;
        case ScanSetupDialog::DetectionMode::QuickScan:
            options.algorithmType = DetectionAlgorithmFactory::QuickScan;
            options.level = DuplicateDetector::DetectionLevel::Quick;
            break;
        case ScanSetupDialog::DetectionMode::PerceptualHash:
            options.algorithmType = DetectionAlgorithmFactory::PerceptualHash;
            options.level = DuplicateDetector::DetectionLevel::Media;
            break;
        case ScanSetupDialog::DetectionMode::DocumentSimilarity:
            options.algorithmType = DetectionAlgorithmFactory::DocumentSimilarity;
            options.level = DuplicateDetector::DetectionLevel::Deep;
            break;
        case ScanSetupDialog::DetectionMode::Smart:
        default:
            options.algorithmType = DetectionAlgorithmFactory::ExactHash; // Default fallback
            options.level = DuplicateDetector::DetectionLevel::Standard;
            options.enableAutoAlgorithmSelection = true;
            break;
    }
    
    // Copy other configuration options
    options.similarityThreshold = config.similarityThreshold;
    options.enableAutoAlgorithmSelection = config.enableAutoAlgorithmSelection;
    options.minimumFileSize = config.minimumFileSize;
    options.maximumFileSize = config.maximumFileSize;
    options.skipEmptyFiles = config.skipEmptyFiles;
    options.skipSystemFiles = !config.includeSystem;
    
    // Create algorithm-specific configuration
    QVariantMap algorithmConfig;
    
    // Add preset-based configuration
    if (config.algorithmPreset == "Fast") {
        algorithmConfig["preset"] = "fast";
        algorithmConfig["optimization"] = "speed";
    } else if (config.algorithmPreset == "Thorough") {
        algorithmConfig["preset"] = "thorough";
        algorithmConfig["optimization"] = "accuracy";
    } else {
        algorithmConfig["preset"] = "balanced";
        algorithmConfig["optimization"] = "balanced";
    }
    
    options.algorithmConfig = algorithmConfig;
    
    LOG_INFO(LogCategories::UI, QString("Converted scan config to detection options:"));
    LOG_INFO(LogCategories::UI, QString("  - Algorithm type: %1").arg(static_cast<int>(options.algorithmType)));
    LOG_INFO(LogCategories::UI, QString("  - Detection level: %1").arg(static_cast<int>(options.level)));
    LOG_INFO(LogCategories::UI, QString("  - Similarity threshold: %1").arg(options.similarityThreshold));
    LOG_INFO(LogCategories::UI, QString("  - Auto algorithm selection: %1").arg(options.enableAutoAlgorithmSelection ? "Yes" : "No"));
    
    return options;
}
