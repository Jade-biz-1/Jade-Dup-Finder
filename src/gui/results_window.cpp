#include "results_window.h"
#include "app_config.h"
#include <QtWidgets/QApplication>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QFileDialog>
#include <QtGui/QDesktopServices>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QInputDialog>
#include <QtGui/QClipboard>
#include <QtGui/QPixmap>
#include <QtGui/QPainter>
#include <QtGui/QImageReader>
#include <QtCore/QUrl>
#include <QtCore/QDir>
#include <QtCore/QMimeDatabase>
#include <QtCore/QStandardPaths>
#include <QtCore/QDebug>

// Constants
const QSize ResultsWindow::MIN_WINDOW_SIZE(800, 600);
const QSize ResultsWindow::DEFAULT_WINDOW_SIZE(1200, 800);

ResultsWindow::ResultsWindow(QWidget* parent)
    : QMainWindow(parent)
    , m_centralWidget(nullptr)
    , m_mainLayout(nullptr)
    , m_headerPanel(nullptr)
    , m_mainSplitter(nullptr)
    , m_resultsPanel(nullptr)
    , m_resultsTree(nullptr)
    , m_detailsPanel(nullptr)
    , m_detailsTabs(nullptr)
    , m_actionsPanel(nullptr)
    , m_thumbnailTimer(new QTimer(this))
    , m_isProcessingBulkOperation(false)
{
    setWindowTitle(tr("Duplicate Files Found - DupFinder"));
    setMinimumSize(MIN_WINDOW_SIZE);
    resize(DEFAULT_WINDOW_SIZE);
    
    // Make window distinguishable with lighter background
    setStyleSheet(
        "QMainWindow {"
        "    background-color: palette(light);"
        "    border: 2px solid palette(highlight);"
        "    border-radius: 8px;"
        "}"
    );
    
    // Initialize thumbnail timer
    m_thumbnailTimer->setSingleShot(false);
    m_thumbnailTimer->setInterval(100); // Generate thumbnails every 100ms
    
    initializeUI();
    setupConnections();
    applyTheme();
    
    // Don't load sample data - wait for real scan results
    // loadSampleData();
}

ResultsWindow::~ResultsWindow()
{
    // Qt handles cleanup
}

void ResultsWindow::initializeUI()
{
    m_centralWidget = new QWidget(this);
    setCentralWidget(m_centralWidget);
    
    m_mainLayout = new QVBoxLayout(m_centralWidget);
    m_mainLayout->setContentsMargins(12, 12, 12, 12);
    m_mainLayout->setSpacing(8);
    
    createHeaderPanel();
    createMainContent();
    createStatusBar();
    createToolBar();
}

void ResultsWindow::createHeaderPanel()
{
    m_headerPanel = new QWidget(this);
    m_headerLayout = new QHBoxLayout(m_headerPanel);
    m_headerLayout->setContentsMargins(16, 12, 16, 12);
    m_headerLayout->setSpacing(12);
    
    // Title and summary
    m_titleLabel = new QLabel(tr("🔍 Duplicate Files Results"), this);
    m_titleLabel->setStyleSheet("font-size: 18pt; font-weight: bold; color: palette(window-text);");
    
    m_summaryLabel = new QLabel(tr("No results to display"), this);
    m_summaryLabel->setStyleSheet("font-size: 11pt; color: palette(mid);");
    
    // Action buttons
    m_refreshButton = new QPushButton(tr("🔄 Refresh"), this);
    m_exportButton = new QPushButton(tr("📤 Export"), this);
    m_settingsButton = new QPushButton(tr("⚙️ Settings"), this);
    
    // Style header buttons
    QString headerButtonStyle = 
        "QPushButton {"
        "    background: palette(button);"
        "    border: 1px solid palette(mid);"
        "    padding: 8px 16px;"
        "    border-radius: 6px;"
        "    font-weight: bold;"
        "    min-width: 80px;"
        "}"
        "QPushButton:hover {"
        "    background: palette(light);"
        "    border-color: palette(highlight);"
        "}"
    ;
    
    m_refreshButton->setStyleSheet(headerButtonStyle);
    m_exportButton->setStyleSheet(headerButtonStyle);
    m_settingsButton->setStyleSheet(headerButtonStyle);
    
    // Layout header
    QVBoxLayout* titleLayout = new QVBoxLayout();
    titleLayout->addWidget(m_titleLabel);
    titleLayout->addWidget(m_summaryLabel);
    titleLayout->setSpacing(4);
    
    m_headerLayout->addLayout(titleLayout);
    m_headerLayout->addStretch();
    m_headerLayout->addWidget(m_refreshButton);
    m_headerLayout->addWidget(m_exportButton);
    m_headerLayout->addWidget(m_settingsButton);
    
    // Style header panel
    m_headerPanel->setStyleSheet(
        "QWidget {"
        "    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 palette(window), stop:1 palette(base));"
        "    border: 1px solid palette(mid);"
        "    border-radius: 6px;"
        "}"
    );
    
    m_mainLayout->addWidget(m_headerPanel);
}

void ResultsWindow::createMainContent()
{
    m_mainSplitter = new QSplitter(Qt::Horizontal, this);
    
    createResultsTree();
    createDetailsPanel();
    createActionsPanel();
    
    // Set splitter proportions: Results (60%), Details (25%), Actions (15%)
    m_mainSplitter->addWidget(m_resultsPanel);
    m_mainSplitter->addWidget(m_detailsPanel);
    m_mainSplitter->addWidget(m_actionsPanel);
    m_mainSplitter->setSizes({600, 300, 200});
    
    m_mainLayout->addWidget(m_mainSplitter, 1);  // Expand to fill space
}

void ResultsWindow::createResultsTree()
{
    m_resultsPanel = new QWidget(this);
    m_resultsPanelLayout = new QVBoxLayout(m_resultsPanel);
    m_resultsPanelLayout->setContentsMargins(8, 8, 8, 8);
    m_resultsPanelLayout->setSpacing(8);
    
    // Filter panel
    m_filterPanel = new QWidget(this);
    m_filterLayout = new QHBoxLayout(m_filterPanel);
    m_filterLayout->setContentsMargins(0, 0, 0, 0);
    m_filterLayout->setSpacing(8);
    
    m_filterLabel = new QLabel(tr("Filter:"), this);
    m_searchFilter = new QLineEdit(this);
    m_searchFilter->setPlaceholderText(tr("Search files..."));
    
    m_sizeFilter = new QComboBox(this);
    m_sizeFilter->addItem(tr("All Sizes"));
    m_sizeFilter->addItem(tr("> 1MB"));
    m_sizeFilter->addItem(tr("> 10MB"));
    m_sizeFilter->addItem(tr("> 100MB"));
    
    m_typeFilter = new QComboBox(this);
    m_typeFilter->addItem(tr("All Types"));
    m_typeFilter->addItem(tr("Images"));
    m_typeFilter->addItem(tr("Documents"));
    m_typeFilter->addItem(tr("Videos"));
    m_typeFilter->addItem(tr("Audio"));
    
    m_sortCombo = new QComboBox(this);
    m_sortCombo->addItem(tr("Sort by Size"));
    m_sortCombo->addItem(tr("Sort by Name"));
    m_sortCombo->addItem(tr("Sort by Date"));
    m_sortCombo->addItem(tr("Sort by Type"));
    
    m_clearFiltersButton = new QPushButton(tr("Clear"), this);
    
    m_filterLayout->addWidget(m_filterLabel);
    m_filterLayout->addWidget(m_searchFilter, 1);
    m_filterLayout->addWidget(m_sizeFilter);
    m_filterLayout->addWidget(m_typeFilter);
    m_filterLayout->addWidget(m_sortCombo);
    m_filterLayout->addWidget(m_clearFiltersButton);
    
    // Selection panel
    m_selectionPanel = new QWidget(this);
    m_selectionLayout = new QHBoxLayout(m_selectionPanel);
    m_selectionLayout->setContentsMargins(0, 0, 0, 0);
    m_selectionLayout->setSpacing(8);
    
    m_selectAllCheckbox = new QCheckBox(tr("Select All"), this);
    m_selectRecommendedButton = new QPushButton(tr("Select Recommended"), this);
    m_selectByTypeButton = new QPushButton(tr("Select by Type"), this);
    m_clearSelectionButton = new QPushButton(tr("Clear Selection"), this);
    m_selectionSummaryLabel = new QLabel(tr("0 files selected"), this);
    
    m_selectionLayout->addWidget(m_selectAllCheckbox);
    m_selectionLayout->addWidget(m_selectRecommendedButton);
    m_selectionLayout->addWidget(m_selectByTypeButton);
    m_selectionLayout->addWidget(m_clearSelectionButton);
    m_selectionLayout->addStretch();
    m_selectionLayout->addWidget(m_selectionSummaryLabel);
    
    // Results tree
    m_resultsTree = new QTreeWidget(this);
    m_resultsTree->setHeaderLabels({tr("File"), tr("Size"), tr("Modified"), tr("Path")});
    m_resultsTree->setRootIsDecorated(true);
    m_resultsTree->setAlternatingRowColors(true);
    m_resultsTree->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_resultsTree->setSortingEnabled(true);
    m_resultsTree->setItemsExpandable(true);
    
    // Configure tree columns
    QHeaderView* header = m_resultsTree->header();
    header->setSectionResizeMode(0, QHeaderView::Stretch);
    header->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(3, QHeaderView::Interactive);
    
    m_resultsTree->setStyleSheet(
        "QTreeWidget {"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    background: palette(base);"
        "    alternate-background-color: palette(alternate-base);"
        "}"
        "QTreeWidget::item {"
        "    padding: 4px;"
        "    margin: 1px;"
        "}"
        "QTreeWidget::item:selected {"
        "    background: palette(highlight);"
        "    color: palette(highlighted-text);"
        "}"
    );
    
    // Add to layout
    m_resultsPanelLayout->addWidget(m_filterPanel);
    m_resultsPanelLayout->addWidget(m_selectionPanel);
    m_resultsPanelLayout->addWidget(m_resultsTree, 1);
}

void ResultsWindow::createDetailsPanel()
{
    m_detailsPanel = new QWidget(this);
    m_detailsPanelLayout = new QVBoxLayout(m_detailsPanel);
    m_detailsPanelLayout->setContentsMargins(8, 8, 8, 8);
    m_detailsPanelLayout->setSpacing(8);
    
    m_detailsTabs = new QTabWidget(this);
    
    // File Info Tab
    m_fileInfoTab = new QWidget(this);
    m_fileInfoLayout = new QVBoxLayout(m_fileInfoTab);
    m_fileInfoLayout->setContentsMargins(12, 12, 12, 12);
    m_fileInfoLayout->setSpacing(8);
    
    // Preview area
    m_previewScrollArea = new QScrollArea(this);
    m_previewLabel = new QLabel(this);
    m_previewLabel->setAlignment(Qt::AlignCenter);
    m_previewLabel->setStyleSheet(
        "QLabel {"
        "    border: 2px dashed palette(mid);"
        "    border-radius: 4px;"
        "    background: palette(base);"
        "    min-height: 150px;"
        "    color: palette(mid);"
        "}"
    );
    m_previewLabel->setText(tr("No file selected"));
    m_previewScrollArea->setWidget(m_previewLabel);
    m_previewScrollArea->setWidgetResizable(true);
    
    // File details
    m_fileNameLabel = new QLabel(tr("Name: -"), this);
    m_fileSizeLabel = new QLabel(tr("Size: -"), this);
    m_filePathLabel = new QLabel(tr("Path: -"), this);
    m_fileDateLabel = new QLabel(tr("Modified: -"), this);
    m_fileTypeLabel = new QLabel(tr("Type: -"), this);
    m_fileHashLabel = new QLabel(tr("Hash: -"), this);
    
    // Style detail labels
    QString detailLabelStyle = 
        "QLabel {"
        "    padding: 4px;"
        "    border-bottom: 1px solid palette(mid);"
        "    font-size: 10pt;"
        "}"
    ;
    m_fileNameLabel->setStyleSheet(detailLabelStyle);
    m_fileSizeLabel->setStyleSheet(detailLabelStyle);
    m_filePathLabel->setStyleSheet(detailLabelStyle);
    m_fileDateLabel->setStyleSheet(detailLabelStyle);
    m_fileTypeLabel->setStyleSheet(detailLabelStyle);
    m_fileHashLabel->setStyleSheet(detailLabelStyle);
    
    m_fileInfoLayout->addWidget(m_previewScrollArea);
    m_fileInfoLayout->addWidget(m_fileNameLabel);
    m_fileInfoLayout->addWidget(m_fileSizeLabel);
    m_fileInfoLayout->addWidget(m_filePathLabel);
    m_fileInfoLayout->addWidget(m_fileDateLabel);
    m_fileInfoLayout->addWidget(m_fileTypeLabel);
    m_fileInfoLayout->addWidget(m_fileHashLabel);
    m_fileInfoLayout->addStretch();
    
    // Group Info Tab
    m_groupInfoTab = new QWidget(this);
    m_groupInfoLayout = new QVBoxLayout(m_groupInfoTab);
    m_groupInfoLayout->setContentsMargins(12, 12, 12, 12);
    m_groupInfoLayout->setSpacing(8);
    
    m_groupSummaryLabel = new QLabel(tr("No group selected"), this);
    m_groupSummaryLabel->setStyleSheet("font-weight: bold; padding: 8px; background: palette(base); border-radius: 4px;");
    
    m_groupFilesTable = new QTableWidget(0, 4, this);
    m_groupFilesTable->setHorizontalHeaderLabels({tr("File"), tr("Size"), tr("Modified"), tr("Recommended")});
    m_groupFilesTable->horizontalHeader()->setStretchLastSection(true);
    m_groupFilesTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_groupFilesTable->setAlternatingRowColors(true);
    
    m_groupInfoLayout->addWidget(m_groupSummaryLabel);
    m_groupInfoLayout->addWidget(m_groupFilesTable, 1);
    
    // Add tabs
    m_detailsTabs->addTab(m_fileInfoTab, tr("📄 File Info"));
    m_detailsTabs->addTab(m_groupInfoTab, tr("📁 Group Info"));
    
    m_detailsPanelLayout->addWidget(m_detailsTabs, 1);
}

void ResultsWindow::createActionsPanel()
{
    m_actionsPanel = new QWidget(this);
    m_actionsPanelLayout = new QVBoxLayout(m_actionsPanel);
    m_actionsPanelLayout->setContentsMargins(8, 8, 8, 8);
    m_actionsPanelLayout->setSpacing(12);
    
    // File Actions Group
    m_fileActionsGroup = new QGroupBox(tr("📄 File Actions"), this);
    m_fileActionsLayout = new QVBoxLayout(m_fileActionsGroup);
    m_fileActionsLayout->setSpacing(6);
    
    m_deleteButton = new QPushButton(tr("🗑️ Delete File"), this);
    m_moveButton = new QPushButton(tr("📁 Move File"), this);
    m_ignoreButton = new QPushButton(tr("👁️ Ignore File"), this);
    m_previewButton = new QPushButton(tr("👀 Preview"), this);
    m_openLocationButton = new QPushButton(tr("📂 Open Location"), this);
    m_copyPathButton = new QPushButton(tr("📋 Copy Path"), this);
    
    // Style action buttons
    QString actionButtonStyle = 
        "QPushButton {"
        "    text-align: left;"
        "    padding: 8px 12px;"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    background: palette(button);"
        "}"
        "QPushButton:hover {"
        "    background: palette(light);"
        "    border-color: palette(highlight);"
        "}"
        "QPushButton:pressed {"
        "    background: palette(mid);"
        "}"
        "QPushButton:disabled {"
        "    color: palette(mid);"
        "    background: palette(window);"
        "}"
    ;
    
    m_deleteButton->setStyleSheet(actionButtonStyle);
    m_moveButton->setStyleSheet(actionButtonStyle);
    m_ignoreButton->setStyleSheet(actionButtonStyle);
    m_previewButton->setStyleSheet(actionButtonStyle);
    m_openLocationButton->setStyleSheet(actionButtonStyle);
    m_copyPathButton->setStyleSheet(actionButtonStyle);
    
    m_fileActionsLayout->addWidget(m_deleteButton);
    m_fileActionsLayout->addWidget(m_moveButton);
    m_fileActionsLayout->addWidget(m_ignoreButton);
    m_fileActionsLayout->addWidget(m_previewButton);
    m_fileActionsLayout->addWidget(m_openLocationButton);
    m_fileActionsLayout->addWidget(m_copyPathButton);
    
    // Bulk Actions Group
    m_bulkActionsGroup = new QGroupBox(tr("⚡ Bulk Actions"), this);
    m_bulkActionsLayout = new QVBoxLayout(m_bulkActionsGroup);
    m_bulkActionsLayout->setSpacing(6);
    
    m_bulkDeleteButton = new QPushButton(tr("🗑️ Delete Selected"), this);
    m_bulkMoveButton = new QPushButton(tr("📁 Move Selected"), this);
    m_bulkIgnoreButton = new QPushButton(tr("👁️ Ignore Selected"), this);
    
    // Style bulk buttons with warning colors
    QString bulkButtonStyle = 
        "QPushButton {"
        "    text-align: left;"
        "    padding: 8px 12px;"
        "    border: 1px solid palette(mid);"
        "    border-radius: 4px;"
        "    background: palette(button);"
        "    font-weight: bold;"
        "}"
        "QPushButton:hover {"
        "    background: orange;"
        "    border-color: red;"
        "    color: white;"
        "}"
    ;
    
    m_bulkDeleteButton->setStyleSheet(bulkButtonStyle);
    m_bulkMoveButton->setStyleSheet(bulkButtonStyle);
    m_bulkIgnoreButton->setStyleSheet(bulkButtonStyle);
    
    m_bulkActionsLayout->addWidget(m_bulkDeleteButton);
    m_bulkActionsLayout->addWidget(m_bulkMoveButton);
    m_bulkActionsLayout->addWidget(m_bulkIgnoreButton);
    
    // Add groups to panel
    m_actionsPanelLayout->addWidget(m_fileActionsGroup);
    m_actionsPanelLayout->addWidget(m_bulkActionsGroup);
    m_actionsPanelLayout->addStretch();
    
    // Initially disable all actions
    m_deleteButton->setEnabled(false);
    m_moveButton->setEnabled(false);
    m_ignoreButton->setEnabled(false);
    m_previewButton->setEnabled(false);
    m_openLocationButton->setEnabled(false);
    m_copyPathButton->setEnabled(false);
    m_bulkDeleteButton->setEnabled(false);
    m_bulkMoveButton->setEnabled(false);
    m_bulkIgnoreButton->setEnabled(false);
}

void ResultsWindow::createStatusBar()
{
    m_statusLabel = new QLabel(tr("Ready"), this);
    m_progressBar = new QProgressBar(this);
    m_progressBar->setVisible(false);
    m_statisticsLabel = new QLabel(tr("No results"), this);
    
    statusBar()->addWidget(m_statusLabel);
    statusBar()->addPermanentWidget(m_statisticsLabel);
    statusBar()->addPermanentWidget(m_progressBar);
}

void ResultsWindow::createToolBar()
{
    // For now, keep it simple without a toolbar
    // Can be added later if needed
}

void ResultsWindow::setupConnections()
{
    // Header buttons
    connect(m_refreshButton, &QPushButton::clicked, this, &ResultsWindow::refreshResults);
    connect(m_exportButton, &QPushButton::clicked, this, &ResultsWindow::exportResults);
    connect(m_settingsButton, &QPushButton::clicked, this, [this]() {
        qDebug() << "Settings clicked";
    });
    
    // Filter and search
    connect(m_searchFilter, &QLineEdit::textChanged, this, &ResultsWindow::onFilterChanged);
    connect(m_sizeFilter, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ResultsWindow::onFilterChanged);
    connect(m_typeFilter, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ResultsWindow::onFilterChanged);
    connect(m_sortCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ResultsWindow::onSortChanged);
    connect(m_clearFiltersButton, &QPushButton::clicked, this, [this]() {
        m_searchFilter->clear();
        m_sizeFilter->setCurrentIndex(0);
        m_typeFilter->setCurrentIndex(0);
        m_sortCombo->setCurrentIndex(0);
    });
    
    // Selection
    connect(m_selectAllCheckbox, &QCheckBox::toggled, this, &ResultsWindow::selectAllDuplicates);
    connect(m_selectRecommendedButton, &QPushButton::clicked, this, &ResultsWindow::selectRecommended);
    connect(m_selectByTypeButton, &QPushButton::clicked, this, [this]() {
        selectByType("image"); // Example
    });
    connect(m_clearSelectionButton, &QPushButton::clicked, this, &ResultsWindow::selectNoneFiles);
    
    // Tree widget
    connect(m_resultsTree, &QTreeWidget::itemSelectionChanged, this, &ResultsWindow::onFileSelectionChanged);
    connect(m_resultsTree, &QTreeWidget::itemExpanded, this, &ResultsWindow::onGroupExpanded);
    connect(m_resultsTree, &QTreeWidget::itemCollapsed, this, &ResultsWindow::onGroupCollapsed);
    
    // File actions
    connect(m_deleteButton, &QPushButton::clicked, this, &ResultsWindow::deleteSelectedFiles);
    connect(m_moveButton, &QPushButton::clicked, this, &ResultsWindow::moveSelectedFiles);
    connect(m_ignoreButton, &QPushButton::clicked, this, &ResultsWindow::ignoreSelectedFiles);
    connect(m_previewButton, &QPushButton::clicked, this, &ResultsWindow::previewSelectedFile);
    connect(m_openLocationButton, &QPushButton::clicked, this, &ResultsWindow::openFileLocation);
    connect(m_copyPathButton, &QPushButton::clicked, this, &ResultsWindow::copyFilePath);
    
    // Bulk actions
    connect(m_bulkDeleteButton, &QPushButton::clicked, this, &ResultsWindow::performBulkDelete);
    connect(m_bulkMoveButton, &QPushButton::clicked, this, &ResultsWindow::performBulkMove);
    connect(m_bulkIgnoreButton, &QPushButton::clicked, this, [this]() {
        qDebug() << "Bulk ignore not implemented yet";
    });
    
    // Thumbnail timer
    connect(m_thumbnailTimer, &QTimer::timeout, this, [this]() {
        // Generate thumbnails in batches
        qDebug() << "Generating thumbnails...";
    });
}

void ResultsWindow::applyTheme()
{
    // Apply consistent theme
    updateStatusBar();
}

void ResultsWindow::loadSampleData()
{
    // Create sample duplicate groups for testing
    ScanResults sampleResults;
    sampleResults.scanPath = "/home/user/Documents";
    sampleResults.scanTime = QDateTime::currentDateTime();
    sampleResults.scanDuration = "2m 15s";
    sampleResults.totalFilesScanned = 15420;
    
    // Group 1: Images
    DuplicateGroup group1;
    group1.groupId = "group_001";
    group1.isExpanded = false;
    group1.hasSelection = false;
    
    DuplicateFile file1;
    file1.filePath = "/home/user/Pictures/vacation.jpg";
    file1.fileName = "vacation.jpg";
    file1.directory = "/home/user/Pictures";
    file1.fileSize = 2048576; // 2MB
    file1.lastModified = QDateTime::currentDateTime().addDays(-10);
    file1.created = QDateTime::currentDateTime().addDays(-15);
    file1.hash = "abc123def456";
    file1.isSelected = false;
    file1.isMarkedForDeletion = false;
    file1.fileType = "JPEG Image";
    
    DuplicateFile file2;
    file2.filePath = "/home/user/Downloads/vacation_copy.jpg";
    file2.fileName = "vacation_copy.jpg";
    file2.directory = "/home/user/Downloads";
    file2.fileSize = 2048576; // Same size
    file2.lastModified = QDateTime::currentDateTime().addDays(-5);
    file2.created = QDateTime::currentDateTime().addDays(-5);
    file2.hash = "abc123def456"; // Same hash
    file2.isSelected = false;
    file2.isMarkedForDeletion = false;
    file2.fileType = "JPEG Image";
    
    group1.files << file1 << file2;
    group1.fileCount = 2;
    group1.totalSize = file1.fileSize + file2.fileSize;
    group1.primaryFile = file1.filePath; // Recommend keeping original
    
    // Group 2: Documents
    DuplicateGroup group2;
    group2.groupId = "group_002";
    group2.isExpanded = false;
    group2.hasSelection = false;
    
    DuplicateFile file3;
    file3.filePath = "/home/user/Documents/report.pdf";
    file3.fileName = "report.pdf";
    file3.directory = "/home/user/Documents";
    file3.fileSize = 1024000; // 1MB
    file3.lastModified = QDateTime::currentDateTime().addDays(-3);
    file3.created = QDateTime::currentDateTime().addDays(-7);
    file3.hash = "xyz789abc123";
    file3.isSelected = false;
    file3.isMarkedForDeletion = false;
    file3.fileType = "PDF Document";
    
    DuplicateFile file4;
    file4.filePath = "/home/user/Desktop/report_backup.pdf";
    file4.fileName = "report_backup.pdf";
    file4.directory = "/home/user/Desktop";
    file4.fileSize = 1024000;
    file4.lastModified = QDateTime::currentDateTime().addDays(-1);
    file4.created = QDateTime::currentDateTime().addDays(-1);
    file4.hash = "xyz789abc123";
    file4.isSelected = false;
    file4.isMarkedForDeletion = false;
    file4.fileType = "PDF Document";
    
    DuplicateFile file5;
    file5.filePath = "/home/user/Downloads/report (1).pdf";
    file5.fileName = "report (1).pdf";
    file5.directory = "/home/user/Downloads";
    file5.fileSize = 1024000;
    file5.lastModified = QDateTime::currentDateTime().addSecs(-2 * 3600); // 2 hours ago
    file5.created = QDateTime::currentDateTime().addSecs(-2 * 3600);
    file5.hash = "xyz789abc123";
    file5.isSelected = false;
    file5.isMarkedForDeletion = false;
    file5.fileType = "PDF Document";
    
    group2.files << file3 << file4 << file5;
    group2.fileCount = 3;
    group2.totalSize = file3.fileSize + file4.fileSize + file5.fileSize;
    group2.primaryFile = file3.filePath; // Recommend keeping original
    
    sampleResults.duplicateGroups << group1 << group2;
    sampleResults.calculateTotals();
    
    displayResults(sampleResults);
}

void ResultsWindow::displayResults(const ScanResults& results)
{
    m_currentResults = results;
    
    // Update header
    m_summaryLabel->setText(tr("%1 duplicate groups found, %2 potential savings")
                           .arg(results.duplicateGroups.size())
                           .arg(formatFileSize(results.potentialSavings)));
    
    populateResultsTree();
    updateStatusBar();
}

void ResultsWindow::populateResultsTree()
{
    m_resultsTree->clear();
    
    for (int i = 0; i < m_currentResults.duplicateGroups.size(); ++i) {
        const auto& group = m_currentResults.duplicateGroups[i];
        
        // Create group item
        QTreeWidgetItem* groupItem = new QTreeWidgetItem(m_resultsTree);
        updateGroupItem(groupItem, group);
        
        // Add file items
        for (const auto& file : group.files) {
            QTreeWidgetItem* fileItem = new QTreeWidgetItem(groupItem);
            updateFileItem(fileItem, file);
        }
        
        groupItem->setExpanded(group.isExpanded);
    }
}

void ResultsWindow::updateGroupItem(QTreeWidgetItem* groupItem, const DuplicateGroup& group)
{
    groupItem->setText(0, tr("📁 Group: %1 files").arg(group.fileCount));
    groupItem->setText(1, formatFileSize(group.totalSize));
    groupItem->setText(2, tr("Waste: %1").arg(formatFileSize(group.getWastedSpace())));
    groupItem->setText(3, tr("%1 duplicates").arg(group.fileCount - 1));
    
    // Store group data
    groupItem->setData(0, Qt::UserRole, QVariant::fromValue(group.groupId));
    
    // Style group item
    QFont font = groupItem->font(0);
    font.setBold(true);
    groupItem->setFont(0, font);
}

void ResultsWindow::updateFileItem(QTreeWidgetItem* fileItem, const DuplicateFile& file)
{
    fileItem->setText(0, file.fileName);
    fileItem->setText(1, formatFileSize(file.fileSize));
    fileItem->setText(2, file.lastModified.toString("yyyy-MM-dd hh:mm"));
    fileItem->setText(3, file.directory);
    
    // Store file data
    fileItem->setData(0, Qt::UserRole, QVariant::fromValue(file.filePath));
    
    // Add checkbox
    fileItem->setCheckState(0, file.isSelected ? Qt::Checked : Qt::Unchecked);
    
    // Style recommended file differently
    if (!m_currentResults.duplicateGroups.isEmpty()) {
        QString recommended = getRecommendedFileToKeep(m_currentResults.duplicateGroups.first());
        if (file.filePath == recommended) {
            QFont font = fileItem->font(0);
            font.setItalic(true);
            fileItem->setFont(0, font);
            fileItem->setToolTip(0, tr("Recommended to keep"));
        }
    }
}

// Utility methods implementation
QString ResultsWindow::formatFileSize(qint64 bytes) const
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

QString ResultsWindow::getRecommendedFileToKeep(const DuplicateGroup& group) const
{
    if (group.files.isEmpty()) {
        return QString();
    }
    
    // Recommend keeping the oldest file (likely the original)
    const DuplicateFile* recommended = &group.files.first();
    for (const auto& file : group.files) {
        if (file.created < recommended->created) {
            recommended = &file;
        }
    }
    
    return recommended->filePath;
}

bool ResultsWindow::isImageFile(const QString& filePath) const
{
    QFileInfo fileInfo(filePath);
    QString suffix = fileInfo.suffix().toLower();
    QStringList imageExtensions = {"jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp"};
    return imageExtensions.contains(suffix);
}

bool ResultsWindow::isVideoFile(const QString& filePath) const
{
    QFileInfo fileInfo(filePath);
    QString suffix = fileInfo.suffix().toLower();
    QStringList videoExtensions = {"mp4", "avi", "mkv", "mov", "wmv", "flv", "webm"};
    return videoExtensions.contains(suffix);
}

// Slot implementations
void ResultsWindow::updateStatusBar()
{
    if (m_currentResults.duplicateGroups.isEmpty()) {
        m_statusLabel->setText(tr("No duplicates found"));
        m_statisticsLabel->setText(tr("Ready"));
    } else {
        m_statusLabel->setText(tr("%1 groups, %2 files")
                              .arg(m_currentResults.duplicateGroups.size())
                              .arg(m_currentResults.totalDuplicatesFound));
        m_statisticsLabel->setText(tr("Savings: %1")
                                  .arg(formatFileSize(m_currentResults.potentialSavings)));
    }
}

void ResultsWindow::onFileSelectionChanged()
{
    QList<QTreeWidgetItem*> selectedItems = m_resultsTree->selectedItems();
    bool hasFileSelected = false;
    bool hasMultipleSelected = selectedItems.size() > 1;
    
    // Check if any files are selected
    for (QTreeWidgetItem* item : selectedItems) {
        if (item->parent() != nullptr) { // It's a file item, not a group
            hasFileSelected = true;
            break;
        }
    }
    
    // Enable/disable single file actions
    m_deleteButton->setEnabled(hasFileSelected && !hasMultipleSelected);
    m_moveButton->setEnabled(hasFileSelected && !hasMultipleSelected);
    m_ignoreButton->setEnabled(hasFileSelected && !hasMultipleSelected);
    m_previewButton->setEnabled(hasFileSelected && !hasMultipleSelected);
    m_openLocationButton->setEnabled(hasFileSelected && !hasMultipleSelected);
    m_copyPathButton->setEnabled(hasFileSelected && !hasMultipleSelected);
    
    // Enable/disable bulk actions
    m_bulkDeleteButton->setEnabled(hasFileSelected);
    m_bulkMoveButton->setEnabled(hasFileSelected);
    m_bulkIgnoreButton->setEnabled(hasFileSelected);
    
    updateSelectionSummary();
}

void ResultsWindow::updateSelectionSummary()
{
    QList<DuplicateFile> selectedFiles = getSelectedFiles();
    if (selectedFiles.isEmpty()) {
        m_selectionSummaryLabel->setText(tr("0 files selected"));
    } else {
        qint64 totalSize = getSelectedFilesSize();
        m_selectionSummaryLabel->setText(tr("%1 files selected (%2)")
                                        .arg(selectedFiles.size())
                                        .arg(formatFileSize(totalSize)));
    }
}

QList<ResultsWindow::DuplicateFile> ResultsWindow::getSelectedFiles() const
{
    QList<DuplicateFile> selectedFiles;
    QTreeWidgetItemIterator it(m_resultsTree);
    
    while (*it) {
        QTreeWidgetItem* item = *it;
        if (item->parent() != nullptr && item->checkState(0) == Qt::Checked) {
            QString filePath = item->data(0, Qt::UserRole).toString();
            
            // Find the file in our data
            for (const auto& group : m_currentResults.duplicateGroups) {
                for (const auto& file : group.files) {
                    if (file.filePath == filePath) {
                        selectedFiles.append(file);
                        break;
                    }
                }
            }
        }
        ++it;
    }
    
    return selectedFiles;
}

qint64 ResultsWindow::getSelectedFilesSize() const
{
    qint64 totalSize = 0;
    QList<DuplicateFile> selected = getSelectedFiles();
    for (const auto& file : selected) {
        totalSize += file.fileSize;
    }
    return totalSize;
}

int ResultsWindow::getSelectedFilesCount() const
{
    return static_cast<int>(getSelectedFiles().size());
}

// Action slots (basic implementations)
void ResultsWindow::refreshResults()
{
    qDebug() << "Refreshing results...";
    populateResultsTree();
    updateStatusBar();
}

void ResultsWindow::exportResults()
{
    QString fileName = QFileDialog::getSaveFileName(this, 
                                                   tr("Export Results"),
                                                   "duplicate_files_report.txt",
                                                   tr("Text Files (*.txt);;CSV Files (*.csv)"));
    if (!fileName.isEmpty()) {
        qDebug() << "Exporting results to:" << fileName;
        // TODO: Implement export functionality
        QMessageBox::information(this, tr("Export"), tr("Export functionality will be implemented soon."));
    }
}

void ResultsWindow::selectAllDuplicates()
{
    LOG_INFO("User clicked 'Select All' button");
    
    int selectedCount = 0;
    QTreeWidgetItemIterator it(m_resultsTree);
    while (*it) {
        QTreeWidgetItem* item = *it;
        if (item->parent() != nullptr) { // File item
            item->setCheckState(0, Qt::Checked);
            selectedCount++;
        }
        ++it;
    }
    
    LOG_INFO(QString("Selected all %1 files").arg(selectedCount));
    updateSelectionSummary();
}

void ResultsWindow::selectNoneFiles()
{
    LOG_INFO("User clicked 'Clear Selection' button");
    
    QTreeWidgetItemIterator it(m_resultsTree);
    while (*it) {
        QTreeWidgetItem* item = *it;
        if (item->parent() != nullptr) { // File item
            item->setCheckState(0, Qt::Unchecked);
        }
        ++it;
    }
    
    LOG_INFO("Cleared all selections");
    updateSelectionSummary();
}

void ResultsWindow::selectRecommended()
{
    LOG_INFO("User clicked 'Select Recommended' button");
    
    int selectedCount = 0;
    // Select all non-recommended files (files to delete)
    for (const auto& group : m_currentResults.duplicateGroups) {
        QString recommended = getRecommendedFileToKeep(group);
        LOG_DEBUG(QString("Group recommended file: %1").arg(recommended));
        
        QTreeWidgetItemIterator it(m_resultsTree);
        while (*it) {
            QTreeWidgetItem* item = *it;
            if (item->parent() != nullptr) {
                QString filePath = item->data(0, Qt::UserRole).toString();
                if (filePath != recommended && 
                    group.files.contains(DuplicateFile{filePath, "", "", 0, QDateTime(), QDateTime(), "", QPixmap(), false, false, ""})) {
                    item->setCheckState(0, Qt::Checked);
                    selectedCount++;
                } else if (filePath == recommended) {
                    item->setCheckState(0, Qt::Unchecked);
                }
            }
            ++it;
        }
    }
    
    LOG_INFO(QString("Selected %1 recommended files for deletion").arg(selectedCount));
    updateSelectionSummary();
}

void ResultsWindow::selectBySize(qint64 minSize)
{
    LOG_INFO(QString("Selecting files by size (min: %1 bytes)").arg(minSize));
    
    int selectedCount = 0;
    QTreeWidgetItemIterator it(m_resultsTree);
    while (*it) {
        QTreeWidgetItem* item = *it;
        if (item->parent() != nullptr) {
            QString filePath = item->data(0, Qt::UserRole).toString();
            
            // Find file size
            for (const auto& group : m_currentResults.duplicateGroups) {
                for (const auto& file : group.files) {
                    if (file.filePath == filePath && file.fileSize >= minSize) {
                        item->setCheckState(0, Qt::Checked);
                        selectedCount++;
                    }
                }
            }
        }
        ++it;
    }
    
    LOG_INFO(QString("Selected %1 files by size").arg(selectedCount));
    updateSelectionSummary();
}

void ResultsWindow::selectByType(const QString& fileType)
{
    LOG_INFO(QString("Selecting files by type: %1").arg(fileType));
    
    int selectedCount = 0;
    QTreeWidgetItemIterator it(m_resultsTree);
    while (*it) {
        QTreeWidgetItem* item = *it;
        if (item->parent() != nullptr) {
            QString filePath = item->data(0, Qt::UserRole).toString();
            
            bool shouldSelect = false;
            if (fileType == "image" && isImageFile(filePath)) {
                shouldSelect = true;
            } else if (fileType == "video" && isVideoFile(filePath)) {
                shouldSelect = true;
            }
            
            if (shouldSelect) {
                item->setCheckState(0, Qt::Checked);
                selectedCount++;
            }
        }
        ++it;
    }
    
    LOG_INFO(QString("Selected %1 files by type").arg(selectedCount));
    updateSelectionSummary();
}

// Basic file action implementations
void ResultsWindow::deleteSelectedFiles()
{
    QList<DuplicateFile> selected = getSelectedFiles();
    if (selected.isEmpty()) return;
    
    QString message = tr("Are you sure you want to delete %1 file(s)?")
                     .arg(selected.size());
    
    QMessageBox::StandardButton reply = QMessageBox::question(this, 
                                                              tr("Confirm Delete"),
                                                              message,
                                                              QMessageBox::Yes | QMessageBox::No);
    
    if (reply == QMessageBox::Yes) {
        LOG_WARNING(QString("File deletion confirmed for %1 files (not yet implemented)").arg(selected.size()));
        for (int i = 0; i < qMin(5, selected.size()); ++i) {
            LOG_DEBUG(QString("  - Would delete: %1").arg(selected[i].filePath));
        }
        if (selected.size() > 5) {
            LOG_DEBUG(QString("  ... and %1 more files").arg(selected.size() - 5));
        }
        // TODO: Implement actual file deletion with FileManager
        QMessageBox::information(this, tr("Delete"), tr("File deletion will be implemented soon.\nThis will integrate with FileManager and SafetyManager."));
    } else {
        LOG_INFO("User cancelled file deletion");
    }
}

void ResultsWindow::moveSelectedFiles()
{
    LOG_INFO("User clicked 'Move Selected Files' button");
    
    QList<DuplicateFile> selected = getSelectedFiles();
    if (selected.isEmpty()) {
        LOG_WARNING("No files selected for moving");
        return;
    }
    
    QString destination = QFileDialog::getExistingDirectory(this, tr("Select Destination Folder"));
    if (!destination.isEmpty()) {
        LOG_INFO(QString("User selected destination: %1").arg(destination));
        LOG_WARNING(QString("Would move %1 files (not yet implemented)").arg(selected.size()));
        for (int i = 0; i < qMin(5, selected.size()); ++i) {
            LOG_DEBUG(QString("  - Would move: %1 -> %2").arg(selected[i].filePath).arg(destination));
        }
        // TODO: Implement actual file moving with FileManager
        QMessageBox::information(this, tr("Move"), tr("File moving will be implemented soon.\nThis will integrate with FileManager."));
    } else {
        LOG_INFO("User cancelled file move operation");
    }
}

void ResultsWindow::ignoreSelectedFiles()
{
    LOG_INFO("User clicked 'Ignore Selected Files' button");
    QList<DuplicateFile> selected = getSelectedFiles();
    LOG_WARNING(QString("Ignore functionality not yet implemented (%1 files selected)").arg(selected.size()));
    // TODO: Implement ignore functionality - add to ignore list
    QMessageBox::information(this, tr("Ignore"), tr("Ignore functionality will be implemented soon.\nThis will add files to an ignore list."));
}

void ResultsWindow::previewSelectedFile()
{
    QList<QTreeWidgetItem*> selected = m_resultsTree->selectedItems();
    if (selected.isEmpty()) return;
    
    QTreeWidgetItem* item = selected.first();
    if (item->parent() == nullptr) return; // Group item
    
    QString filePath = item->data(0, Qt::UserRole).toString();
    qDebug() << "Preview file:" << filePath;
    
    // TODO: Implement file preview
    QMessageBox::information(this, tr("Preview"), tr("File preview will be implemented soon."));
}

void ResultsWindow::openFileLocation()
{
    LOG_INFO("User clicked 'Open File Location' button");
    
    QList<QTreeWidgetItem*> selected = m_resultsTree->selectedItems();
    if (selected.isEmpty()) {
        LOG_WARNING("No file selected");
        return;
    }
    
    QTreeWidgetItem* item = selected.first();
    if (item->parent() == nullptr) {
        LOG_WARNING("Group item selected, not a file");
        return;
    }
    
    QString filePath = item->data(0, Qt::UserRole).toString();
    QFileInfo fileInfo(filePath);
    QString dirPath = fileInfo.dir().absolutePath();
    
    LOG_INFO(QString("Opening file location: %1").arg(dirPath));
    QDesktopServices::openUrl(QUrl::fromLocalFile(dirPath));
}

void ResultsWindow::copyFilePath()
{
    LOG_INFO("User clicked 'Copy File Path' button");
    
    QList<QTreeWidgetItem*> selected = m_resultsTree->selectedItems();
    if (selected.isEmpty()) {
        LOG_WARNING("No file selected");
        return;
    }
    
    QTreeWidgetItem* item = selected.first();
    if (item->parent() == nullptr) {
        LOG_WARNING("Group item selected, not a file");
        return;
    }
    
    QString filePath = item->data(0, Qt::UserRole).toString();
    LOG_INFO(QString("Copying file path to clipboard: %1").arg(filePath));
    QApplication::clipboard()->setText(filePath);
    
    m_statusLabel->setText(tr("File path copied to clipboard"));
}

// Bulk operations
void ResultsWindow::performBulkDelete()
{
    QList<DuplicateFile> selected = getSelectedFiles();
    if (selected.isEmpty()) {
        QMessageBox::information(this, tr("No Selection"), tr("Please select files to delete."));
        return;
    }
    
    confirmBulkOperation(tr("delete"), static_cast<int>(selected.size()), getSelectedFilesSize());
}

void ResultsWindow::performBulkMove()
{
    QList<DuplicateFile> selected = getSelectedFiles();
    if (selected.isEmpty()) {
        QMessageBox::information(this, tr("No Selection"), tr("Please select files to move."));
        return;
    }
    
    QString destination = QFileDialog::getExistingDirectory(this, tr("Select Destination Folder"));
    if (!destination.isEmpty()) {
        confirmBulkOperation(tr("move to %1").arg(destination), static_cast<int>(selected.size()), getSelectedFilesSize());
    }
}

void ResultsWindow::confirmBulkOperation(const QString& operation, int fileCount, qint64 totalSize)
{
    QString message = tr("Are you sure you want to %1 %2 files (%3)?")
                     .arg(operation)
                     .arg(fileCount)
                     .arg(formatFileSize(totalSize));
    
    QMessageBox::StandardButton reply = QMessageBox::question(this,
                                                              tr("Confirm Bulk Operation"),
                                                              message,
                                                              QMessageBox::Yes | QMessageBox::No);
    
    if (reply == QMessageBox::Yes) {
        qDebug() << "Would" << operation << fileCount << "files";
        // TODO: Implement actual bulk operations
        QMessageBox::information(this, tr("Bulk Operation"), tr("Bulk operations will be implemented soon."));
    }
}

// Filter and sort implementations
void ResultsWindow::onFilterChanged()
{
    LOG_INFO("User changed filter settings");
    LOG_DEBUG(QString("  - Type filter: %1").arg(m_typeFilter->currentText()));
    LOG_DEBUG(QString("  - Size filter: %1").arg(m_sizeFilter->currentText()));
    LOG_DEBUG(QString("  - Search text: %1").arg(m_searchFilter->text()));
    
    // Apply filters to the tree view
    applyFilters();
}

void ResultsWindow::onSortChanged()
{
    LOG_INFO(QString("User changed sort order to: %1").arg(m_sortCombo->currentText()));
    
    // Apply sorting to the tree view
    applySorting();
}

void ResultsWindow::onGroupExpanded(QTreeWidgetItem* item)
{
    if (!item) return;
    
    LOG_DEBUG(QString("Group expanded: %1").arg(item->text(0)));
    
    // Update the group's expanded state in our data
    QString groupId = item->data(0, Qt::UserRole + 3).toString();
    for (auto& group : m_currentResults.duplicateGroups) {
        if (group.groupId == groupId) {
            group.isExpanded = true;
            break;
        }
    }
    
    // Load file details for this group if not already loaded
    // This could trigger thumbnail loading, etc.
}

void ResultsWindow::onGroupCollapsed(QTreeWidgetItem* item)
{
    if (!item) return;
    
    LOG_DEBUG(QString("Group collapsed: %1").arg(item->text(0)));
    
    // Update the group's expanded state in our data
    QString groupId = item->data(0, Qt::UserRole + 3).toString();
    for (auto& group : m_currentResults.duplicateGroups) {
        if (group.groupId == groupId) {
            group.isExpanded = false;
            break;
        }
    }
}

void ResultsWindow::onGroupSelectionChanged()
{
    LOG_DEBUG("Group selection changed");
    
    // Update the details panel with the selected group's information
    QList<QTreeWidgetItem*> selected = m_resultsTree->selectedItems();
    if (!selected.isEmpty()) {
        QTreeWidgetItem* item = selected.first();
        
        // If it's a group item (no parent)
        if (!item->parent()) {
            QString groupId = item->data(0, Qt::UserRole + 3).toString();
            LOG_DEBUG(QString("Selected group: %1").arg(groupId));
            
            // Update details panel with group information
            // This could show group statistics, recommendations, etc.
        }
    }
    
    // Update selection summary
    updateSelectionSummary();
}

// Event handlers
void ResultsWindow::closeEvent(QCloseEvent* event)
{
    emit windowClosed();
    QMainWindow::closeEvent(event);
}

void ResultsWindow::showEvent(QShowEvent* event)
{
    QMainWindow::showEvent(event);
    updateStatusBar();
}

// Clear results
void ResultsWindow::clearResults()
{
    m_currentResults = ScanResults();
    m_resultsTree->clear();
    m_summaryLabel->setText(tr("No results to display"));
    updateStatusBar();
}

// Update progress
void ResultsWindow::updateProgress(const QString& operation, int percentage)
{
    m_statusLabel->setText(operation);
    m_progressBar->setValue(percentage);
    m_progressBar->setVisible(percentage > 0 && percentage < 100);
}

// Filter and sort helper implementations
void ResultsWindow::applyFilters()
{
    LOG_DEBUG("Applying filters to results tree");
    
    if (!m_resultsTree) {
        return;
    }
    
    QString searchText = m_searchFilter ? m_searchFilter->text().toLower() : "";
    int sizeFilterIndex = m_sizeFilter ? m_sizeFilter->currentIndex() : 0;
    int typeFilterIndex = m_typeFilter ? m_typeFilter->currentIndex() : 0;
    
    int visibleGroups = 0;
    int hiddenGroups = 0;
    
    // Iterate through all top-level items (groups)
    for (int i = 0; i < m_resultsTree->topLevelItemCount(); ++i) {
        QTreeWidgetItem* groupItem = m_resultsTree->topLevelItem(i);
        bool shouldShow = true;
        
        // Apply search filter
        if (!searchText.isEmpty()) {
            QString groupText = groupItem->text(0).toLower();
            bool matchFound = groupText.contains(searchText);
            
            // Also check child items
            if (!matchFound) {
                for (int j = 0; j < groupItem->childCount(); ++j) {
                    QString childText = groupItem->child(j)->text(0).toLower();
                    if (childText.contains(searchText)) {
                        matchFound = true;
                        break;
                    }
                }
            }
            
            shouldShow = matchFound;
        }
        
        // Apply size filter
        if (shouldShow && sizeFilterIndex > 0) {
            // Size filter logic based on index
            // 0 = All sizes, 1 = < 1MB, 2 = 1-10MB, 3 = 10-100MB, 4 = > 100MB
            qint64 groupSize = groupItem->data(0, Qt::UserRole + 1).toLongLong();
            
            switch (sizeFilterIndex) {
                case 1: shouldShow = (groupSize < 1024 * 1024); break;
                case 2: shouldShow = (groupSize >= 1024 * 1024 && groupSize < 10 * 1024 * 1024); break;
                case 3: shouldShow = (groupSize >= 10 * 1024 * 1024 && groupSize < 100 * 1024 * 1024); break;
                case 4: shouldShow = (groupSize >= 100 * 1024 * 1024); break;
            }
        }
        
        // Apply type filter
        if (shouldShow && typeFilterIndex > 0) {
            QString groupType = groupItem->data(0, Qt::UserRole + 2).toString();
            QString filterType = m_typeFilter->currentText().toLower();
            shouldShow = groupType.contains(filterType, Qt::CaseInsensitive);
        }
        
        groupItem->setHidden(!shouldShow);
        if (shouldShow) {
            visibleGroups++;
        } else {
            hiddenGroups++;
        }
    }
    
    LOG_DEBUG(QString("Filter results: %1 visible, %2 hidden").arg(visibleGroups).arg(hiddenGroups));
}

void ResultsWindow::applySorting()
{
    LOG_DEBUG("Applying sorting to results tree");
    
    if (!m_resultsTree || !m_sortCombo) {
        return;
    }
    
    int sortIndex = m_sortCombo->currentIndex();
    
    // Sort options:
    // 0 = Size (largest first)
    // 1 = Size (smallest first)
    // 2 = Name (A-Z)
    // 3 = Name (Z-A)
    // 4 = Count (most duplicates first)
    
    int column = 0;  // Default to first column
    Qt::SortOrder order = Qt::DescendingOrder;
    
    switch (sortIndex) {
        case 0: // Size (largest first)
            column = 1;  // Size column
            order = Qt::DescendingOrder;
            break;
        case 1: // Size (smallest first)
            column = 1;
            order = Qt::AscendingOrder;
            break;
        case 2: // Name (A-Z)
            column = 0;  // Name column
            order = Qt::AscendingOrder;
            break;
        case 3: // Name (Z-A)
            column = 0;
            order = Qt::DescendingOrder;
            break;
        case 4: // Count (most duplicates first)
            column = 2;  // Count column (if exists)
            order = Qt::DescendingOrder;
            break;
    }
    
    m_resultsTree->sortItems(column, order);
    LOG_DEBUG(QString("Sorted by column %1, order %2").arg(column).arg(order == Qt::AscendingOrder ? "ascending" : "descending"));
}

bool ResultsWindow::matchesCurrentFilters(const DuplicateGroup& group) const
{
    // This method can be used when adding new groups dynamically
    Q_UNUSED(group);
    return true;  // Placeholder for now
}
