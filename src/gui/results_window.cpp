#include "results_window.h"
#include "app_config.h"
#include "duplicate_detector.h"
#include "file_manager.h"
#include "thumbnail_cache.h"
#include "thumbnail_delegate.h"
#include "core/logger.h"
#include <QtWidgets/QApplication>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QScrollBar>
#include <QtGui/QDesktopServices>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QInputDialog>
#include <QtGui/QClipboard>
#include <QtGui/QShortcut>
#include <QtGui/QPixmap>
#include <QtGui/QPainter>
#include <QtGui/QImageReader>
#include <QtCore/QUrl>
#include <QtCore/QDir>
#include <QtCore/QMimeDatabase>
#include <QtCore/QStandardPaths>
#include <QtCore/QDebug>

// Undefine old AppConfig logging macros to use new Logger system
#undef LOG_DEBUG
#undef LOG_INFO
#undef LOG_WARNING
#undef LOG_ERROR
#undef LOG_FILE

// Redefine for ResultsWindow to use UI category
#define LOG_DEBUG(msg) Logger::instance()->debug(LogCategories::UI, msg)
#define LOG_INFO(msg) Logger::instance()->info(LogCategories::UI, msg)
#define LOG_WARNING(msg) Logger::instance()->warning(LogCategories::UI, msg)
#define LOG_ERROR(msg) Logger::instance()->error(LogCategories::UI, msg)

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
    , m_fileManager(nullptr)
    , m_thumbnailCache(new ThumbnailCache(this))
    , m_thumbnailDelegate(nullptr)
    , m_selectionHistory(new SelectionHistoryManager(this))
    , m_operationQueue(new FileOperationQueue(this))  // Task 30
    , m_progressDialog(new FileOperationProgressDialog(this))  // Task 30
    , m_groupingDialog(new GroupingOptionsDialog(this))  // Task 13
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
    setupOperationQueue();  // Task 30
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
    m_refreshButton->setToolTip(tr("Refresh results display"));
    m_exportButton = new QPushButton(tr("📤 Export"), this);
    m_exportButton->setToolTip(tr("Export results to CSV, JSON, or text file (Ctrl+S)"));
    m_settingsButton = new QPushButton(tr("⚙️ Settings"), this);
    m_settingsButton->setToolTip(tr("Open settings dialog"));
    
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
    
    // Task 13: Add grouping button
    m_groupingButton = new QPushButton(tr("Grouping..."), this);
    m_groupingButton->setToolTip(tr("Change how duplicate files are grouped"));
    
    m_filterLayout->addWidget(m_filterLabel);
    m_filterLayout->addWidget(m_searchFilter, 1);
    m_filterLayout->addWidget(m_sizeFilter);
    m_filterLayout->addWidget(m_typeFilter);
    m_filterLayout->addWidget(m_sortCombo);
    m_filterLayout->addWidget(m_groupingButton);
    m_filterLayout->addWidget(m_clearFiltersButton);
    
    // Selection panel
    m_selectionPanel = new QWidget(this);
    m_selectionLayout = new QHBoxLayout(m_selectionPanel);
    m_selectionLayout->setContentsMargins(0, 0, 0, 0);
    m_selectionLayout->setSpacing(8);
    
    m_selectAllCheckbox = new QCheckBox(tr("Select All"), this);
    m_selectAllCheckbox->setToolTip(tr("Select all duplicate files"));
    m_selectRecommendedButton = new QPushButton(tr("Select Recommended"), this);
    m_selectRecommendedButton->setToolTip(tr("Select files recommended for deletion (keeps newest/largest)"));
    m_selectByTypeButton = new QPushButton(tr("Select by Type"), this);
    m_selectByTypeButton->setToolTip(tr("Select files by type (images, documents, etc.)"));
    m_clearSelectionButton = new QPushButton(tr("Clear Selection"), this);
    m_clearSelectionButton->setToolTip(tr("Deselect all files"));
    
    // Selection History buttons (Task 17)
    m_undoButton = new QPushButton(tr("Undo"), this);
    m_undoButton->setToolTip(tr("Undo last selection change (Ctrl+Z)"));
    m_undoButton->setEnabled(false);
    m_redoButton = new QPushButton(tr("Redo"), this);
    m_redoButton->setToolTip(tr("Redo last undone selection change (Ctrl+Y)"));
    m_redoButton->setEnabled(false);
    m_invertSelectionButton = new QPushButton(tr("Invert"), this);
    m_invertSelectionButton->setToolTip(tr("Invert current selection (Ctrl+I)"));
    
    m_selectionSummaryLabel = new QLabel(tr("0 files selected"), this);
    
    m_selectionLayout->addWidget(m_selectAllCheckbox);
    m_selectionLayout->addWidget(m_selectRecommendedButton);
    m_selectionLayout->addWidget(m_selectByTypeButton);
    m_selectionLayout->addWidget(m_clearSelectionButton);
    
    // Add separator
    QFrame* separator = new QFrame(this);
    separator->setFrameShape(QFrame::VLine);
    separator->setFrameShadow(QFrame::Sunken);
    m_selectionLayout->addWidget(separator);
    
    // Add selection history buttons (Task 17)
    m_selectionLayout->addWidget(m_undoButton);
    m_selectionLayout->addWidget(m_redoButton);
    m_selectionLayout->addWidget(m_invertSelectionButton);
    
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
    
    // Set up thumbnail delegate
    m_thumbnailDelegate = new ThumbnailDelegate(m_thumbnailCache, this);
    m_resultsTree->setItemDelegateForColumn(0, m_thumbnailDelegate);
    
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
    m_deleteButton->setToolTip(tr("Delete selected file (backup created automatically)"));
    m_moveButton = new QPushButton(tr("📁 Move File"), this);
    m_moveButton->setToolTip(tr("Move selected file to another location"));
    m_ignoreButton = new QPushButton(tr("👁️ Ignore File"), this);
    m_ignoreButton->setToolTip(tr("Ignore this file in current results"));
    m_previewButton = new QPushButton(tr("👀 Preview"), this);
    m_previewButton->setToolTip(tr("Preview file content (images, text files)"));
    m_openLocationButton = new QPushButton(tr("📂 Open Location"), this);
    m_openLocationButton->setToolTip(tr("Open file location in file manager"));
    m_copyPathButton = new QPushButton(tr("📋 Copy Path"), this);
    m_copyPathButton->setToolTip(tr("Copy file path to clipboard"));
    
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
    m_bulkDeleteButton->setToolTip(tr("Delete all selected files (backups created automatically)"));
    m_bulkMoveButton = new QPushButton(tr("📁 Move Selected"), this);
    m_bulkMoveButton->setToolTip(tr("Move all selected files to another location"));
    m_bulkIgnoreButton = new QPushButton(tr("👁️ Ignore Selected"), this);
    m_bulkIgnoreButton->setToolTip(tr("Ignore all selected files in current results"));
    
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
    
    // Task 13: Grouping button
    connect(m_groupingButton, &QPushButton::clicked, this, &ResultsWindow::showGroupingOptions);
    
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
    connect(m_bulkIgnoreButton, &QPushButton::clicked, this, &ResultsWindow::ignoreSelectedFiles);
    
    // Thumbnail timer
    connect(m_thumbnailTimer, &QTimer::timeout, this, [this]() {
        // Generate thumbnails in batches
        qDebug() << "Generating thumbnails...";
    });
    
    // Thumbnail cache signals
    connect(m_thumbnailCache, &ThumbnailCache::thumbnailReady,
            this, [this](const QString& filePath, const QPixmap& thumbnail) {
                Q_UNUSED(filePath);
                Q_UNUSED(thumbnail);
                // Force repaint of the tree to show the new thumbnail
                m_resultsTree->viewport()->update();
            });
    
    // Preload thumbnails when tree is scrolled
    connect(m_resultsTree->verticalScrollBar(), &QScrollBar::valueChanged,
            this, &ResultsWindow::preloadVisibleThumbnails);
    
    // Selection History connections (Task 17)
    connect(m_undoButton, &QPushButton::clicked, this, &ResultsWindow::onUndoRequested);
    connect(m_redoButton, &QPushButton::clicked, this, &ResultsWindow::onRedoRequested);
    connect(m_invertSelectionButton, &QPushButton::clicked, this, &ResultsWindow::onInvertSelection);
    
    // Connect selection history manager signals
    connect(m_selectionHistory, &SelectionHistoryManager::undoAvailabilityChanged,
            m_undoButton, &QPushButton::setEnabled);
    connect(m_selectionHistory, &SelectionHistoryManager::redoAvailabilityChanged,
            m_redoButton, &QPushButton::setEnabled);
    
    // Task 13: Connect grouping dialog
    connect(m_groupingDialog, &GroupingOptionsDialog::groupingChanged,
            this, &ResultsWindow::applyGrouping);
    
    // T19: Keyboard shortcuts for results window
    setupKeyboardShortcuts();
}

void ResultsWindow::setupKeyboardShortcuts()
{
    // T19: Keyboard shortcuts for results window
    
    // Ctrl+A - Select All
    QShortcut* selectAllShortcut = new QShortcut(QKeySequence::SelectAll, this);
    connect(selectAllShortcut, &QShortcut::activated, this, [this]() {
        if (m_selectAllCheckbox) {
            m_selectAllCheckbox->setChecked(true);
        }
    });
    
    // Delete - Delete selected files
    QShortcut* deleteShortcut = new QShortcut(QKeySequence::Delete, this);
    connect(deleteShortcut, &QShortcut::activated, this, &ResultsWindow::deleteSelectedFiles);
    
    // Ctrl+S - Export results
    QShortcut* exportShortcut = new QShortcut(QKeySequence::Save, this);
    connect(exportShortcut, &QShortcut::activated, this, &ResultsWindow::exportResults);
    
    // Ctrl+F - Focus search filter
    QShortcut* findShortcut = new QShortcut(QKeySequence::Find, this);
    connect(findShortcut, &QShortcut::activated, this, [this]() {
        if (m_searchFilter) {
            m_searchFilter->setFocus();
            m_searchFilter->selectAll();
        }
    });
    
    // Ctrl+R - Refresh results
    QShortcut* refreshShortcut = new QShortcut(QKeySequence::Refresh, this);
    connect(refreshShortcut, &QShortcut::activated, this, &ResultsWindow::refreshResults);
    
    // F5 - Refresh (alternative)
    QShortcut* f5Shortcut = new QShortcut(QKeySequence(Qt::Key_F5), this);
    connect(f5Shortcut, &QShortcut::activated, this, &ResultsWindow::refreshResults);
    
    // Ctrl+Z - Undo selection
    QShortcut* undoShortcut = new QShortcut(QKeySequence::Undo, this);
    connect(undoShortcut, &QShortcut::activated, this, &ResultsWindow::onUndoRequested);
    
    // Ctrl+Y - Redo selection
    QShortcut* redoShortcut = new QShortcut(QKeySequence::Redo, this);
    connect(redoShortcut, &QShortcut::activated, this, &ResultsWindow::onRedoRequested);
    
    // Ctrl+I - Invert selection
    QShortcut* invertShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_I), this);
    connect(invertShortcut, &QShortcut::activated, this, &ResultsWindow::onInvertSelection);
    
    // Ctrl+D - Clear selection
    QShortcut* clearShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_D), this);
    connect(clearShortcut, &QShortcut::activated, this, &ResultsWindow::selectNoneFiles);
    
    // Space - Preview selected file
    QShortcut* previewShortcut = new QShortcut(QKeySequence(Qt::Key_Space), this);
    connect(previewShortcut, &QShortcut::activated, this, &ResultsWindow::previewSelectedFile);
    
    // Enter - Open file location
    QShortcut* openShortcut = new QShortcut(QKeySequence(Qt::Key_Return), this);
    connect(openShortcut, &QShortcut::activated, this, &ResultsWindow::openFileLocation);
    
    // Ctrl+C - Copy file path
    QShortcut* copyShortcut = new QShortcut(QKeySequence::Copy, this);
    connect(copyShortcut, &QShortcut::activated, this, &ResultsWindow::copyFilePath);
    
    // Escape - Clear filters or close window
    QShortcut* escapeShortcut = new QShortcut(QKeySequence(Qt::Key_Escape), this);
    connect(escapeShortcut, &QShortcut::activated, this, [this]() {
        if (m_searchFilter && !m_searchFilter->text().isEmpty()) {
            m_searchFilter->clear();
        } else {
            // Close window if no filters to clear
            close();
        }
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

void ResultsWindow::displayDuplicateGroups(const QList<DuplicateDetector::DuplicateGroup>& groups)
{
    qDebug() << "ResultsWindow: Displaying" << groups.size() << "duplicate groups";
    
    // Convert DuplicateDetector groups to ResultsWindow format
    ScanResults results;
    results.duplicateGroups.clear();
    results.duplicateGroups.reserve(groups.size());
    
    for (const auto& detectorGroup : groups) {
        DuplicateGroup displayGroup;
        convertDetectorGroupToDisplayGroup(detectorGroup, displayGroup);
        results.duplicateGroups.append(displayGroup);
    }
    
    // Calculate totals
    results.calculateTotals();
    results.scanTime = QDateTime::currentDateTime();
    
    qDebug() << "ResultsWindow: Converted" << results.duplicateGroups.size() << "groups";
    qDebug() << "ResultsWindow: Total duplicates:" << results.totalDuplicatesFound;
    qDebug() << "ResultsWindow: Potential savings:" << formatFileSize(results.potentialSavings);
    
    // Display the results
    displayResults(results);
}

void ResultsWindow::setFileManager(FileManager* fileManager)
{
    m_fileManager = fileManager;
    qDebug() << "ResultsWindow: FileManager reference set";
}

void ResultsWindow::convertDetectorGroupToDisplayGroup(const DuplicateDetector::DuplicateGroup& source, DuplicateGroup& target)
{
    target.groupId = source.groupId;
    target.totalSize = source.totalSize;
    target.wastedSpace = source.wastedSpace;
    target.fileCount = source.fileCount;
    target.primaryFile = source.recommendedAction;
    target.isExpanded = false;
    target.hasSelection = false;
    
    target.files.clear();
    target.files.reserve(source.files.size());
    
    for (const auto& detectorFile : source.files) {
        DuplicateFile displayFile;
        displayFile.filePath = detectorFile.filePath;
        displayFile.fileName = detectorFile.fileName;
        displayFile.directory = detectorFile.directory;
        displayFile.fileSize = detectorFile.fileSize;
        displayFile.lastModified = detectorFile.lastModified;
        displayFile.created = detectorFile.lastModified; // Use lastModified as created if not available
        displayFile.hash = detectorFile.hash;
        displayFile.isSelected = false;
        displayFile.isMarkedForDeletion = false;
        
        // Extract file type from filename
        QFileInfo fileInfo(detectorFile.filePath);
        displayFile.fileType = fileInfo.suffix();
        
        target.files.append(displayFile);
    }
}

void ResultsWindow::updateStatisticsDisplay()
{
    if (!m_statisticsLabel) {
        return;
    }
    
    QString stats = tr("Groups: %1 | Files: %2 | Savings: %3")
        .arg(m_currentResults.duplicateGroups.size())
        .arg(m_currentResults.totalDuplicatesFound)
        .arg(formatFileSize(m_currentResults.potentialSavings));
    
    m_statisticsLabel->setText(stats);
}

void ResultsWindow::removeFilesFromDisplay(const QStringList& filePaths)
{
    qDebug() << "ResultsWindow: Removing" << filePaths.size() << "files from display";
    
    // Remove files from groups
    for (auto& group : m_currentResults.duplicateGroups) {
        for (const QString& path : filePaths) {
            group.files.removeIf([&path](const DuplicateFile& file) {
                return file.filePath == path;
            });
        }
        group.fileCount = static_cast<int>(group.files.size());
    }
    
    // Remove empty groups
    m_currentResults.duplicateGroups.removeIf([](const DuplicateGroup& group) {
        return group.fileCount <= 1; // Groups with 0 or 1 file are not duplicates
    });
    
    // Recalculate totals
    m_currentResults.calculateTotals();
    
    // Refresh display
    populateResultsTree();
    updateStatisticsDisplay();
    updateStatusBar();
    
    qDebug() << "ResultsWindow: Remaining groups:" << m_currentResults.duplicateGroups.size();
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
                                                   "duplicate_files_report.csv",
                                                   tr("CSV Files (*.csv);;JSON Files (*.json);;Text Files (*.txt)"));
    if (fileName.isEmpty()) {
        return;
    }
    
    qDebug() << "Exporting results to:" << fileName;
    
    // Determine format from file extension
    QString format = "csv";
    if (fileName.endsWith(".json", Qt::CaseInsensitive)) {
        format = "json";
    } else if (fileName.endsWith(".txt", Qt::CaseInsensitive)) {
        format = "txt";
    }
    
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, tr("Export Error"), 
                            tr("Failed to open file for writing: %1").arg(fileName));
        qWarning() << "Failed to open export file:" << fileName;
        return;
    }
    
    QTextStream out(&file);
    bool success = false;
    
    if (format == "csv") {
        success = exportToCSV(out);
    } else if (format == "json") {
        success = exportToJSON(out);
    } else {
        success = exportToText(out);
    }
    
    file.close();
    
    if (success) {
        QMessageBox::information(this, tr("Export Complete"), 
                               tr("Results exported successfully to:\n%1").arg(fileName));
        qDebug() << "Export completed successfully:" << fileName;
        emit resultsExported(fileName);
    } else {
        QMessageBox::warning(this, tr("Export Warning"), 
                           tr("Export completed with warnings. Check the file."));
        qWarning() << "Export completed with warnings:" << fileName;
    }
}

void ResultsWindow::selectAllDuplicates()
{
    LOG_INFO("User clicked 'Select All' button");
    
    // Record current state before changing selection (Task 17)
    recordSelectionState("Select all files");
    
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
    
    // Record current state before changing selection (Task 17)
    recordSelectionState("Clear all selections");
    
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
    if (selected.isEmpty()) {
        LOG_WARNING("No files selected for deletion");
        return;
    }
    
    // Calculate total size
    qint64 totalSize = 0;
    for (const auto& file : selected) {
        totalSize += file.fileSize;
    }
    
    // Show confirmation dialog with details
    QString message = tr("Are you sure you want to delete %1 file(s)?\n\nTotal size: %2\n\nBackups will be created automatically.")
                     .arg(selected.size())
                     .arg(formatFileSize(totalSize));
    
    QMessageBox::StandardButton reply = QMessageBox::question(this, 
                                                              tr("Confirm Delete"),
                                                              message,
                                                              QMessageBox::Yes | QMessageBox::No);
    
    if (reply != QMessageBox::Yes) {
        LOG_INFO("User cancelled file deletion");
        return;
    }
    
    LOG_INFO(QString("=== File Deletion Started ==="));
    LOG_INFO(QString("  - Files to delete: %1").arg(selected.size()));
    LOG_INFO(QString("  - Total size: %1").arg(formatFileSize(totalSize)));
    
    // Log files to be deleted
    for (int i = 0; i < qMin(5, selected.size()); ++i) {
        LOG_DEBUG(QString("  - Delete: %1").arg(selected[i].filePath));
    }
    if (selected.size() > 5) {
        LOG_DEBUG(QString("  ... and %1 more files").arg(selected.size() - 5));
    }
    
    // Check if FileManager is available
    if (!m_fileManager) {
        LOG_ERROR("FileManager not available - cannot delete files");
        QMessageBox::critical(this, tr("Error"), 
                            tr("File manager is not initialized. Cannot delete files."));
        return;
    }
    
    // Collect file paths
    QStringList filePaths;
    for (const auto& file : selected) {
        filePaths.append(file.filePath);
    }
    
    // Call FileManager to delete files
    QString operationId = m_fileManager->deleteFiles(filePaths, tr("User deleted duplicates from results window"));
    
    if (operationId.isEmpty()) {
        LOG_ERROR("Failed to start delete operation");
        QMessageBox::critical(this, tr("Error"), 
                            tr("Failed to start delete operation."));
        return;
    }
    
    LOG_INFO(QString("Delete operation started with ID: %1").arg(operationId));
    
    // Connect to operation completion signals
    connect(m_fileManager, &FileManager::operationCompleted, this,
            [this, filePaths](const FileManager::OperationResult& result) {
                if (result.success) {
                    LOG_INFO(QString("=== File Deletion Completed ==="));
                    LOG_INFO(QString("  - Files deleted: %1").arg(result.processedFiles.size()));
                    LOG_INFO(QString("  - Failed: %1").arg(result.failedFiles.size()));
                    
                    // Remove deleted files from display
                    removeFilesFromDisplay(result.processedFiles);
                    
                    // Update statistics
                    updateStatisticsDisplay();
                    
                    // Show success message
                    QString message = tr("Successfully deleted %1 file(s)").arg(result.processedFiles.size());
                    if (!result.failedFiles.isEmpty()) {
                        message += tr("\n\nFailed to delete %1 file(s)").arg(result.failedFiles.size());
                    }
                    QMessageBox::information(this, tr("Delete Complete"), message);
                } else {
                    LOG_ERROR(QString("Delete operation failed: %1").arg(result.errorMessage));
                    QMessageBox::critical(this, tr("Delete Failed"), 
                                        tr("Failed to delete files:\n%1").arg(result.errorMessage));
                }
            });
    
    connect(m_fileManager, &FileManager::operationError, this,
            [this](const QString& operationId, const QString& error) {
                LOG_ERROR(QString("Delete operation error [%1]: %2").arg(operationId).arg(error));
                QMessageBox::critical(this, tr("Delete Error"), 
                                    tr("An error occurred during deletion:\n%1").arg(error));
            });
}

void ResultsWindow::moveSelectedFiles()
{
    LOG_INFO("User clicked 'Move Selected Files' button");
    
    QList<DuplicateFile> selected = getSelectedFiles();
    if (selected.isEmpty()) {
        LOG_WARNING("No files selected for moving");
        QMessageBox::information(this, tr("Move Files"), 
                               tr("Please select files to move first."));
        return;
    }
    
    // Prompt user for destination folder
    QString destination = QFileDialog::getExistingDirectory(this, 
                                                           tr("Select Destination Folder"),
                                                           QDir::homePath(),
                                                           QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    
    if (destination.isEmpty()) {
        LOG_INFO("User cancelled file move operation");
        return;
    }
    
    // Validate destination is writable
    QFileInfo destInfo(destination);
    if (!destInfo.isWritable()) {
        LOG_ERROR(QString("Destination folder is not writable: %1").arg(destination));
        QMessageBox::critical(this, tr("Error"), 
                            tr("The selected destination folder is not writable."));
        return;
    }
    
    LOG_INFO(QString("=== File Move Started ==="));
    LOG_INFO(QString("  - Files to move: %1").arg(selected.size()));
    LOG_INFO(QString("  - Destination: %1").arg(destination));
    
    // Log files to be moved
    for (int i = 0; i < qMin(5, selected.size()); ++i) {
        LOG_DEBUG(QString("  - Move: %1 -> %2").arg(selected[i].filePath).arg(destination));
    }
    if (selected.size() > 5) {
        LOG_DEBUG(QString("  ... and %1 more files").arg(selected.size() - 5));
    }
    
    // Check if FileManager is available
    if (!m_fileManager) {
        LOG_ERROR("FileManager not available - cannot move files");
        QMessageBox::critical(this, tr("Error"), 
                            tr("File manager is not initialized. Cannot move files."));
        return;
    }
    
    // Collect file paths
    QStringList filePaths;
    for (const auto& file : selected) {
        filePaths.append(file.filePath);
    }
    
    // Call FileManager to move files
    QString operationId = m_fileManager->moveFiles(filePaths, destination);
    
    if (operationId.isEmpty()) {
        LOG_ERROR("Failed to start move operation");
        QMessageBox::critical(this, tr("Error"), 
                            tr("Failed to start move operation."));
        return;
    }
    
    LOG_INFO(QString("Move operation started with ID: %1").arg(operationId));
    
    // Connect to operation completion signals
    connect(m_fileManager, &FileManager::operationCompleted, this,
            [this, destination](const FileManager::OperationResult& result) {
                if (result.success) {
                    LOG_INFO(QString("=== File Move Completed ==="));
                    LOG_INFO(QString("  - Files moved: %1").arg(result.processedFiles.size()));
                    LOG_INFO(QString("  - Failed: %1").arg(result.failedFiles.size()));
                    LOG_INFO(QString("  - Skipped: %1").arg(result.skippedFiles.size()));
                    
                    // Remove moved files from display
                    removeFilesFromDisplay(result.processedFiles);
                    
                    // Update statistics
                    updateStatisticsDisplay();
                    
                    // Show success message
                    QString message = tr("Successfully moved %1 file(s) to:\n%2")
                                     .arg(result.processedFiles.size())
                                     .arg(destination);
                    if (!result.failedFiles.isEmpty()) {
                        message += tr("\n\nFailed to move %1 file(s)").arg(result.failedFiles.size());
                    }
                    if (!result.skippedFiles.isEmpty()) {
                        message += tr("\n\nSkipped %1 file(s) due to conflicts").arg(result.skippedFiles.size());
                    }
                    QMessageBox::information(this, tr("Move Complete"), message);
                } else {
                    LOG_ERROR(QString("Move operation failed: %1").arg(result.errorMessage));
                    QMessageBox::critical(this, tr("Move Failed"), 
                                        tr("Failed to move files:\n%1").arg(result.errorMessage));
                }
            });
    
    connect(m_fileManager, &FileManager::operationError, this,
            [this](const QString& operationId, const QString& error) {
                LOG_ERROR(QString("Move operation error [%1]: %2").arg(operationId).arg(error));
                QMessageBox::critical(this, tr("Move Error"), 
                                    tr("An error occurred during move:\n%1").arg(error));
            });
}

void ResultsWindow::ignoreSelectedFiles()
{
    LOG_INFO("User clicked 'Ignore Selected Files' button");
    QList<DuplicateFile> selected = getSelectedFiles();
    
    if (selected.isEmpty()) {
        QMessageBox::information(this, tr("Ignore"), tr("No files selected to ignore."));
        return;
    }
    
    LOG_INFO(QString("Ignoring %1 files").arg(selected.size()));
    
    // Remove ignored files from the display
    for (const DuplicateFile& file : selected) {
        // Find and remove the item from the tree
        QTreeWidgetItemIterator it(m_resultsTree);
        while (*it) {
            QTreeWidgetItem* item = *it;
            if (item->data(0, Qt::UserRole).toString() == file.filePath) {
                // Remove from parent or tree
                if (item->parent()) {
                    item->parent()->removeChild(item);
                } else {
                    int index = m_resultsTree->indexOfTopLevelItem(item);
                    if (index >= 0) {
                        m_resultsTree->takeTopLevelItem(index);
                    }
                }
                delete item;
                break;
            }
            ++it;
        }
    }
    
    // Update statistics
    updateStatisticsDisplay();
    
    QMessageBox::information(this, tr("Files Ignored"),
                           tr("%1 file(s) have been removed from the results.\n\nNote: Files are only hidden from current results, not permanently ignored.")
                           .arg(selected.size()));
}

void ResultsWindow::previewSelectedFile()
{
    QList<QTreeWidgetItem*> selected = m_resultsTree->selectedItems();
    if (selected.isEmpty()) {
        QMessageBox::information(this, tr("Preview"), tr("Please select a file to preview."));
        return;
    }
    
    QTreeWidgetItem* item = selected.first();
    if (item->parent() == nullptr) {
        QMessageBox::information(this, tr("Preview"), tr("Please select a file, not a group."));
        return;
    }
    
    QString filePath = item->data(0, Qt::UserRole).toString();
    qDebug() << "Preview file:" << filePath;
    
    if (!QFile::exists(filePath)) {
        QMessageBox::warning(this, tr("Preview Error"), 
                           tr("File not found: %1").arg(filePath));
        return;
    }
    
    QFileInfo fileInfo(filePath);
    
    // Check if file is an image
    if (isImageFile(filePath)) {
        previewImageFile(filePath);
    }
    // Check if file is a text file
    else if (isTextFile(filePath)) {
        previewTextFile(filePath);
    }
    // For other files, show file info
    else {
        showFileInfo(filePath);
    }
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
    
    // Confirm the operation
    QString message = tr("Are you sure you want to delete %1 files (%2)?")
                     .arg(selected.size())
                     .arg(formatFileSize(getSelectedFilesSize()));
    
    QMessageBox::StandardButton reply = QMessageBox::question(this,
                                                              tr("Confirm Delete Operation"),
                                                              message,
                                                              QMessageBox::Yes | QMessageBox::No);
    
    if (reply == QMessageBox::Yes) {
        // Convert to file paths
        QStringList filePaths;
        for (const DuplicateFile& file : selected) {
            filePaths.append(file.filePath);
        }
        
        // Queue the delete operation (Task 30)
        if (m_operationQueue) {
            QString operationId = m_operationQueue->queueDeleteOperation(filePaths);
            if (!operationId.isEmpty()) {
                m_statusLabel->setText(tr("Delete operation queued..."));
            } else {
                m_statusLabel->setText(tr("Failed to queue delete operation"));
            }
        }
    }
}

void ResultsWindow::performBulkMove()
{
    QList<DuplicateFile> selected = getSelectedFiles();
    if (selected.isEmpty()) {
        QMessageBox::information(this, tr("No Selection"), tr("Please select files to move."));
        return;
    }
    
    QString destination = QFileDialog::getExistingDirectory(this, tr("Select Destination Folder"));
    if (destination.isEmpty()) {
        return;
    }
    
    // Confirm the operation
    QString message = tr("Are you sure you want to move %1 files (%2) to %3?")
                     .arg(selected.size())
                     .arg(formatFileSize(getSelectedFilesSize()))
                     .arg(destination);
    
    QMessageBox::StandardButton reply = QMessageBox::question(this,
                                                              tr("Confirm Move Operation"),
                                                              message,
                                                              QMessageBox::Yes | QMessageBox::No);
    
    if (reply == QMessageBox::Yes) {
        // Convert to file paths
        QStringList filePaths;
        for (const DuplicateFile& file : selected) {
            filePaths.append(file.filePath);
        }
        
        // Queue the move operation (Task 30)
        if (m_operationQueue) {
            QString operationId = m_operationQueue->queueMoveOperation(filePaths, destination);
            if (!operationId.isEmpty()) {
                m_statusLabel->setText(tr("Move operation queued..."));
            } else {
                m_statusLabel->setText(tr("Failed to queue move operation"));
            }
        }
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
        LOG_INFO(QString("User confirmed bulk %1 operation for %2 files").arg(operation).arg(fileCount));
        
        // Get selected files
        QList<DuplicateFile> selected = getSelectedFiles();
        QStringList filePaths;
        for (const DuplicateFile& file : selected) {
            filePaths.append(file.filePath);
        }
        
        // Perform the operation based on type
        if (operation == "delete") {
            deleteSelectedFiles();
        } else if (operation == "move") {
            moveSelectedFiles();
        }
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


// Progress dialog methods
void ResultsWindow::showProgressDialog(const QString& title)
{
    qDebug() << "ResultsWindow: Show progress dialog:" << title;
    
    if (m_progressBar) {
        m_progressBar->setVisible(true);
        m_progressBar->setValue(0);
    }
    
    if (m_statusLabel) {
        m_statusLabel->setText(title);
    }
}

void ResultsWindow::hideProgressDialog()
{
    qDebug() << "ResultsWindow: Hide progress dialog";
    
    if (m_progressBar) {
        m_progressBar->setVisible(false);
    }
    
    if (m_statusLabel) {
        m_statusLabel->setText(tr("Ready"));
    }
}

// Sorting and filtering methods
void ResultsWindow::sortResults()
{
    qDebug() << "ResultsWindow: Sort results";
    
    if (!m_sortCombo) {
        return;
    }
    
    QString sortBy = m_sortCombo->currentText();
    qDebug() << "Sorting by:" << sortBy;
    
    // Sort duplicate groups based on selected criteria
    if (sortBy.contains("Size")) {
        std::sort(m_currentResults.duplicateGroups.begin(), 
                  m_currentResults.duplicateGroups.end(),
                  [](const DuplicateGroup& a, const DuplicateGroup& b) {
                      return a.wastedSpace > b.wastedSpace;
                  });
    } else if (sortBy.contains("Count")) {
        std::sort(m_currentResults.duplicateGroups.begin(),
                  m_currentResults.duplicateGroups.end(),
                  [](const DuplicateGroup& a, const DuplicateGroup& b) {
                      return a.fileCount > b.fileCount;
                  });
    } else if (sortBy.contains("Name")) {
        std::sort(m_currentResults.duplicateGroups.begin(),
                  m_currentResults.duplicateGroups.end(),
                  [](const DuplicateGroup& a, const DuplicateGroup& b) {
                      return a.files.first().fileName < b.files.first().fileName;
                  });
    }
    
    // Refresh display
    populateResultsTree();
}

void ResultsWindow::filterResults()
{
    qDebug() << "ResultsWindow: Filter results";
    
    // Apply current filter settings and refresh display
    populateResultsTree();
}

// Export helper methods

bool ResultsWindow::exportToCSV(QTextStream& out)
{
    qDebug() << "Exporting to CSV format";
    
    // Write CSV header
    out << "Group ID,File Path,File Name,Directory,Size (bytes),Size (formatted),Last Modified,Hash,Recommended Action\n";
    
    // Write data for each duplicate group
    for (const DuplicateGroup& group : m_currentResults.duplicateGroups) {
        for (const DuplicateFile& file : group.files) {
            // Escape CSV fields that contain commas or quotes
            auto escapeCSV = [](const QString& field) -> QString {
                if (field.contains(',') || field.contains('"') || field.contains('\n')) {
                    QString escaped = field;
                    escaped.replace("\"", "\"\"");
                    return "\"" + escaped + "\"";
                }
                return field;
            };
            
            out << escapeCSV(group.groupId) << ","
                << escapeCSV(file.filePath) << ","
                << escapeCSV(file.fileName) << ","
                << escapeCSV(file.directory) << ","
                << file.fileSize << ","
                << escapeCSV(formatFileSize(file.fileSize)) << ","
                << escapeCSV(file.lastModified.toString(Qt::ISODate)) << ","
                << escapeCSV(file.hash) << ","
                << escapeCSV(group.primaryFile == file.filePath ? "Keep" : "Consider Deleting")
                << "\n";
        }
    }
    
    qDebug() << "CSV export completed";
    return true;
}

bool ResultsWindow::exportToJSON(QTextStream& out)
{
    qDebug() << "Exporting to JSON format";
    
    // Start JSON document
    out << "{\n";
    out << "  \"exportDate\": \"" << QDateTime::currentDateTime().toString(Qt::ISODate) << "\",\n";
    out << "  \"totalGroups\": " << m_currentResults.duplicateGroups.size() << ",\n";
    out << "  \"totalDuplicates\": " << m_currentResults.totalDuplicatesFound << ",\n";
    out << "  \"totalSpaceWasted\": " << m_currentResults.totalSpaceWasted << ",\n";
    out << "  \"potentialSavings\": " << m_currentResults.potentialSavings << ",\n";
    out << "  \"duplicateGroups\": [\n";
    
    // Write each duplicate group
    for (int g = 0; g < m_currentResults.duplicateGroups.size(); ++g) {
        const DuplicateGroup& group = m_currentResults.duplicateGroups[g];
        
        out << "    {\n";
        out << "      \"groupId\": \"" << group.groupId << "\",\n";
        out << "      \"fileCount\": " << group.fileCount << ",\n";
        out << "      \"fileSize\": " << group.totalSize / group.fileCount << ",\n";
        out << "      \"totalSize\": " << group.totalSize << ",\n";
        out << "      \"wastedSpace\": " << group.wastedSpace << ",\n";
        
        QString escapedPrimaryFile = group.primaryFile;
        escapedPrimaryFile.replace("\"", "\\\"");
        out << "      \"primaryFile\": \"" << escapedPrimaryFile << "\",\n";
        out << "      \"files\": [\n";
        
        // Write each file in the group
        for (int f = 0; f < group.files.size(); ++f) {
            const DuplicateFile& file = group.files[f];
            
            QString escapedFilePath = file.filePath;
            escapedFilePath.replace("\"", "\\\"");
            QString escapedFileName = file.fileName;
            escapedFileName.replace("\"", "\\\"");
            QString escapedDirectory = file.directory;
            escapedDirectory.replace("\"", "\\\"");
            
            out << "        {\n";
            out << "          \"filePath\": \"" << escapedFilePath << "\",\n";
            out << "          \"fileName\": \"" << escapedFileName << "\",\n";
            out << "          \"directory\": \"" << escapedDirectory << "\",\n";
            out << "          \"fileSize\": " << file.fileSize << ",\n";
            out << "          \"lastModified\": \"" << file.lastModified.toString(Qt::ISODate) << "\",\n";
            out << "          \"hash\": \"" << file.hash << "\",\n";
            out << "          \"recommendedAction\": \"" << (group.primaryFile == file.filePath ? "Keep" : "Consider Deleting") << "\"\n";
            out << "        }";
            
            if (f < group.files.size() - 1) {
                out << ",";
            }
            out << "\n";
        }
        
        out << "      ]\n";
        out << "    }";
        
        if (g < m_currentResults.duplicateGroups.size() - 1) {
            out << ",";
        }
        out << "\n";
    }
    
    out << "  ]\n";
    out << "}\n";
    
    qDebug() << "JSON export completed";
    return true;
}

bool ResultsWindow::exportToText(QTextStream& out)
{
    qDebug() << "Exporting to text format";
    
    // Write header
    out << "===========================================\n";
    out << "Duplicate Files Report\n";
    out << "===========================================\n";
    out << "Generated: " << QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss") << "\n";
    out << "\n";
    
    // Write summary
    out << "Summary:\n";
    out << "--------\n";
    out << "Total Duplicate Groups: " << m_currentResults.duplicateGroups.size() << "\n";
    out << "Total Duplicate Files: " << m_currentResults.totalDuplicatesFound << "\n";
    out << "Total Space Wasted: " << formatFileSize(m_currentResults.totalSpaceWasted) << "\n";
    out << "Potential Savings: " << formatFileSize(m_currentResults.potentialSavings) << "\n";
    out << "\n";
    
    // Write detailed results
    out << "Detailed Results:\n";
    out << "=================\n";
    out << "\n";
    
    int groupNum = 1;
    for (const DuplicateGroup& group : m_currentResults.duplicateGroups) {
        out << "Group " << groupNum << " (ID: " << group.groupId << ")\n";
        out << "  File Count: " << group.fileCount << "\n";
        out << "  File Size: " << formatFileSize(group.totalSize / group.fileCount) << "\n";
        out << "  Total Size: " << formatFileSize(group.totalSize) << "\n";
        out << "  Wasted Space: " << formatFileSize(group.wastedSpace) << "\n";
        out << "  Primary File (Recommended to Keep): " << group.primaryFile << "\n";
        out << "  Files:\n";
        
        for (const DuplicateFile& file : group.files) {
            out << "    - " << file.filePath << "\n";
            out << "      Last Modified: " << file.lastModified.toString("yyyy-MM-dd hh:mm:ss") << "\n";
            out << "      Action: " << (group.primaryFile == file.filePath ? "KEEP" : "Consider Deleting") << "\n";
        }
        
        out << "\n";
        groupNum++;
    }
    
    out << "===========================================\n";
    out << "End of Report\n";
    out << "===========================================\n";
    
    qDebug() << "Text export completed";
    return true;
}

// Preview helper methods

void ResultsWindow::previewImageFile(const QString& filePath)
{
    qDebug() << "Previewing image file:" << filePath;
    
    QPixmap pixmap(filePath);
    if (pixmap.isNull()) {
        QMessageBox::warning(this, tr("Preview Error"), 
                           tr("Failed to load image: %1").arg(filePath));
        return;
    }
    
    // Create a dialog to show the image
    QDialog* previewDialog = new QDialog(this);
    previewDialog->setWindowTitle(tr("Image Preview - %1").arg(QFileInfo(filePath).fileName()));
    previewDialog->resize(800, 600);
    
    QVBoxLayout* layout = new QVBoxLayout(previewDialog);
    
    // Add image label with scroll area
    QScrollArea* scrollArea = new QScrollArea(previewDialog);
    QLabel* imageLabel = new QLabel();
    
    // Scale image if too large
    QPixmap scaledPixmap = pixmap;
    if (pixmap.width() > 1200 || pixmap.height() > 900) {
        scaledPixmap = pixmap.scaled(1200, 900, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }
    
    imageLabel->setPixmap(scaledPixmap);
    imageLabel->setAlignment(Qt::AlignCenter);
    scrollArea->setWidget(imageLabel);
    scrollArea->setWidgetResizable(true);
    
    layout->addWidget(scrollArea);
    
    // Add file info
    QLabel* infoLabel = new QLabel(tr("Size: %1 | Dimensions: %2x%3")
                                   .arg(formatFileSize(QFileInfo(filePath).size()))
                                   .arg(pixmap.width())
                                   .arg(pixmap.height()));
    infoLabel->setAlignment(Qt::AlignCenter);
    layout->addWidget(infoLabel);
    
    // Add close button
    QPushButton* closeButton = new QPushButton(tr("Close"));
    connect(closeButton, &QPushButton::clicked, previewDialog, &QDialog::accept);
    layout->addWidget(closeButton);
    
    previewDialog->exec();
    delete previewDialog;
}

void ResultsWindow::previewTextFile(const QString& filePath)
{
    qDebug() << "Previewing text file:" << filePath;
    
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::warning(this, tr("Preview Error"), 
                           tr("Failed to open file: %1").arg(filePath));
        return;
    }
    
    // Read first 1000 lines or 1MB, whichever comes first
    QTextStream in(&file);
    QString content;
    int lineCount = 0;
    const int maxLines = 1000;
    const qint64 maxBytes = 1024 * 1024; // 1MB
    
    while (!in.atEnd() && lineCount < maxLines && content.size() < maxBytes) {
        content += in.readLine() + "\n";
        lineCount++;
    }
    
    bool truncated = !in.atEnd();
    file.close();
    
    // Create a dialog to show the text
    QDialog* previewDialog = new QDialog(this);
    previewDialog->setWindowTitle(tr("Text Preview - %1").arg(QFileInfo(filePath).fileName()));
    previewDialog->resize(800, 600);
    
    QVBoxLayout* layout = new QVBoxLayout(previewDialog);
    
    // Add text edit
    QTextEdit* textEdit = new QTextEdit();
    textEdit->setPlainText(content);
    textEdit->setReadOnly(true);
    textEdit->setFont(QFont("Monospace", 10));
    layout->addWidget(textEdit);
    
    // Add info label
    QString infoText = tr("Size: %1 | Lines shown: %2")
                      .arg(formatFileSize(QFileInfo(filePath).size()))
                      .arg(lineCount);
    if (truncated) {
        infoText += tr(" (truncated)");
    }
    
    QLabel* infoLabel = new QLabel(infoText);
    infoLabel->setAlignment(Qt::AlignCenter);
    layout->addWidget(infoLabel);
    
    // Add close button
    QPushButton* closeButton = new QPushButton(tr("Close"));
    connect(closeButton, &QPushButton::clicked, previewDialog, &QDialog::accept);
    layout->addWidget(closeButton);
    
    previewDialog->exec();
    delete previewDialog;
}

void ResultsWindow::showFileInfo(const QString& filePath)
{
    qDebug() << "Showing file info:" << filePath;
    
    QFileInfo fileInfo(filePath);
    
    QString info;
    info += tr("File Information\n");
    info += tr("================\n\n");
    info += tr("Name: %1\n").arg(fileInfo.fileName());
    info += tr("Path: %1\n").arg(fileInfo.absolutePath());
    info += tr("Size: %1 (%2 bytes)\n").arg(formatFileSize(fileInfo.size())).arg(fileInfo.size());
    info += tr("Type: %1\n").arg(fileInfo.suffix().toUpper());
    info += tr("Created: %1\n").arg(fileInfo.birthTime().toString("yyyy-MM-dd hh:mm:ss"));
    info += tr("Modified: %1\n").arg(fileInfo.lastModified().toString("yyyy-MM-dd hh:mm:ss"));
    info += tr("Accessed: %1\n").arg(fileInfo.lastRead().toString("yyyy-MM-dd hh:mm:ss"));
    info += tr("\nPermissions: ");
    
    if (fileInfo.isReadable()) info += tr("Read ");
    if (fileInfo.isWritable()) info += tr("Write ");
    if (fileInfo.isExecutable()) info += tr("Execute");
    
    info += tr("\n\nPreview not available for this file type.");
    
    QMessageBox::information(this, tr("File Information"), info);
}

bool ResultsWindow::isTextFile(const QString& filePath) const
{
    QFileInfo fileInfo(filePath);
    QString suffix = fileInfo.suffix().toLower();
    
    // Common text file extensions
    QStringList textExtensions = {
        "txt", "log", "md", "markdown", "rst",
        "c", "cpp", "h", "hpp", "cc", "cxx",
        "java", "py", "js", "ts", "html", "htm", "css",
        "xml", "json", "yaml", "yml", "toml", "ini", "cfg", "conf",
        "sh", "bash", "bat", "cmd", "ps1",
        "sql", "csv", "tsv"
    };
    
    return textExtensions.contains(suffix);
}

// Thumbnail support methods

void ResultsWindow::enableThumbnails(bool enable)
{
    if (m_thumbnailDelegate) {
        m_thumbnailDelegate->setThumbnailsEnabled(enable);
        m_resultsTree->viewport()->update();
        
        if (enable) {
            preloadVisibleThumbnails();
        }
        
        LOG_INFO(QString("Thumbnails %1").arg(enable ? "enabled" : "disabled"));
    }
}

void ResultsWindow::setThumbnailSize(int size)
{
    if (m_thumbnailDelegate) {
        m_thumbnailDelegate->setThumbnailSize(size);
        m_resultsTree->viewport()->update();
        
        // Reload thumbnails with new size
        if (m_thumbnailDelegate->thumbnailsEnabled()) {
            m_thumbnailCache->clearCache();
            preloadVisibleThumbnails();
        }
        
        LOG_INFO(QString("Thumbnail size set to %1").arg(size));
    }
}

void ResultsWindow::preloadVisibleThumbnails()
{
    if (!m_thumbnailDelegate || !m_thumbnailDelegate->thumbnailsEnabled()) {
        return;
    }
    
    // Get visible items in the tree
    QStringList visibleFilePaths;
    
    // Iterate through top-level items (groups)
    for (int i = 0; i < m_resultsTree->topLevelItemCount(); ++i) {
        QTreeWidgetItem* groupItem = m_resultsTree->topLevelItem(i);
        
        // Check if group is expanded and visible
        if (groupItem->isExpanded() && !m_resultsTree->visualItemRect(groupItem).isEmpty()) {
            // Get all child items (files)
            for (int j = 0; j < groupItem->childCount(); ++j) {
                QTreeWidgetItem* fileItem = groupItem->child(j);
                
                // Check if file item is visible
                if (!m_resultsTree->visualItemRect(fileItem).isEmpty()) {
                    // Get file path from item data
                    QVariant pathData = fileItem->data(0, Qt::UserRole);
                    if (pathData.isValid()) {
                        QString filePath = pathData.toString();
                        if (!filePath.isEmpty()) {
                            visibleFilePaths.append(filePath);
                        }
                    }
                }
            }
        }
    }
    
    // Preload thumbnails for visible files
    if (!visibleFilePaths.isEmpty()) {
        QSize thumbSize(m_thumbnailDelegate->thumbnailSize(), 
                       m_thumbnailDelegate->thumbnailSize());
        m_thumbnailCache->preloadThumbnails(visibleFilePaths, thumbSize);
        
        LOG_DEBUG(QString("Preloading %1 thumbnails").arg(visibleFilePaths.size()));
    }
}

// Selection History Implementation (Task 17)

void ResultsWindow::undoSelection()
{
    if (!m_selectionHistory->canUndo()) {
        return;
    }
    
    SelectionHistoryManager::SelectionState previousState = m_selectionHistory->undo();
    if (!previousState.selectedFiles.isEmpty() || previousState.description == "Initial state") {
        applySelectionState(previousState);
        updateUndoRedoButtons();
        LOG_INFO(QString("Undo selection: %1").arg(previousState.description));
    }
}

void ResultsWindow::redoSelection()
{
    if (!m_selectionHistory->canRedo()) {
        return;
    }
    
    SelectionHistoryManager::SelectionState nextState = m_selectionHistory->redo();
    if (!nextState.selectedFiles.isEmpty()) {
        applySelectionState(nextState);
        updateUndoRedoButtons();
        LOG_INFO(QString("Redo selection: %1").arg(nextState.description));
    }
}

bool ResultsWindow::canUndo() const
{
    return m_selectionHistory->canUndo();
}

bool ResultsWindow::canRedo() const
{
    return m_selectionHistory->canRedo();
}

void ResultsWindow::onUndoRequested()
{
    undoSelection();
}

void ResultsWindow::onRedoRequested()
{
    redoSelection();
}

void ResultsWindow::onInvertSelection()
{
    // Record current state before inverting
    recordSelectionState("Invert selection");
    
    // Get all file paths
    QStringList allFilePaths;
    QStringList currentlySelected = getCurrentSelectedFilePaths();
    
    // Collect all file paths from all groups
    for (const auto& group : m_currentResults.duplicateGroups) {
        for (const auto& file : group.files) {
            allFilePaths.append(file.filePath);
        }
    }
    
    // Create inverted selection (files that are not currently selected)
    QStringList invertedSelection;
    for (const QString& filePath : allFilePaths) {
        if (!currentlySelected.contains(filePath)) {
            invertedSelection.append(filePath);
        }
    }
    
    // Apply inverted selection
    setSelectedFilePaths(invertedSelection);
    updateSelectionSummary();
    
    LOG_INFO(QString("Inverted selection: %1 files now selected").arg(invertedSelection.size()));
}

void ResultsWindow::recordSelectionState(const QString& description)
{
    QStringList selectedFiles = getCurrentSelectedFilePaths();
    m_selectionHistory->pushState(selectedFiles, description);
    updateUndoRedoButtons();
}

void ResultsWindow::applySelectionState(const SelectionHistoryManager::SelectionState& state)
{
    setSelectedFilePaths(state.selectedFiles);
    updateSelectionSummary();
}

QStringList ResultsWindow::getCurrentSelectedFilePaths() const
{
    QStringList selectedPaths;
    
    // Iterate through all items in the tree to find selected ones
    QTreeWidgetItemIterator it(m_resultsTree);
    while (*it) {
        QTreeWidgetItem* item = *it;
        
        // Check if this is a file item (has parent) and is selected
        if (item->parent() && item->checkState(0) == Qt::Checked) {
            QVariant pathData = item->data(0, Qt::UserRole);
            if (pathData.isValid()) {
                QString filePath = pathData.toString();
                if (!filePath.isEmpty()) {
                    selectedPaths.append(filePath);
                }
            }
        }
        ++it;
    }
    
    return selectedPaths;
}

void ResultsWindow::setSelectedFilePaths(const QStringList& filePaths)
{
    // Clear all selections first
    QTreeWidgetItemIterator it(m_resultsTree);
    while (*it) {
        QTreeWidgetItem* item = *it;
        if (item->parent()) { // File item
            item->setCheckState(0, Qt::Unchecked);
        }
        ++it;
    }
    
    // Set selections for specified files
    QTreeWidgetItemIterator it2(m_resultsTree);
    while (*it2) {
        QTreeWidgetItem* item = *it2;
        if (item->parent()) { // File item
            QVariant pathData = item->data(0, Qt::UserRole);
            if (pathData.isValid()) {
                QString filePath = pathData.toString();
                if (filePaths.contains(filePath)) {
                    item->setCheckState(0, Qt::Checked);
                }
            }
        }
        ++it2;
    }
}

void ResultsWindow::updateUndoRedoButtons()
{
    if (m_undoButton) {
        m_undoButton->setEnabled(m_selectionHistory->canUndo());
        if (m_selectionHistory->canUndo()) {
            m_undoButton->setToolTip(tr("Undo: %1").arg(m_selectionHistory->getUndoDescription()));
        } else {
            m_undoButton->setToolTip(tr("Undo last selection change (Ctrl+Z)"));
        }
    }
    
    if (m_redoButton) {
        m_redoButton->setEnabled(m_selectionHistory->canRedo());
        if (m_selectionHistory->canRedo()) {
            m_redoButton->setToolTip(tr("Redo: %1").arg(m_selectionHistory->getRedoDescription()));
        } else {
            m_redoButton->setToolTip(tr("Redo last undone selection change (Ctrl+Y)"));
        }
    }
}

// File Operation Queue methods (Task 30)

void ResultsWindow::setupOperationQueue()
{
    if (!m_operationQueue || !m_progressDialog) {
        return;
    }
    
    // Set the operation queue for the progress dialog
    m_progressDialog->setOperationQueue(m_operationQueue);
    
    // Connect operation queue signals
    connect(m_operationQueue, &FileOperationQueue::operationStarted,
            this, &ResultsWindow::showOperationProgress);
    
    connect(m_operationQueue, &FileOperationQueue::operationCompleted,
            this, &ResultsWindow::onOperationCompleted);
    
    connect(m_operationQueue, &FileOperationQueue::operationCancelled,
            this, [this](const QString& operationId) {
                onOperationCompleted(operationId, false, "Operation was cancelled");
            });
    
    // Connect progress dialog cancel signal to operation queue
    connect(m_progressDialog, &FileOperationProgressDialog::cancelRequested,
            m_operationQueue, &FileOperationQueue::cancelOperation);
}

void ResultsWindow::showOperationProgress(const QString& operationId)
{
    if (m_progressDialog) {
        m_progressDialog->showForOperation(operationId);
    }
}

void ResultsWindow::onOperationCompleted(const QString& operationId, bool success, const QString& errorMessage)
{
    Q_UNUSED(operationId)
    
    if (success) {
        // Refresh the results to remove deleted files
        refreshResults();
        
        // Show success message in status bar
        if (m_statusLabel) {
            m_statusLabel->setText(tr("Operation completed successfully"));
        }
    } else {
        // Show error message
        if (m_statusLabel) {
            m_statusLabel->setText(tr("Operation failed: %1").arg(errorMessage));
        }
    }
}

// Grouping methods (Task 13)

void ResultsWindow::showGroupingOptions()
{
    if (m_groupingDialog) {
        m_groupingDialog->setGroupingOptions(m_currentGroupingOptions);
        m_groupingDialog->show();
    }
}

void ResultsWindow::applyGrouping(const GroupingOptionsDialog::GroupingOptions& options)
{
    m_currentGroupingOptions = options;
    regroupResults();
    
    // Update status
    QString description = GroupingOptionsDialog::getGroupingDescription(options);
    if (m_statusLabel) {
        m_statusLabel->setText(tr("Regrouped: %1").arg(description));
    }
}

void ResultsWindow::regroupResults()
{
    if (m_currentResults.duplicateGroups.isEmpty()) {
        return;
    }
    
    // Collect all files from current groups
    QList<DuplicateFile> allFiles;
    for (const DuplicateGroup& group : m_currentResults.duplicateGroups) {
        allFiles.append(group.files);
    }
    
    // Regroup files using new criteria
    QList<DuplicateGroup> newGroups = groupFilesByCriteria(allFiles, m_currentGroupingOptions);
    
    // Update current results
    m_currentResults.duplicateGroups = newGroups;
    
    // Refresh the display
    populateResultsTree();
    updateStatisticsDisplay();
}

QList<ResultsWindow::DuplicateGroup> ResultsWindow::groupFilesByCriteria(
    const QList<DuplicateFile>& files, 
    const GroupingOptionsDialog::GroupingOptions& options) const
{
    QHash<QString, QList<DuplicateFile>> primaryGroups;
    
    // Group by primary criteria
    for (const DuplicateFile& file : files) {
        QString primaryKey = getGroupKey(file, options.primaryCriteria, options);
        primaryGroups[primaryKey].append(file);
    }
    
    QList<DuplicateGroup> result;
    
    // Process each primary group
    for (auto it = primaryGroups.begin(); it != primaryGroups.end(); ++it) {
        const QString& primaryKey = it.key();
        const QList<DuplicateFile>& primaryFiles = it.value();
        
        if (options.useSecondaryCriteria && primaryFiles.size() > 1) {
            // Further group by secondary criteria
            QHash<QString, QList<DuplicateFile>> secondaryGroups;
            
            for (const DuplicateFile& file : primaryFiles) {
                QString secondaryKey = getGroupKey(file, options.secondaryCriteria, options);
                QString combinedKey = primaryKey + "|" + secondaryKey;
                secondaryGroups[combinedKey].append(file);
            }
            
            // Create groups from secondary grouping
            for (auto secIt = secondaryGroups.begin(); secIt != secondaryGroups.end(); ++secIt) {
                const QList<DuplicateFile>& groupFiles = secIt.value();
                if (groupFiles.size() > 1) { // Only create groups with multiple files
                    DuplicateGroup group;
                    group.groupId = secIt.key();
                    group.files = groupFiles;
                    result.append(group);
                }
            }
        } else {
            // Create group from primary criteria only
            if (primaryFiles.size() > 1) { // Only create groups with multiple files
                DuplicateGroup group;
                group.groupId = primaryKey;
                group.files = primaryFiles;
                result.append(group);
            }
        }
    }
    
    return result;
}

QString ResultsWindow::getGroupKey(const DuplicateFile& file, 
                                  GroupingOptionsDialog::GroupingCriteria criteria,
                                  const GroupingOptionsDialog::GroupingOptions& options) const
{
    switch (criteria) {
        case GroupingOptionsDialog::GroupingCriteria::Hash:
            return file.hash;
            
        case GroupingOptionsDialog::GroupingCriteria::Size:
            return QString::number(file.fileSize);
            
        case GroupingOptionsDialog::GroupingCriteria::Type: {
            QFileInfo fileInfo(file.filePath);
            QString extension = fileInfo.suffix();
            if (!options.caseSensitiveTypes) {
                extension = extension.toLower();
            }
            return extension;
        }
        
        case GroupingOptionsDialog::GroupingCriteria::CreationDate: {
            QDateTime dateTime = file.created;
            switch (options.dateGrouping) {
                case GroupingOptionsDialog::DateGrouping::ExactDate:
                    return dateTime.toString(Qt::ISODate);
                case GroupingOptionsDialog::DateGrouping::SameDay:
                    return dateTime.date().toString(Qt::ISODate);
                case GroupingOptionsDialog::DateGrouping::SameWeek:
                    return QString("%1-W%2").arg(dateTime.date().year()).arg(dateTime.date().weekNumber());
                case GroupingOptionsDialog::DateGrouping::SameMonth:
                    return dateTime.date().toString("yyyy-MM");
                case GroupingOptionsDialog::DateGrouping::SameYear:
                    return QString::number(dateTime.date().year());
            }
            break;
        }
        
        case GroupingOptionsDialog::GroupingCriteria::ModificationDate: {
            QDateTime dateTime = file.lastModified;
            switch (options.dateGrouping) {
                case GroupingOptionsDialog::DateGrouping::ExactDate:
                    return dateTime.toString(Qt::ISODate);
                case GroupingOptionsDialog::DateGrouping::SameDay:
                    return dateTime.date().toString(Qt::ISODate);
                case GroupingOptionsDialog::DateGrouping::SameWeek:
                    return QString("%1-W%2").arg(dateTime.date().year()).arg(dateTime.date().weekNumber());
                case GroupingOptionsDialog::DateGrouping::SameMonth:
                    return dateTime.date().toString("yyyy-MM");
                case GroupingOptionsDialog::DateGrouping::SameYear:
                    return QString::number(dateTime.date().year());
            }
            break;
        }
        
        case GroupingOptionsDialog::GroupingCriteria::Location: {
            QFileInfo fileInfo(file.filePath);
            QString directory = fileInfo.absolutePath();
            
            if (options.groupByParentDirectory) {
                QDir dir(directory);
                if (dir.cdUp()) {
                    directory = dir.absolutePath();
                }
            }
            
            return directory;
        }
    }
    
    return file.hash; // Fallback to hash
}
