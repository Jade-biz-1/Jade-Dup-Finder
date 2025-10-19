#include "duplicate_relationship_widget.h"
#include <QApplication>
#include <QGraphicsProxyWidget>
#include <QGraphicsEffect>
#include <QPropertyAnimation>
#include <QSequentialAnimationGroup>
#include <QParallelAnimationGroup>
#include <QEasingCurve>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QToolTip>

#include <QtMath>
#include <QRandomGenerator>

DuplicateRelationshipWidget::DuplicateRelationshipWidget(QWidget* parent)
    : QWidget(parent)
    , m_mainLayout(nullptr)
    , m_controlsLayout(nullptr)
    , m_titleLabel(nullptr)
    , m_fitButton(nullptr)
    , m_zoomInButton(nullptr)
    , m_zoomOutButton(nullptr)
    , m_resetButton(nullptr)
    , m_layoutButton(nullptr)
    , m_graphicsView(nullptr)
    , m_graphicsScene(nullptr)
    , m_showFileNames(true)
    , m_showFileSizes(false)
    , m_nodeSize(DEFAULT_NODE_SIZE)
    , m_zoomLevel(DEFAULT_ZOOM)
    , m_layoutTimer(new QTimer(this))
    , m_layoutMode(Circular)
{
    setupUI();
    setupGraphicsView();
    
    // Setup layout timer
    m_layoutTimer->setSingleShot(true);
    m_layoutTimer->setInterval(100);
    connect(m_layoutTimer, &QTimer::timeout, this, &DuplicateRelationshipWidget::updateLayout);
}

DuplicateRelationshipWidget::~DuplicateRelationshipWidget()
{
    clearGraphicsItems();
}

void DuplicateRelationshipWidget::setupUI()
{
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setContentsMargins(8, 8, 8, 8);
    m_mainLayout->setSpacing(8);
    
    // Controls layout
    m_controlsLayout = new QHBoxLayout();
    m_controlsLayout->setSpacing(8);
    
    // Title
    m_titleLabel = new QLabel(tr("Duplicate Relationships"), this);
    m_titleLabel->setStyleSheet("font-weight: bold; font-size: 12pt;");
    
    // Control buttons
    m_fitButton = new QPushButton(tr("Fit"), this);
    m_fitButton->setToolTip(tr("Fit visualization to view"));
    m_fitButton->setMaximumWidth(60);
    
    m_zoomInButton = new QPushButton(tr("+"), this);
    m_zoomInButton->setToolTip(tr("Zoom in"));
    m_zoomInButton->setMaximumWidth(30);
    
    m_zoomOutButton = new QPushButton(tr("-"), this);
    m_zoomOutButton->setToolTip(tr("Zoom out"));
    m_zoomOutButton->setMaximumWidth(30);
    
    m_resetButton = new QPushButton(tr("Reset"), this);
    m_resetButton->setToolTip(tr("Reset zoom and position"));
    m_resetButton->setMaximumWidth(60);
    
    m_layoutButton = new QPushButton(tr("Layout"), this);
    m_layoutButton->setToolTip(tr("Change layout algorithm"));
    m_layoutButton->setMaximumWidth(70);
    
    // Add to controls layout
    m_controlsLayout->addWidget(m_titleLabel);
    m_controlsLayout->addStretch();
    m_controlsLayout->addWidget(m_fitButton);
    m_controlsLayout->addWidget(m_zoomInButton);
    m_controlsLayout->addWidget(m_zoomOutButton);
    m_controlsLayout->addWidget(m_resetButton);
    m_controlsLayout->addWidget(m_layoutButton);
    
    m_mainLayout->addLayout(m_controlsLayout);
    
    // Connect buttons
    connect(m_fitButton, &QPushButton::clicked, this, &DuplicateRelationshipWidget::fitToView);
    connect(m_zoomInButton, &QPushButton::clicked, this, &DuplicateRelationshipWidget::zoomIn);
    connect(m_zoomOutButton, &QPushButton::clicked, this, &DuplicateRelationshipWidget::zoomOut);
    connect(m_resetButton, &QPushButton::clicked, this, &DuplicateRelationshipWidget::resetZoom);
    connect(m_layoutButton, &QPushButton::clicked, this, [this]() {
        // Cycle through layout modes
        m_layoutMode = static_cast<LayoutMode>((m_layoutMode + 1) % 3);
        updateLayout();
    });
}

void DuplicateRelationshipWidget::setupGraphicsView()
{
    // Create graphics scene
    m_graphicsScene = new QGraphicsScene(this);
    m_graphicsScene->setBackgroundBrush(QBrush(QColor(250, 250, 250)));
    
    // Create graphics view
    m_graphicsView = new QGraphicsView(m_graphicsScene, this);
    m_graphicsView->setRenderHint(QPainter::Antialiasing);
    m_graphicsView->setDragMode(QGraphicsView::RubberBandDrag);
    m_graphicsView->setInteractive(true);
    m_graphicsView->setMinimumHeight(300);
    
    // Enable zooming with mouse wheel
    m_graphicsView->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    m_graphicsView->setResizeAnchor(QGraphicsView::AnchorUnderMouse);
    
    m_mainLayout->addWidget(m_graphicsView);
    
    // Connect scene signals
    connect(m_graphicsScene, &QGraphicsScene::selectionChanged,
            this, &DuplicateRelationshipWidget::onSceneSelectionChanged);
}

void DuplicateRelationshipWidget::setDuplicateGroups(const QList<DuplicateGroup>& groups)
{
    m_duplicateGroups = groups;
    createVisualization();
}

void DuplicateRelationshipWidget::clearVisualization()
{
    clearGraphicsItems();
    m_duplicateGroups.clear();
    m_fileNodes.clear();
    m_selectedFiles.clear();
}

void DuplicateRelationshipWidget::createVisualization()
{
    clearGraphicsItems();
    
    if (m_duplicateGroups.isEmpty()) {
        return;
    }
    
    // Create nodes for each file
    int groupIndex = 0;
    for (const auto& group : m_duplicateGroups) {
        QColor groupColor = generateGroupColor(groupIndex++);
        
        for (const auto& file : group.files) {
            FileNode* node = new FileNode(file);
            node->groupColor = groupColor;
            node->graphicsItem = createNodeItem(*node);
            node->labelItem = createLabelItem(*node);
            
            m_fileNodes[file.filePath] = node;
            m_graphicsScene->addItem(node->graphicsItem);
            m_graphicsScene->addItem(node->labelItem);
        }
    }
    
    // Layout nodes
    layoutNodes();
    
    // Create connections between duplicates
    createConnections();
    
    // Update tooltips
    updateTooltips();
    
    // Fit to view
    QTimer::singleShot(100, this, &DuplicateRelationshipWidget::fitToView);
}

void DuplicateRelationshipWidget::layoutNodes()
{
    switch (m_layoutMode) {
        case Circular:
            layoutCircular();
            break;
        case ForceDirected:
            layoutForceDirected();
            break;
        case Hierarchical:
            layoutHierarchical();
            break;
    }
}

void DuplicateRelationshipWidget::layoutCircular()
{
    if (m_duplicateGroups.isEmpty()) {
        return;
    }
    
    const double centerX = 0;
    const double centerY = 0;
    const double groupRadius = 150;
    const double nodeSpacing = 80;
    
    int groupIndex = 0;
    for (const auto& group : m_duplicateGroups) {
        if (group.files.isEmpty()) {
            continue;
        }
        
        // Calculate group position
        double groupAngle = (2.0 * M_PI * groupIndex) / m_duplicateGroups.size();
        double groupX = centerX + groupRadius * cos(groupAngle);
        double groupY = centerY + groupRadius * sin(groupAngle);
        
        // Layout files in this group in a circle
        int fileIndex = 0;
        for (const auto& file : group.files) {
            FileNode* node = m_fileNodes.value(file.filePath);
            if (!node || !node->graphicsItem) {
                continue;
            }
            
            double fileAngle = (2.0 * M_PI * fileIndex) / group.files.size();
            double fileX = groupX + nodeSpacing * cos(fileAngle);
            double fileY = groupY + nodeSpacing * sin(fileAngle);
            
            node->graphicsItem->setPos(fileX, fileY);
            if (node->labelItem) {
                node->labelItem->setPos(fileX + m_nodeSize + 5, fileY - 10);
            }
            
            fileIndex++;
        }
        
        groupIndex++;
    }
}

void DuplicateRelationshipWidget::layoutForceDirected()
{
    // Simple force-directed layout
    const int iterations = 50;
    const double repulsionForce = 1000.0;
    const double attractionForce = 0.1;
    const double damping = 0.9;
    
    // Initialize random positions
    for (auto it = m_fileNodes.begin(); it != m_fileNodes.end(); ++it) {
        FileNode* node = it.value();
        if (!node || !node->graphicsItem) {
            continue;
        }
        
        double x = (QRandomGenerator::global()->generateDouble() - 0.5) * 400;
        double y = (QRandomGenerator::global()->generateDouble() - 0.5) * 400;
        node->graphicsItem->setPos(x, y);
    }
    
    // Run force-directed algorithm
    for (int iter = 0; iter < iterations; ++iter) {
        QMap<FileNode*, QPointF> forces;
        
        // Calculate repulsion forces
        for (auto it1 = m_fileNodes.begin(); it1 != m_fileNodes.end(); ++it1) {
            FileNode* node1 = it1.value();
            if (!node1 || !node1->graphicsItem) {
                continue;
            }
            
            QPointF pos1 = node1->graphicsItem->pos();
            QPointF totalForce(0, 0);
            
            for (auto it2 = m_fileNodes.begin(); it2 != m_fileNodes.end(); ++it2) {
                FileNode* node2 = it2.value();
                if (!node2 || !node2->graphicsItem || node1 == node2) {
                    continue;
                }
                
                QPointF pos2 = node2->graphicsItem->pos();
                QPointF diff = pos1 - pos2;
                double distance = sqrt(diff.x() * diff.x() + diff.y() * diff.y());
                
                if (distance > 0) {
                    QPointF force = diff * (repulsionForce / (distance * distance));
                    totalForce += force;
                }
            }
            
            forces[node1] = totalForce;
        }
        
        // Apply forces
        for (auto it = forces.begin(); it != forces.end(); ++it) {
            FileNode* node = it.key();
            QPointF force = it.value() * damping;
            QPointF newPos = node->graphicsItem->pos() + force;
            node->graphicsItem->setPos(newPos);
            
            if (node->labelItem) {
                node->labelItem->setPos(newPos.x() + m_nodeSize + 5, newPos.y() - 10);
            }
        }
    }
}

void DuplicateRelationshipWidget::layoutHierarchical()
{
    // Simple hierarchical layout - groups in rows
    const double groupSpacing = 200;
    const double nodeSpacing = 80;
    const double rowHeight = 100;
    
    int groupIndex = 0;
    for (const auto& group : m_duplicateGroups) {
        if (group.files.isEmpty()) {
            continue;
        }
        
        double groupY = groupIndex * rowHeight;
        double startX = -(group.files.size() * nodeSpacing) / 2.0;
        
        int fileIndex = 0;
        for (const auto& file : group.files) {
            FileNode* node = m_fileNodes.value(file.filePath);
            if (!node || !node->graphicsItem) {
                continue;
            }
            
            double fileX = startX + fileIndex * nodeSpacing;
            node->graphicsItem->setPos(fileX, groupY);
            
            if (node->labelItem) {
                node->labelItem->setPos(fileX + m_nodeSize + 5, groupY - 10);
            }
            
            fileIndex++;
        }
        
        groupIndex++;
    }
}

void DuplicateRelationshipWidget::createConnections()
{
    // Create lines connecting duplicate files within each group
    for (const auto& group : m_duplicateGroups) {
        if (group.files.size() < 2) {
            continue;
        }
        
        // Connect each file to every other file in the group
        for (int i = 0; i < group.files.size(); ++i) {
            for (int j = i + 1; j < group.files.size(); ++j) {
                const FileNode& file1 = group.files[i];
                const FileNode& file2 = group.files[j];
                
                FileNode* node1 = m_fileNodes.value(file1.filePath);
                FileNode* node2 = m_fileNodes.value(file2.filePath);
                
                if (node1 && node2 && node1->graphicsItem && node2->graphicsItem) {
                    QGraphicsLineItem* line = createConnectionItem(*node1, *node2);
                    m_graphicsScene->addItem(line);
                }
            }
        }
    }
}

QGraphicsEllipseItem* DuplicateRelationshipWidget::createNodeItem(const FileNode& node)
{
    QGraphicsEllipseItem* item = new QGraphicsEllipseItem(-m_nodeSize/2, -m_nodeSize/2, m_nodeSize, m_nodeSize);
    
    // Set colors
    item->setBrush(QBrush(node.groupColor));
    item->setPen(QPen(node.groupColor.darker(150), 2));
    
    // Make interactive
    item->setFlag(QGraphicsItem::ItemIsSelectable, true);
    item->setFlag(QGraphicsItem::ItemIsMovable, true);
    item->setAcceptHoverEvents(true);
    
    // Store file path in item data
    item->setData(0, node.filePath);
    
    return item;
}

QGraphicsTextItem* DuplicateRelationshipWidget::createLabelItem(const FileNode& node)
{
    if (!m_showFileNames) {
        return nullptr;
    }
    
    QGraphicsTextItem* item = new QGraphicsTextItem(node.fileName);
    item->setFont(QFont("Arial", 8));
    item->setDefaultTextColor(QColor(60, 60, 60));
    
    return item;
}

QGraphicsLineItem* DuplicateRelationshipWidget::createConnectionItem(const FileNode& from, const FileNode& to)
{
    FileNode* fromNode = m_fileNodes.value(from.filePath);
    FileNode* toNode = m_fileNodes.value(to.filePath);
    
    if (!fromNode || !toNode || !fromNode->graphicsItem || !toNode->graphicsItem) {
        return nullptr;
    }
    
    QPointF fromPos = fromNode->graphicsItem->pos();
    QPointF toPos = toNode->graphicsItem->pos();
    
    QGraphicsLineItem* line = new QGraphicsLineItem(fromPos.x(), fromPos.y(), toPos.x(), toPos.y());
    line->setPen(QPen(from.groupColor.lighter(150), 1, Qt::DashLine));
    line->setZValue(-1); // Behind nodes
    
    return line;
}

void DuplicateRelationshipWidget::updateTooltips()
{
    for (auto it = m_fileNodes.begin(); it != m_fileNodes.end(); ++it) {
        FileNode* node = it.value();
        if (!node || !node->graphicsItem) {
            continue;
        }
        
        QString tooltip = createTooltipText(*node);
        node->graphicsItem->setToolTip(tooltip);
    }
}

QString DuplicateRelationshipWidget::createTooltipText(const FileNode& node) const
{
    QString tooltip = QString("<b>%1</b><br>").arg(node.fileName);
    tooltip += QString("Path: %1<br>").arg(node.filePath);
    tooltip += QString("Size: %1<br>").arg(formatFileSize(node.fileSize));
    tooltip += QString("Hash: %1<br>").arg(node.hash.left(16) + "...");
    
    if (node.isRecommended) {
        tooltip += "<br><b>Recommended to keep</b>";
    }
    
    return tooltip;
}

QColor DuplicateRelationshipWidget::generateGroupColor(int groupIndex)
{
    // Generate distinct colors for different groups
    const QList<QColor> colors = {
        QColor(255, 99, 132),   // Red
        QColor(54, 162, 235),   // Blue
        QColor(255, 205, 86),   // Yellow
        QColor(75, 192, 192),   // Teal
        QColor(153, 102, 255),  // Purple
        QColor(255, 159, 64),   // Orange
        QColor(199, 199, 199),  // Grey
        QColor(83, 102, 255),   // Indigo
        QColor(255, 99, 255),   // Pink
        QColor(99, 255, 132)    // Green
    };
    
    return colors[groupIndex % colors.size()];
}

QString DuplicateRelationshipWidget::formatFileSize(qint64 bytes) const
{
    const qint64 KB = 1024;
    const qint64 MB = KB * 1024;
    const qint64 GB = MB * 1024;
    
    if (bytes >= GB) {
        return QString("%1 GB").arg(bytes / (double)GB, 0, 'f', 1);
    } else if (bytes >= MB) {
        return QString("%1 MB").arg(bytes / (double)MB, 0, 'f', 1);
    } else if (bytes >= KB) {
        return QString("%1 KB").arg(bytes / (double)KB, 0, 'f', 1);
    } else {
        return QString("%1 bytes").arg(bytes);
    }
}

void DuplicateRelationshipWidget::clearGraphicsItems()
{
    if (m_graphicsScene) {
        m_graphicsScene->clear();
    }
    
    // Clean up file nodes
    for (auto it = m_fileNodes.begin(); it != m_fileNodes.end(); ++it) {
        delete it.value();
    }
    m_fileNodes.clear();
}

void DuplicateRelationshipWidget::fitToView()
{
    if (m_graphicsView && m_graphicsScene) {
        m_graphicsView->fitInView(m_graphicsScene->itemsBoundingRect(), Qt::KeepAspectRatio);
        m_zoomLevel = m_graphicsView->transform().m11();
    }
}

void DuplicateRelationshipWidget::zoomIn()
{
    if (m_zoomLevel < MAX_ZOOM) {
        m_graphicsView->scale(1.2, 1.2);
        m_zoomLevel *= 1.2;
    }
}

void DuplicateRelationshipWidget::zoomOut()
{
    if (m_zoomLevel > MIN_ZOOM) {
        m_graphicsView->scale(0.8, 0.8);
        m_zoomLevel *= 0.8;
    }
}

void DuplicateRelationshipWidget::resetZoom()
{
    m_graphicsView->resetTransform();
    m_zoomLevel = DEFAULT_ZOOM;
    fitToView();
}

void DuplicateRelationshipWidget::updateLayout()
{
    layoutNodes();
    createConnections();
}

void DuplicateRelationshipWidget::onSceneSelectionChanged()
{
    QStringList selectedFiles;
    
    for (QGraphicsItem* item : m_graphicsScene->selectedItems()) {
        QString filePath = item->data(0).toString();
        if (!filePath.isEmpty()) {
            selectedFiles << filePath;
        }
    }
    
    if (selectedFiles != m_selectedFiles) {
        m_selectedFiles = selectedFiles;
        emit selectionChanged(selectedFiles);
    }
}

void DuplicateRelationshipWidget::highlightFile(const QString& filePath)
{
    // Clear previous highlights
    for (auto it = m_fileNodes.begin(); it != m_fileNodes.end(); ++it) {
        FileNode* node = it.value();
        if (node && node->graphicsItem) {
            node->graphicsItem->setPen(QPen(node->groupColor.darker(150), 2));
        }
    }
    
    // Highlight selected file
    FileNode* node = m_fileNodes.value(filePath);
    if (node && node->graphicsItem) {
        node->graphicsItem->setPen(QPen(Qt::red, 4));
        m_graphicsView->centerOn(node->graphicsItem);
    }
}

void DuplicateRelationshipWidget::setSelectedFiles(const QStringList& filePaths)
{
    m_selectedFiles = filePaths;
    
    // Update graphics scene selection
    m_graphicsScene->clearSelection();
    for (const QString& filePath : filePaths) {
        FileNode* node = m_fileNodes.value(filePath);
        if (node && node->graphicsItem) {
            node->graphicsItem->setSelected(true);
        }
    }
}

// Getters
int DuplicateRelationshipWidget::getGroupCount() const
{
    return m_duplicateGroups.size();
}

int DuplicateRelationshipWidget::getTotalFiles() const
{
    int total = 0;
    for (const auto& group : m_duplicateGroups) {
        total += group.files.size();
    }
    return total;
}

QStringList DuplicateRelationshipWidget::getFilesInGroup(const QString& groupId) const
{
    for (const auto& group : m_duplicateGroups) {
        if (group.groupId == groupId) {
            QStringList files;
            for (const auto& file : group.files) {
                files << file.filePath;
            }
            return files;
        }
    }
    return QStringList();
}

// Configuration setters
void DuplicateRelationshipWidget::setShowFileNames(bool show)
{
    if (m_showFileNames != show) {
        m_showFileNames = show;
        createVisualization(); // Recreate to show/hide labels
    }
}

void DuplicateRelationshipWidget::setShowFileSizes(bool show)
{
    m_showFileSizes = show;
    updateTooltips();
}

void DuplicateRelationshipWidget::setNodeSize(int size)
{
    m_nodeSize = qBound(MIN_NODE_SIZE, size, MAX_NODE_SIZE);
    createVisualization(); // Recreate with new node size
}

void DuplicateRelationshipWidget::setZoomLevel(double zoom)
{
    m_zoomLevel = qBound(MIN_ZOOM, zoom, MAX_ZOOM);
    m_graphicsView->resetTransform();
    m_graphicsView->scale(m_zoomLevel, m_zoomLevel);
}

void DuplicateRelationshipWidget::refreshVisualization()
{
    createVisualization();
}

void DuplicateRelationshipWidget::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    
    // Delay layout update to avoid too frequent updates
    m_layoutTimer->start();
}

void DuplicateRelationshipWidget::showEvent(QShowEvent* event)
{
    QWidget::showEvent(event);
    
    // Fit to view when first shown
    if (!m_duplicateGroups.isEmpty()) {
        QTimer::singleShot(100, this, &DuplicateRelationshipWidget::fitToView);
    }
}