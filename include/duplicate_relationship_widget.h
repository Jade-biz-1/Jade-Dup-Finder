#ifndef DUPLICATE_RELATIONSHIP_WIDGET_H
#define DUPLICATE_RELATIONSHIP_WIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QPushButton>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QGraphicsTextItem>
#include <QPen>
#include <QBrush>
#include <QColor>
#include <QMap>
#include <QList>
#include <QTimer>

/**
 * @brief Widget for visualizing duplicate file relationships
 * 
 * This widget provides visual representation of which files are duplicates
 * of each other, showing connections between duplicate files in a group.
 * 
 * Features:
 * - Visual graph showing file relationships
 * - Color-coded duplicate groups
 * - Interactive hover tooltips
 * - Zoom and pan capabilities
 * - Integration with ResultsWindow
 * 
 * Requirements: 3.7
 */
class DuplicateRelationshipWidget : public QWidget {
    Q_OBJECT

public:
    struct FileNode {
        QString filePath;
        QString fileName;
        qint64 fileSize;
        QString hash;
        QColor groupColor;
        bool isRecommended;
        QGraphicsEllipseItem* graphicsItem;
        QGraphicsTextItem* labelItem;
    };

    struct DuplicateGroup {
        QString groupId;
        QString hash;
        QList<FileNode> files;
        QColor groupColor;
        int totalFiles;
        qint64 totalSize;
    };

    explicit DuplicateRelationshipWidget(QWidget* parent = nullptr);
    ~DuplicateRelationshipWidget();

    // Main interface
    void setDuplicateGroups(const QList<DuplicateGroup>& groups);
    void clearVisualization();
    void highlightFile(const QString& filePath);
    void setSelectedFiles(const QStringList& filePaths);

    // Configuration
    void setShowFileNames(bool show);
    void setShowFileSizes(bool show);
    void setNodeSize(int size);
    void setZoomLevel(double zoom);

    // Information
    int getGroupCount() const;
    int getTotalFiles() const;
    QStringList getFilesInGroup(const QString& groupId) const;

public slots:
    void refreshVisualization();
    void fitToView();
    void zoomIn();
    void zoomOut();
    void resetZoom();

signals:
    void fileClicked(const QString& filePath);
    void fileDoubleClicked(const QString& filePath);
    void groupClicked(const QString& groupId);
    void selectionChanged(const QStringList& selectedFiles);

protected:
    void resizeEvent(QResizeEvent* event) override;
    void showEvent(QShowEvent* event) override;

private slots:
    void onNodeClicked(QGraphicsItem* item);
    void onNodeDoubleClicked(QGraphicsItem* item);
    void onSceneSelectionChanged();
    void updateLayout();

private:
    void setupUI();
    void setupGraphicsView();
    void createVisualization();
    void layoutNodes();
    void createConnections();
    void updateNodeAppearance();
    void updateTooltips();
    
    // Layout algorithms
    void layoutCircular();
    void layoutForceDirected();
    void layoutHierarchical();
    
    // Visual helpers
    QColor generateGroupColor(int groupIndex);
    QString formatFileSize(qint64 bytes) const;
    QString createTooltipText(const FileNode& node) const;
    
    // Graphics items management
    void clearGraphicsItems();
    QGraphicsEllipseItem* createNodeItem(const FileNode& node);
    QGraphicsTextItem* createLabelItem(const FileNode& node);
    QGraphicsLineItem* createConnectionItem(const FileNode& from, const FileNode& to);

    // UI Components
    QVBoxLayout* m_mainLayout;
    QHBoxLayout* m_controlsLayout;
    
    // Controls
    QLabel* m_titleLabel;
    QPushButton* m_fitButton;
    QPushButton* m_zoomInButton;
    QPushButton* m_zoomOutButton;
    QPushButton* m_resetButton;
    QPushButton* m_layoutButton;
    
    // Graphics view
    QGraphicsView* m_graphicsView;
    QGraphicsScene* m_graphicsScene;
    
    // Data
    QList<DuplicateGroup> m_duplicateGroups;
    QMap<QString, FileNode*> m_fileNodes;
    QStringList m_selectedFiles;
    
    // Configuration
    bool m_showFileNames;
    bool m_showFileSizes;
    int m_nodeSize;
    double m_zoomLevel;
    
    // Layout
    QTimer* m_layoutTimer;
    enum LayoutMode {
        Circular,
        ForceDirected,
        Hierarchical
    } m_layoutMode;
    
    // Constants
    static const int DEFAULT_NODE_SIZE = 20;
    static const int MIN_NODE_SIZE = 10;
    static const int MAX_NODE_SIZE = 50;
    static constexpr double DEFAULT_ZOOM = 1.0;
    static constexpr double MIN_ZOOM = 0.1;
    static constexpr double MAX_ZOOM = 5.0;
};

#endif // DUPLICATE_RELATIONSHIP_WIDGET_H