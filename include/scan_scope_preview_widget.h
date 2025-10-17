#ifndef SCAN_SCOPE_PREVIEW_WIDGET_H
#define SCAN_SCOPE_PREVIEW_WIDGET_H

#include <QtWidgets/QWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QTreeWidget>
#include <QtCore/QTimer>
#include <QtCore/QString>
#include <QtCore/QStringList>

// Forward declaration
class ScanSetupDialog;

class ScanScopePreviewWidget : public QWidget
{
    Q_OBJECT

public:
    struct ScopeStats {
        int folderCount = 0;
        int estimatedFileCount = 0;
        qint64 estimatedSize = 0;
        QStringList includedPaths;
        QStringList excludedPaths;
        bool calculationComplete = false;
        QString errorMessage;
    };

    explicit ScanScopePreviewWidget(QWidget* parent = nullptr);
    ~ScanScopePreviewWidget() override;

    // Update preview with new configuration
    void updatePreview(const QStringList& targetPaths,
                      const QStringList& excludePatterns,
                      const QStringList& excludeFolders,
                      int maxDepth,
                      bool includeHidden);

    // Get current statistics
    ScopeStats getCurrentStats() const;

    // Clear the preview
    void clear();

signals:
    void previewUpdated(const ScopeStats& stats);
    void calculationStarted();
    void calculationFinished();

private slots:
    void performCalculation();

private:
    void setupUI();
    void updateDisplay();
    void calculateStats(const QStringList& targetPaths,
                       const QStringList& excludePatterns,
                       const QStringList& excludeFolders,
                       int maxDepth,
                       bool includeHidden);
    
    QString formatFileSize(qint64 bytes) const;
    QString formatNumber(int number) const;
    bool shouldExcludePath(const QString& path,
                          const QStringList& excludePatterns,
                          const QStringList& excludeFolders) const;
    bool matchesPattern(const QString& path, const QString& pattern) const;

    // UI Components
    QVBoxLayout* m_layout;
    QLabel* m_titleLabel;
    QLabel* m_folderCountLabel;
    QLabel* m_fileCountLabel;
    QLabel* m_sizeLabel;
    QLabel* m_statusLabel;
    QTreeWidget* m_pathsTree;

    // Data
    ScopeStats m_currentStats;
    QTimer* m_updateTimer;
    
    // Pending calculation parameters
    QStringList m_pendingTargetPaths;
    QStringList m_pendingExcludePatterns;
    QStringList m_pendingExcludeFolders;
    int m_pendingMaxDepth;
    bool m_pendingIncludeHidden;
    bool m_calculationPending;

    // Constants
    static const int UPDATE_DELAY_MS = 500;  // Debounce delay
    static const int MAX_SAMPLE_FILES = 1000; // Sample size for estimation
};

Q_DECLARE_METATYPE(ScanScopePreviewWidget::ScopeStats)

#endif // SCAN_SCOPE_PREVIEW_WIDGET_H
