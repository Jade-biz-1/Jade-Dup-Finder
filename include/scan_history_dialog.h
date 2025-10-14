#ifndef SCAN_HISTORY_DIALOG_H
#define SCAN_HISTORY_DIALOG_H

#include <QDialog>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QComboBox>
#include <QDateEdit>
#include <QLabel>
#include "scan_history_manager.h"

class ScanHistoryDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ScanHistoryDialog(QWidget* parent = nullptr);
    ~ScanHistoryDialog();

    void refreshHistory();

signals:
    void scanSelected(const QString& scanId);
    void scanDeleted(const QString& scanId);

private slots:
    void onViewClicked();
    void onDeleteClicked();
    void onRefreshClicked();
    void onExportClicked();
    void onClearOldClicked();
    void onSearchTextChanged(const QString& text);
    void onFilterChanged();
    void onTableDoubleClicked(int row, int column);
    void onSelectionChanged();

private:
    void setupUI();
    void loadHistory();
    void applyFilters();
    void updateStats();
    QString formatFileSize(qint64 bytes) const;
    QString formatDateTime(const QDateTime& dt) const;
    
    // UI Components
    QVBoxLayout* m_mainLayout;
    
    // Filter controls
    QWidget* m_filterWidget;
    QLineEdit* m_searchEdit;
    QComboBox* m_typeFilter;
    QDateEdit* m_dateFromEdit;
    QDateEdit* m_dateToEdit;
    QPushButton* m_refreshButton;
    
    // Table
    QTableWidget* m_historyTable;
    
    // Stats
    QLabel* m_statsLabel;
    
    // Action buttons
    QHBoxLayout* m_buttonLayout;
    QPushButton* m_viewButton;
    QPushButton* m_deleteButton;
    QPushButton* m_exportButton;
    QPushButton* m_clearOldButton;
    QPushButton* m_closeButton;
    
    // Data
    QList<ScanHistoryManager::ScanRecord> m_allScans;
    QList<ScanHistoryManager::ScanRecord> m_filteredScans;
};

#endif // SCAN_HISTORY_DIALOG_H
