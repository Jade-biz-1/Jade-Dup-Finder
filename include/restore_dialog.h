#ifndef RESTORE_DIALOG_H
#define RESTORE_DIALOG_H

#include <QDialog>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QComboBox>
#include "safety_manager.h"

class RestoreDialog : public QDialog
{
    Q_OBJECT

public:
    explicit RestoreDialog(SafetyManager* safetyManager, QWidget* parent = nullptr);
    ~RestoreDialog();

signals:
    void filesRestored(const QStringList& files);

private slots:
    void onRestoreClicked();
    void onRestoreAllClicked();
    void onDeleteBackupClicked();
    void onRefreshClicked();
    void onSearchTextChanged(const QString& text);
    void onFilterChanged();
    void onSelectionChanged();
    void onTableDoubleClicked(int row, int column);

private:
    void setupUI();
    void loadBackups();
    void applyFilters();
    void updateStats();
    QString formatFileSize(qint64 bytes) const;
    QString formatDateTime(const QDateTime& dt) const;
    QString getOperationTypeString(SafetyManager::OperationType type) const;
    
    // UI Components
    QVBoxLayout* m_mainLayout;
    
    // Filter controls
    QWidget* m_filterWidget;
    QLineEdit* m_searchEdit;
    QComboBox* m_typeFilter;
    QPushButton* m_refreshButton;
    
    // Table
    QTableWidget* m_backupTable;
    
    // Stats
    QLabel* m_statsLabel;
    
    // Action buttons
    QHBoxLayout* m_buttonLayout;
    QPushButton* m_restoreButton;
    QPushButton* m_restoreAllButton;
    QPushButton* m_deleteBackupButton;
    QPushButton* m_closeButton;
    
    // Data
    SafetyManager* m_safetyManager;
    QList<SafetyManager::SafetyOperation> m_allBackups;
    QList<SafetyManager::SafetyOperation> m_filteredBackups;
};

#endif // RESTORE_DIALOG_H
