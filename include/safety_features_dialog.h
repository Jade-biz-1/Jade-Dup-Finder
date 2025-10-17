#pragma once

#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QTabWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QGroupBox>
#include <QSplitter>
#include <QTextEdit>
#include <QComboBox>
#include <QLineEdit>
#include <QCheckBox>
#include <QSpinBox>
#include <QProgressBar>
#include <QTimer>
#include "safety_manager.h"

/**
 * @brief Dialog for managing SafetyManager protection features (T17)
 * 
 * This dialog provides UI for the SafetyManager's protection functionality:
 * - View and manage protection rules
 * - Configure system file protection
 * - View protection statistics
 * - Test file protection status
 * - Configure safety levels and backup strategies
 */
class SafetyFeaturesDialog : public QDialog {
    Q_OBJECT

public:
    explicit SafetyFeaturesDialog(SafetyManager* safetyManager, QWidget* parent = nullptr);
    ~SafetyFeaturesDialog() override = default;

    /**
     * @brief Refresh all data from SafetyManager
     */
    void refreshData();

    /**
     * @brief Show dialog and focus on a specific file's protection status
     */
    void showProtectionForFile(const QString& filePath);

signals:
    /**
     * @brief Emitted when protection rules are modified
     */
    void protectionRulesChanged();

    /**
     * @brief Emitted when safety settings are changed
     */
    void safetySettingsChanged();

private slots:
    void onProtectionRuleSelectionChanged();
    void onAddProtectionRuleClicked();
    void onEditProtectionRuleClicked();
    void onRemoveProtectionRuleClicked();
    void onTestFileProtectionClicked();
    void onSafetyLevelChanged();
    void onBackupStrategyChanged();
    void onRefreshClicked();
    void onResetToDefaultsClicked();
    void updateProtectionDetails();
    void updateStatistics();
    void updateSystemPaths();

private:
    // UI Components
    QTabWidget* m_tabWidget;
    
    // Protection Rules Tab
    QWidget* m_protectionTab;
    QSplitter* m_protectionSplitter;
    
    // Left panel - Protection rules list
    QGroupBox* m_rulesGroup;
    QVBoxLayout* m_rulesLayout;
    QTableWidget* m_rulesTable;
    QHBoxLayout* m_rulesButtonLayout;
    QPushButton* m_addRuleButton;
    QPushButton* m_editRuleButton;
    QPushButton* m_removeRuleButton;
    
    // Right panel - Rule details and testing
    QGroupBox* m_detailsGroup;
    QVBoxLayout* m_detailsLayout;
    QTextEdit* m_ruleDetailsText;
    
    QGroupBox* m_testGroup;
    QVBoxLayout* m_testLayout;
    QLineEdit* m_testFileEdit;
    QPushButton* m_testFileButton;
    QPushButton* m_browseFileButton;
    QLabel* m_testResultLabel;
    
    // Settings Tab
    QWidget* m_settingsTab;
    QVBoxLayout* m_settingsLayout;
    
    QGroupBox* m_safetyLevelGroup;
    QGridLayout* m_safetyLevelLayout;
    QComboBox* m_safetyLevelCombo;
    QLabel* m_safetyLevelDescription;
    
    QGroupBox* m_backupGroup;
    QGridLayout* m_backupLayout;
    QComboBox* m_backupStrategyCombo;
    QLineEdit* m_backupDirectoryEdit;
    QPushButton* m_browseBackupDirButton;
    QSpinBox* m_maxBackupAgeSpin;
    QSpinBox* m_maxUndoOperationsSpin;
    
    // System Paths Tab
    QWidget* m_systemPathsTab;
    QVBoxLayout* m_systemPathsLayout;
    QTableWidget* m_systemPathsTable;
    QHBoxLayout* m_systemPathsButtonLayout;
    QPushButton* m_addSystemPathButton;
    QPushButton* m_removeSystemPathButton;
    
    // Statistics Tab
    QWidget* m_statisticsTab;
    QVBoxLayout* m_statisticsLayout;
    
    QGroupBox* m_operationStatsGroup;
    QGridLayout* m_operationStatsLayout;
    QLabel* m_totalOperationsLabel;
    QLabel* m_totalBackupsLabel;
    QLabel* m_backupSizeLabel;
    QLabel* m_protectedFilesLabel;
    
    QGroupBox* m_operationBreakdownGroup;
    QTableWidget* m_operationBreakdownTable;
    
    // Dialog buttons
    QHBoxLayout* m_buttonLayout;
    QPushButton* m_refreshButton;
    QPushButton* m_resetButton;
    QPushButton* m_closeButton;
    
    // Data
    SafetyManager* m_safetyManager;
    QList<SafetyManager::ProtectionEntry> m_protectionRules;
    SafetyManager::ProtectionEntry m_selectedRule;
    
    // Helper methods
    void setupUI();
    void setupConnections();
    void setupProtectionTab();
    void setupSettingsTab();
    void setupSystemPathsTab();
    void setupStatisticsTab();
    void populateProtectionRulesTable();
    void populateSystemPathsTable();
    void populateOperationBreakdownTable();
    SafetyManager::ProtectionEntry getSelectedProtectionRule() const;
    QString formatProtectionLevel(SafetyManager::ProtectionLevel level) const;
    QString formatOperationType(SafetyManager::OperationType type) const;
    QString formatSafetyLevel(SafetyManager::SafetyLevel level) const;
    QString formatBackupStrategy(SafetyManager::BackupStrategy strategy) const;
    QIcon getProtectionIcon(SafetyManager::ProtectionLevel level) const;
    void showAddEditRuleDialog(bool isEdit = false);
    void testFileProtection(const QString& filePath);
    void updateSafetyLevelDescription();
    void validateAndApplySettings();
};

/**
 * @brief Dialog for adding/editing protection rules
 */
class ProtectionRuleDialog : public QDialog {
    Q_OBJECT

public:
    explicit ProtectionRuleDialog(QWidget* parent = nullptr);
    explicit ProtectionRuleDialog(const SafetyManager::ProtectionEntry& rule, QWidget* parent = nullptr);
    
    SafetyManager::ProtectionEntry getProtectionRule() const;

private slots:
    void onPatternChanged();
    void onBrowseClicked();
    void validateInput();

private:
    QVBoxLayout* m_layout;
    QGridLayout* m_formLayout;
    
    QLineEdit* m_patternEdit;
    QPushButton* m_browseButton;
    QComboBox* m_levelCombo;
    QLineEdit* m_descriptionEdit;
    QCheckBox* m_regexCheckBox;
    
    QHBoxLayout* m_buttonLayout;
    QPushButton* m_okButton;
    QPushButton* m_cancelButton;
    
    SafetyManager::ProtectionEntry m_rule;
    bool m_isEdit;
    
    void setupUI();
    void setupConnections();
    void populateFields();
};