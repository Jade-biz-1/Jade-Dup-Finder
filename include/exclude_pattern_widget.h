#ifndef EXCLUDE_PATTERN_WIDGET_H
#define EXCLUDE_PATTERN_WIDGET_H

#include <QtWidgets/QWidget>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtCore/QStringList>
#include <QtCore/QRegularExpression>

/**
 * @brief Widget for managing file exclusion patterns
 * 
 * This widget provides a user interface for adding, removing, and validating
 * file exclusion patterns. Patterns can be wildcards (*.tmp) or regular expressions.
 * 
 * Features:
 * - Pattern validation with visual feedback
 * - Pattern testing against sample filenames
 * - Built-in common patterns
 * - Persistence using QSettings
 */
class ExcludePatternWidget : public QWidget
{
    Q_OBJECT

public:
    /**
     * @brief Construct a new Exclude Pattern Widget
     * @param parent Parent widget
     */
    explicit ExcludePatternWidget(QWidget* parent = nullptr);
    
    /**
     * @brief Destructor
     */
    ~ExcludePatternWidget() override;
    
    /**
     * @brief Get all current patterns
     * @return List of pattern strings
     */
    QStringList getPatterns() const;
    
    /**
     * @brief Set patterns from a list
     * @param patterns List of pattern strings to set
     */
    void setPatterns(const QStringList& patterns);
    
    /**
     * @brief Add a single pattern
     * @param pattern Pattern string to add
     * @return true if pattern was added successfully, false if invalid
     */
    bool addPattern(const QString& pattern);
    
    /**
     * @brief Remove a pattern
     * @param pattern Pattern string to remove
     */
    void removePattern(const QString& pattern);
    
    /**
     * @brief Clear all patterns
     */
    void clearPatterns();
    
    /**
     * @brief Validate a pattern string
     * @param pattern Pattern to validate
     * @param errorMessage Output parameter for error message if invalid
     * @return true if pattern is valid, false otherwise
     */
    static bool validatePattern(const QString& pattern, QString* errorMessage = nullptr);
    
    /**
     * @brief Test if a filename matches any of the current patterns
     * @param filename Filename to test
     * @return true if filename matches any pattern, false otherwise
     */
    bool matchesAnyPattern(const QString& filename) const;
    
    /**
     * @brief Load patterns from QSettings
     * @param settingsKey Key to use in QSettings (default: "excludePatterns")
     */
    void loadFromSettings(const QString& settingsKey = "excludePatterns");
    
    /**
     * @brief Save patterns to QSettings
     * @param settingsKey Key to use in QSettings (default: "excludePatterns")
     */
    void saveToSettings(const QString& settingsKey = "excludePatterns");

signals:
    /**
     * @brief Emitted when the pattern list changes
     * @param patterns Current list of patterns
     */
    void patternsChanged(const QStringList& patterns);
    
    /**
     * @brief Emitted when a pattern is added
     * @param pattern The pattern that was added
     */
    void patternAdded(const QString& pattern);
    
    /**
     * @brief Emitted when a pattern is removed
     * @param pattern The pattern that was removed
     */
    void patternRemoved(const QString& pattern);

private slots:
    void onAddButtonClicked();
    void onRemoveButtonClicked();
    void onPatternInputChanged(const QString& text);
    void onPatternInputReturnPressed();
    void onPatternListSelectionChanged();
    void onTestButtonClicked();
    void onAddCommonButtonClicked();

private:
    void setupUI();
    void setupConnections();
    void updateButtonStates();
    void showValidationFeedback(bool valid, const QString& message = QString());
    QStringList getCommonPatterns() const;
    
    // UI Components
    QVBoxLayout* m_mainLayout;
    QLabel* m_titleLabel;
    QListWidget* m_patternList;
    QLineEdit* m_patternInput;
    QPushButton* m_addButton;
    QPushButton* m_removeButton;
    QPushButton* m_testButton;
    QPushButton* m_addCommonButton;
    QLabel* m_validationLabel;
    QHBoxLayout* m_inputLayout;
    QHBoxLayout* m_buttonLayout;
    
    // Data
    QStringList m_patterns;
    
    // Styling
    void applyStyles();
};

#endif // EXCLUDE_PATTERN_WIDGET_H
