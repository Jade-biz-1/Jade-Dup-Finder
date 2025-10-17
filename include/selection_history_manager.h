#pragma once

#include <QObject>
#include <QStack>
#include <QStringList>
#include <QDateTime>

/**
 * @brief Selection History Manager for undo/redo operations (Task 16)
 * 
 * This class manages a history of selection states to support undo/redo
 * functionality in the results window. It maintains separate stacks for
 * undo and redo operations with a configurable size limit.
 */
class SelectionHistoryManager : public QObject {
    Q_OBJECT

public:
    /**
     * @brief Selection state structure
     */
    struct SelectionState {
        QStringList selectedFiles;      // List of selected file paths
        QString description;            // Human-readable description of the action
        QDateTime timestamp;            // When this state was created
        
        SelectionState() : timestamp(QDateTime::currentDateTime()) {}
        SelectionState(const QStringList& files, const QString& desc)
            : selectedFiles(files), description(desc), timestamp(QDateTime::currentDateTime()) {}
    };

    explicit SelectionHistoryManager(QObject* parent = nullptr);
    ~SelectionHistoryManager() = default;

    /**
     * @brief Push a new selection state onto the undo stack
     * @param state The selection state to save
     */
    void pushState(const SelectionState& state);

    /**
     * @brief Push a new selection state with files and description
     * @param selectedFiles List of selected file paths
     * @param description Description of the action
     */
    void pushState(const QStringList& selectedFiles, const QString& description);

    /**
     * @brief Undo the last action
     * @return The previous selection state, or empty state if no undo available
     */
    SelectionState undo();

    /**
     * @brief Redo the last undone action
     * @return The next selection state, or empty state if no redo available
     */
    SelectionState redo();

    /**
     * @brief Check if undo is available
     */
    bool canUndo() const;

    /**
     * @brief Check if redo is available
     */
    bool canRedo() const;

    /**
     * @brief Get description of the next undo action
     */
    QString getUndoDescription() const;

    /**
     * @brief Get description of the next redo action
     */
    QString getRedoDescription() const;

    /**
     * @brief Clear all history
     */
    void clear();

    /**
     * @brief Set the maximum history size (default: 50)
     */
    void setMaxHistorySize(int size);

    /**
     * @brief Get the maximum history size
     */
    int getMaxHistorySize() const;

    /**
     * @brief Get the current undo stack size
     */
    int getUndoStackSize() const;

    /**
     * @brief Get the current redo stack size
     */
    int getRedoStackSize() const;

signals:
    /**
     * @brief Emitted when undo/redo availability changes
     */
    void undoAvailabilityChanged(bool canUndo);
    void redoAvailabilityChanged(bool canRedo);

    /**
     * @brief Emitted when history is cleared
     */
    void historyCleared();

private:
    QStack<SelectionState> m_undoStack;
    QStack<SelectionState> m_redoStack;
    int m_maxHistorySize;

    void enforceHistoryLimit();
    void emitAvailabilitySignals();
};