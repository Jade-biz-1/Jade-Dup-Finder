#include "selection_history_manager.h"

SelectionHistoryManager::SelectionHistoryManager(QObject* parent)
    : QObject(parent)
    , m_maxHistorySize(50)  // Default limit of 50 items
{
}

void SelectionHistoryManager::pushState(const SelectionState& state) {
    // Clear redo stack when new state is pushed
    if (!m_redoStack.isEmpty()) {
        m_redoStack.clear();
        emit redoAvailabilityChanged(false);
    }

    // Push new state to undo stack
    m_undoStack.push(state);
    
    // Enforce size limit
    enforceHistoryLimit();
    
    emit undoAvailabilityChanged(true);
}

void SelectionHistoryManager::pushState(const QStringList& selectedFiles, const QString& description) {
    SelectionState state(selectedFiles, description);
    pushState(state);
}

SelectionHistoryManager::SelectionState SelectionHistoryManager::undo() {
    if (m_undoStack.isEmpty()) {
        return SelectionState();
    }

    // Move current state to redo stack
    SelectionState currentState = m_undoStack.pop();
    m_redoStack.push(currentState);

    // Emit availability changes
    emit undoAvailabilityChanged(!m_undoStack.isEmpty());
    emit redoAvailabilityChanged(true);

    // Return the previous state (if any)
    if (!m_undoStack.isEmpty()) {
        return m_undoStack.top();
    }
    
    // If no previous state, return empty selection
    return SelectionState(QStringList(), tr("Initial state"));
}

SelectionHistoryManager::SelectionState SelectionHistoryManager::redo() {
    if (m_redoStack.isEmpty()) {
        return SelectionState();
    }

    // Move state back to undo stack
    SelectionState redoState = m_redoStack.pop();
    m_undoStack.push(redoState);

    // Emit availability changes
    emit redoAvailabilityChanged(!m_redoStack.isEmpty());
    emit undoAvailabilityChanged(true);

    return redoState;
}

bool SelectionHistoryManager::canUndo() const {
    return !m_undoStack.isEmpty();
}

bool SelectionHistoryManager::canRedo() const {
    return !m_redoStack.isEmpty();
}

QString SelectionHistoryManager::getUndoDescription() const {
    if (m_undoStack.isEmpty()) {
        return QString();
    }
    return m_undoStack.top().description;
}

QString SelectionHistoryManager::getRedoDescription() const {
    if (m_redoStack.isEmpty()) {
        return QString();
    }
    return m_redoStack.top().description;
}

void SelectionHistoryManager::clear() {
    bool hadUndo = !m_undoStack.isEmpty();
    bool hadRedo = !m_redoStack.isEmpty();

    m_undoStack.clear();
    m_redoStack.clear();

    if (hadUndo) {
        emit undoAvailabilityChanged(false);
    }
    if (hadRedo) {
        emit redoAvailabilityChanged(false);
    }

    emit historyCleared();
}

void SelectionHistoryManager::setMaxHistorySize(int size) {
    if (size <= 0) {
        return;
    }

    m_maxHistorySize = size;
    enforceHistoryLimit();
}

int SelectionHistoryManager::getMaxHistorySize() const {
    return m_maxHistorySize;
}

int SelectionHistoryManager::getUndoStackSize() const {
    return static_cast<int>(m_undoStack.size());
}

int SelectionHistoryManager::getRedoStackSize() const {
    return static_cast<int>(m_redoStack.size());
}

void SelectionHistoryManager::enforceHistoryLimit() {
    // Remove oldest entries if we exceed the limit
    while (m_undoStack.size() > m_maxHistorySize) {
        // Remove from bottom of stack (oldest entries)
        QStack<SelectionState> tempStack;
        
        // Move all but the bottom item to temp stack
        while (m_undoStack.size() > 1) {
            tempStack.push(m_undoStack.pop());
        }
        
        // Remove the bottom item
        if (!m_undoStack.isEmpty()) {
            m_undoStack.pop();
        }
        
        // Restore the rest
        while (!tempStack.isEmpty()) {
            m_undoStack.push(tempStack.pop());
        }
    }
}

void SelectionHistoryManager::emitAvailabilitySignals() {
    emit undoAvailabilityChanged(canUndo());
    emit redoAvailabilityChanged(canRedo());
}