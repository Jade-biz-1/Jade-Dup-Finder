#ifndef TRASH_MANAGER_H
#define TRASH_MANAGER_H

#include <QString>

class TrashManager {
public:
    static bool moveToTrash(const QString& filePath);
};

#endif // TRASH_MANAGER_H