#include "trash_manager.h"
#include <QString>
#include <windows.h>
#include <shellapi.h>

bool TrashManager::moveToTrash(const QString& filePath) {
    SHFILEOPSTRUCTA fileOp = {0};
    fileOp.hwnd = nullptr;
    fileOp.wFunc = FO_DELETE;
    fileOp.pFrom = filePath.toLocal8Bit().constData();
    fileOp.pTo = nullptr;
    fileOp.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT;
    return (SHFileOperationA(&fileOp) == 0);
}