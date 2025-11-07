#include "trash_manager.h"
#include <QString>
#include <Foundation/Foundation.h>

bool TrashManager::moveToTrash(const QString& filePath) {
    @autoreleasepool {
        NSError* error = nil;
        NSFileManager* fileManager = [NSFileManager defaultManager];
        // Use stringWithUTF8String (autoreleased) instead of alloc/release
        NSString* nsPath = [NSString stringWithUTF8String:filePath.toUtf8().constData()];
        NSURL* fileURL = [NSURL fileURLWithPath:nsPath];

        BOOL success = [fileManager trashItemAtURL:fileURL
                                  resultingItemURL:nullptr
                                             error:&error];

        // No manual release needed with ARC-compatible code
        return success == YES;
    }
}
