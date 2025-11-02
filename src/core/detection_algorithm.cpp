#include "detection_algorithm.h"
#include <QFileInfo>

bool DetectionAlgorithm::canHandle(const QString& filePath) const
{
    QStringList extensions = supportedExtensions();
    
    // If no specific extensions, algorithm handles all files
    if (extensions.isEmpty()) {
        return true;
    }
    
    return hasValidExtension(filePath, extensions);
}

bool DetectionAlgorithm::hasValidExtension(const QString& filePath, const QStringList& extensions) const
{
    if (extensions.isEmpty()) {
        return true;
    }
    
    QFileInfo fileInfo(filePath);
    QString extension = fileInfo.suffix().toLower();
    
    for (const QString& ext : extensions) {
        if (extension == ext.toLower()) {
            return true;
        }
    }
    
    return false;
}