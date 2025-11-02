#include "theme_persistence.h"
#include "core/logger.h"
#include <QSettings>
#include <QStandardPaths>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QColor>
#include <QDateTime>

// Constants
const QString ThemePersistence::THEME_FILE_EXTENSION = ".json";
const QString ThemePersistence::THEME_VERSION = "1.0";
const QString ThemePersistence::PREFERENCES_GROUP = "theme";

bool ThemePersistence::saveThemePreference(ThemeManager::Theme theme, const QString& customName)
{
    QSettings settings;
    settings.beginGroup(PREFERENCES_GROUP);
    
    QString themeString;
    switch (theme) {
        case ThemeManager::Light:
            themeString = "light";
            break;
        case ThemeManager::Dark:
            themeString = "dark";
            break;
        case ThemeManager::Custom:
            themeString = "custom";
            break;
        case ThemeManager::SystemDefault:
        default:
            themeString = "system";
            break;
    }
    
    settings.setValue("current_theme", themeString);
    settings.setValue("custom_theme_name", customName);
    settings.setValue("last_updated", QDateTime::currentDateTime());
    
    settings.endGroup();
    
    // Force sync to ensure settings are written to disk immediately
    settings.sync();
    
    // Verify the settings were saved correctly
    settings.beginGroup(PREFERENCES_GROUP);
    QString savedTheme = settings.value("current_theme", "").toString();
    QString savedCustomName = settings.value("custom_theme_name", "").toString();
    settings.endGroup();
    
    bool success = (savedTheme == themeString && savedCustomName == customName);
    
    if (success) {
        LOG_INFO(LogCategories::CONFIG, QString("Saved theme preference: %1 (%2)")
                 .arg(themeString).arg(customName.isEmpty() ? "default" : customName));
    } else {
        LOG_ERROR(LogCategories::CONFIG, QString("Failed to save theme preference - verification failed. Expected: %1/%2, Got: %3/%4")
                 .arg(themeString).arg(customName).arg(savedTheme).arg(savedCustomName));
    }
    
    return success;
}

QPair<ThemeManager::Theme, QString> ThemePersistence::loadThemePreference()
{
    QSettings settings;
    settings.beginGroup(PREFERENCES_GROUP);
    
    QString themeString = settings.value("current_theme", "system").toString();
    QString customName = settings.value("custom_theme_name", "").toString();
    QDateTime lastUpdated = settings.value("last_updated", QDateTime()).toDateTime();
    
    settings.endGroup();
    
    LOG_DEBUG(LogCategories::CONFIG, QString("Reading theme preference from settings: theme='%1', customName='%2', lastUpdated='%3'")
             .arg(themeString).arg(customName).arg(lastUpdated.toString()));
    
    // Check if we have any settings at all
    if (!settings.contains(QString("%1/current_theme").arg(PREFERENCES_GROUP))) {
        LOG_INFO(LogCategories::CONFIG, "No theme preferences found, using system default");
    }
    
    ThemeManager::Theme theme = ThemeManager::SystemDefault;
    if (themeString == "light") {
        theme = ThemeManager::Light;
    } else if (themeString == "dark") {
        theme = ThemeManager::Dark;
    } else if (themeString == "custom") {
        theme = ThemeManager::Custom;
        
        // Validate that the custom theme still exists
        if (!customName.isEmpty() && !customThemeExists(customName)) {
            LOG_WARNING(LogCategories::CONFIG, QString("Custom theme '%1' no longer exists, falling back to system theme").arg(customName));
            theme = ThemeManager::SystemDefault;
            customName.clear();
            
            // Update settings to reflect the fallback
            saveThemePreference(theme, customName);
        } else if (!customName.isEmpty()) {
            // Validate that the custom theme is not corrupted
            ThemeData themeData = loadCustomTheme(customName);
            if (!themeData.isValid()) {
                LOG_WARNING(LogCategories::CONFIG, QString("Custom theme '%1' is corrupted, falling back to system theme").arg(customName));
                theme = ThemeManager::SystemDefault;
                customName.clear();
                
                // Update settings to reflect the fallback
                saveThemePreference(theme, customName);
            }
        }
    }
    
    LOG_INFO(LogCategories::CONFIG, QString("Loaded theme preference: %1 (%2)")
             .arg(themeString).arg(customName.isEmpty() ? "default" : customName));
    
    return qMakePair(theme, customName);
}

bool ThemePersistence::saveCustomTheme(const QString& name, const ThemeData& theme)
{
    if (name.isEmpty() || !theme.isValid()) {
        LOG_ERROR(LogCategories::CONFIG, "Cannot save invalid theme or empty name");
        return false;
    }
    
    if (!ensureThemeDirectory()) {
        LOG_ERROR(LogCategories::CONFIG, "Failed to create theme directory");
        return false;
    }
    
    QString filePath = getCustomThemePath(name);
    QFile file(filePath);
    
    if (!file.open(QIODevice::WriteOnly)) {
        LOG_ERROR(LogCategories::CONFIG, QString("Failed to open theme file for writing: %1").arg(filePath));
        return false;
    }
    
    QJsonObject json = themeToJson(theme);
    QJsonDocument doc(json);
    
    qint64 bytesWritten = file.write(doc.toJson());
    file.close();
    
    if (bytesWritten == -1) {
        LOG_ERROR(LogCategories::CONFIG, QString("Failed to write theme file: %1").arg(filePath));
        return false;
    }
    
    LOG_INFO(LogCategories::CONFIG, QString("Saved custom theme '%1' to %2").arg(name).arg(filePath));
    return true;
}

ThemeData ThemePersistence::loadCustomTheme(const QString& name)
{
    ThemeData theme;
    
    if (name.isEmpty()) {
        LOG_WARNING(LogCategories::CONFIG, "Cannot load theme with empty name");
        return theme;
    }
    
    QString filePath = getCustomThemePath(name);
    QFile file(filePath);
    
    if (!file.exists()) {
        LOG_WARNING(LogCategories::CONFIG, QString("Theme file does not exist: %1").arg(filePath));
        return theme;
    }
    
    if (!file.open(QIODevice::ReadOnly)) {
        LOG_ERROR(LogCategories::CONFIG, QString("Failed to open theme file for reading: %1").arg(filePath));
        
        // Attempt to recover from backup if available
        return attemptThemeRecovery(name);
    }
    
    QByteArray data = file.readAll();
    file.close();
    
    // Check for file corruption (empty or too small)
    if (data.isEmpty() || data.size() < 50) {
        LOG_ERROR(LogCategories::CONFIG, QString("Theme file appears corrupted (size: %1 bytes): %2").arg(data.size()).arg(filePath));
        return attemptThemeRecovery(name);
    }
    
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(data, &error);
    
    if (error.error != QJsonParseError::NoError) {
        LOG_ERROR(LogCategories::CONFIG, QString("Failed to parse theme JSON: %1").arg(error.errorString()));
        return attemptThemeRecovery(name);
    }
    
    if (!doc.isObject()) {
        LOG_ERROR(LogCategories::CONFIG, "Theme JSON is not an object");
        return attemptThemeRecovery(name);
    }
    
    QJsonObject json = doc.object();
    if (!validateThemeJson(json)) {
        LOG_ERROR(LogCategories::CONFIG, "Theme JSON validation failed");
        return attemptThemeRecovery(name);
    }
    
    theme = themeFromJson(json);
    
    // Validate the loaded theme data
    if (!theme.isValid()) {
        LOG_ERROR(LogCategories::CONFIG, QString("Loaded theme data is invalid for theme: %1").arg(name));
        return attemptThemeRecovery(name);
    }
    
    LOG_INFO(LogCategories::CONFIG, QString("Loaded custom theme '%1' from %2").arg(name).arg(filePath));
    
    return theme;
}

ThemeData ThemePersistence::attemptThemeRecovery(const QString& name)
{
    LOG_INFO(LogCategories::CONFIG, QString("Attempting recovery for corrupted theme: %1").arg(name));
    
    // Try to find backup files
    QString dataPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir dataDir(dataPath);
    
    QStringList filters;
    filters << "themes_auto_backup_*.json" << "themes_backup_*.json";
    
    QFileInfoList backupFiles = dataDir.entryInfoList(filters, QDir::Files, QDir::Time);
    
    // Try each backup file (newest first)
    for (const QFileInfo& backupInfo : backupFiles) {
        QFile backupFile(backupInfo.absoluteFilePath());
        if (backupFile.open(QIODevice::ReadOnly)) {
            QByteArray data = backupFile.readAll();
            backupFile.close();
            
            QJsonParseError error;
            QJsonDocument doc = QJsonDocument::fromJson(data, &error);
            
            if (error.error == QJsonParseError::NoError && doc.isObject()) {
                QJsonObject backupData = doc.object();
                
                if (backupData.contains("themes") && backupData["themes"].isArray()) {
                    QJsonArray themesArray = backupData["themes"].toArray();
                    
                    // Look for the specific theme in the backup
                    for (const QJsonValue& themeValue : themesArray) {
                        if (themeValue.isObject()) {
                            QJsonObject themeJson = themeValue.toObject();
                            if (themeJson["name"].toString() == name && validateThemeJson(themeJson)) {
                                ThemeData recoveredTheme = themeFromJson(themeJson);
                                if (recoveredTheme.isValid()) {
                                    LOG_INFO(LogCategories::CONFIG, QString("Successfully recovered theme '%1' from backup: %2")
                                             .arg(name).arg(backupInfo.fileName()));
                                    
                                    // Save the recovered theme
                                    if (saveCustomTheme(name, recoveredTheme)) {
                                        return recoveredTheme;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    LOG_WARNING(LogCategories::CONFIG, QString("Failed to recover theme '%1' from any backup").arg(name));
    return ThemeData(); // Return invalid theme
}

QStringList ThemePersistence::getCustomThemeNames()
{
    QStringList names;
    
    QString themePath = getThemeStoragePath();
    QDir themeDir(themePath);
    
    if (!themeDir.exists()) {
        return names;
    }
    
    QStringList filters;
    filters << QString("*%1").arg(THEME_FILE_EXTENSION);
    
    QFileInfoList files = themeDir.entryInfoList(filters, QDir::Files);
    for (const QFileInfo& fileInfo : files) {
        QString baseName = fileInfo.baseName();
        names.append(baseName);
    }
    
    names.sort();
    return names;
}

bool ThemePersistence::deleteCustomTheme(const QString& name)
{
    if (name.isEmpty()) {
        return false;
    }
    
    QString filePath = getCustomThemePath(name);
    QFile file(filePath);
    
    if (!file.exists()) {
        LOG_WARNING(LogCategories::CONFIG, QString("Theme file does not exist: %1").arg(filePath));
        return false;
    }
    
    bool success = file.remove();
    if (success) {
        LOG_INFO(LogCategories::CONFIG, QString("Deleted custom theme '%1'").arg(name));
    } else {
        LOG_ERROR(LogCategories::CONFIG, QString("Failed to delete custom theme '%1'").arg(name));
    }
    
    return success;
}

bool ThemePersistence::customThemeExists(const QString& name)
{
    if (name.isEmpty()) {
        return false;
    }
    
    QString filePath = getCustomThemePath(name);
    return QFile::exists(filePath);
}

bool ThemePersistence::backupThemes(const QString& backupPath)
{
    QString targetPath = backupPath;
    if (targetPath.isEmpty()) {
        QString dataPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
        targetPath = QDir(dataPath).absoluteFilePath(getBackupFileName());
    }
    
    QJsonObject backupData;
    backupData["version"] = THEME_VERSION;
    backupData["created"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    
    // Backup all custom themes
    QJsonArray themesArray;
    QStringList themeNames = getCustomThemeNames();
    
    for (const QString& name : themeNames) {
        ThemeData theme = loadCustomTheme(name);
        if (theme.isValid()) {
            QJsonObject themeJson = themeToJson(theme);
            themesArray.append(themeJson);
        }
    }
    
    backupData["themes"] = themesArray;
    
    // Backup current theme preference
    auto preference = loadThemePreference();
    QJsonObject preferenceJson;
    preferenceJson["theme"] = static_cast<int>(preference.first);
    preferenceJson["customName"] = preference.second;
    backupData["preference"] = preferenceJson;
    
    // Write backup file
    QFile backupFile(targetPath);
    if (!backupFile.open(QIODevice::WriteOnly)) {
        LOG_ERROR(LogCategories::CONFIG, QString("Failed to create backup file: %1").arg(targetPath));
        return false;
    }
    
    QJsonDocument doc(backupData);
    qint64 bytesWritten = backupFile.write(doc.toJson());
    backupFile.close();
    
    if (bytesWritten == -1) {
        LOG_ERROR(LogCategories::CONFIG, QString("Failed to write backup file: %1").arg(targetPath));
        return false;
    }
    
    LOG_INFO(LogCategories::CONFIG, QString("Themes backed up to: %1 (%2 themes)")
             .arg(targetPath).arg(themeNames.size()));
    return true;
}

bool ThemePersistence::restoreThemes(const QString& backupPath)
{
    if (backupPath.isEmpty() || !QFile::exists(backupPath)) {
        LOG_ERROR(LogCategories::CONFIG, QString("Backup file does not exist: %1").arg(backupPath));
        return false;
    }
    
    QFile backupFile(backupPath);
    if (!backupFile.open(QIODevice::ReadOnly)) {
        LOG_ERROR(LogCategories::CONFIG, QString("Failed to open backup file: %1").arg(backupPath));
        return false;
    }
    
    QByteArray data = backupFile.readAll();
    backupFile.close();
    
    QJsonParseError error;
    QJsonDocument doc = QJsonDocument::fromJson(data, &error);
    
    if (error.error != QJsonParseError::NoError) {
        LOG_ERROR(LogCategories::CONFIG, QString("Failed to parse backup JSON: %1").arg(error.errorString()));
        return false;
    }
    
    QJsonObject backupData = doc.object();
    
    // Validate backup format
    if (!backupData.contains("themes") || !backupData["themes"].isArray()) {
        LOG_ERROR(LogCategories::CONFIG, "Invalid backup format: missing themes array");
        return false;
    }
    
    // Restore themes
    QJsonArray themesArray = backupData["themes"].toArray();
    int restoredCount = 0;
    
    for (const QJsonValue& themeValue : themesArray) {
        if (themeValue.isObject()) {
            QJsonObject themeJson = themeValue.toObject();
            if (validateThemeJson(themeJson)) {
                ThemeData theme = themeFromJson(themeJson);
                if (saveCustomTheme(theme.name, theme)) {
                    restoredCount++;
                } else {
                    LOG_WARNING(LogCategories::CONFIG, QString("Failed to restore theme: %1").arg(theme.name));
                }
            }
        }
    }
    
    // Restore theme preference if available
    if (backupData.contains("preference") && backupData["preference"].isObject()) {
        QJsonObject preferenceJson = backupData["preference"].toObject();
        ThemeManager::Theme theme = static_cast<ThemeManager::Theme>(preferenceJson["theme"].toInt());
        QString customName = preferenceJson["customName"].toString();
        
        // Only restore if the custom theme exists (was successfully restored)
        if (theme != ThemeManager::Custom || customThemeExists(customName)) {
            saveThemePreference(theme, customName);
            LOG_INFO(LogCategories::CONFIG, "Theme preference restored");
        }
    }
    
    LOG_INFO(LogCategories::CONFIG, QString("Restored %1 themes from backup: %2")
             .arg(restoredCount).arg(backupPath));
    
    return restoredCount > 0;
}

bool ThemePersistence::migrateFromOldSettings()
{
    QString currentVersion = getThemeVersion();
    
    // If already at current version, no migration needed
    if (currentVersion == THEME_VERSION) {
        return true;
    }
    
    LOG_INFO(LogCategories::CONFIG, QString("Migrating theme settings from version %1 to %2")
             .arg(currentVersion).arg(THEME_VERSION));
    
    QSettings settings;
    bool migrationPerformed = false;
    
    // Migration from version 0.0 (no version) to 1.0
    if (currentVersion == "0.0" || currentVersion.isEmpty()) {
        // Check for old theme settings format
        QString oldTheme = settings.value("general/theme", "").toString();
        
        if (!oldTheme.isEmpty() && oldTheme != "system") {
            // Migrate old theme preference to new format
            ThemeManager::Theme theme = ThemeManager::SystemDefault;
            if (oldTheme == "light") {
                theme = ThemeManager::Light;
            } else if (oldTheme == "dark") {
                theme = ThemeManager::Dark;
            }
            
            // Save in new format
            saveThemePreference(theme, QString());
            
            migrationPerformed = true;
            LOG_INFO(LogCategories::CONFIG, QString("Migrated theme preference: %1").arg(oldTheme));
        }
        
        // Always remove old setting to avoid conflicts
        if (settings.contains("general/theme")) {
            settings.remove("general/theme");
            LOG_DEBUG(LogCategories::CONFIG, "Removed old theme setting key");
        }
        
        // Create backup of existing custom themes if any exist
        QStringList existingThemes = getCustomThemeNames();
        if (!existingThemes.isEmpty()) {
            QString backupPath = QString("%1/themes_migration_backup_%2.json")
                               .arg(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation))
                               .arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"));
            
            if (backupThemes(backupPath)) {
                LOG_INFO(LogCategories::CONFIG, QString("Created migration backup: %1").arg(backupPath));
            }
        }
    }
    
    // Set new version
    setThemeVersion(THEME_VERSION);
    
    if (migrationPerformed) {
        LOG_INFO(LogCategories::CONFIG, "Theme settings migration completed successfully");
    } else {
        LOG_DEBUG(LogCategories::CONFIG, "No theme settings migration required");
    }
    
    return true;
}

QString ThemePersistence::getThemeVersion()
{
    QSettings settings;
    return settings.value("theme/version", "0.0").toString();
}

void ThemePersistence::setThemeVersion(const QString& version)
{
    QSettings settings;
    settings.setValue("theme/version", version);
}

QString ThemePersistence::getThemeStoragePath()
{
    QString dataPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    return QDir(dataPath).absoluteFilePath("themes");
}

QString ThemePersistence::getCustomThemePath(const QString& name)
{
    QString themePath = getThemeStoragePath();
    return QDir(themePath).absoluteFilePath(name + THEME_FILE_EXTENSION);
}

QString ThemePersistence::getPreferencesKey()
{
    return QString("%1/current_theme").arg(PREFERENCES_GROUP);
}

QString ThemePersistence::getBackupFileName()
{
    return QString("themes_backup_%1.json").arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"));
}

QJsonObject ThemePersistence::themeToJson(const ThemeData& theme)
{
    QJsonObject json;
    
    // Basic info
    json["name"] = theme.name;
    json["description"] = theme.description;
    json["created"] = theme.created.toString(Qt::ISODate);
    json["modified"] = theme.modified.toString(Qt::ISODate);
    json["version"] = THEME_VERSION;
    
    // Colors
    json["colors"] = colorSchemeToJson(theme.colors);
    
    // Typography
    json["typography"] = typographyToJson(theme.typography);
    
    // Spacing
    json["spacing"] = spacingToJson(theme.spacing);
    
    return json;
}

ThemeData ThemePersistence::themeFromJson(const QJsonObject& json)
{
    ThemeData theme;
    
    // Basic info
    theme.name = json["name"].toString();
    theme.description = json["description"].toString();
    theme.created = QDateTime::fromString(json["created"].toString(), Qt::ISODate);
    theme.modified = QDateTime::fromString(json["modified"].toString(), Qt::ISODate);
    
    // Colors
    if (json.contains("colors") && json["colors"].isObject()) {
        theme.colors = colorSchemeFromJson(json["colors"].toObject());
    }
    
    // Typography
    if (json.contains("typography") && json["typography"].isObject()) {
        theme.typography = typographyFromJson(json["typography"].toObject());
    }
    
    // Spacing
    if (json.contains("spacing") && json["spacing"].isObject()) {
        theme.spacing = spacingFromJson(json["spacing"].toObject());
    }
    
    return theme;
}

QJsonObject ThemePersistence::colorSchemeToJson(const ThemeData::ColorScheme& colors)
{
    QJsonObject json;
    
    json["background"] = colors.background.name();
    json["foreground"] = colors.foreground.name();
    json["accent"] = colors.accent.name();
    json["border"] = colors.border.name();
    json["hover"] = colors.hover.name();
    json["disabled"] = colors.disabled.name();
    json["success"] = colors.success.name();
    json["warning"] = colors.warning.name();
    json["error"] = colors.error.name();
    json["info"] = colors.info.name();
    
    return json;
}

ThemeData::ColorScheme ThemePersistence::colorSchemeFromJson(const QJsonObject& json)
{
    ThemeData::ColorScheme colors;
    
    colors.background = QColor(json["background"].toString());
    colors.foreground = QColor(json["foreground"].toString());
    colors.accent = QColor(json["accent"].toString());
    colors.border = QColor(json["border"].toString());
    colors.hover = QColor(json["hover"].toString());
    colors.disabled = QColor(json["disabled"].toString());
    colors.success = QColor(json["success"].toString());
    colors.warning = QColor(json["warning"].toString());
    colors.error = QColor(json["error"].toString());
    colors.info = QColor(json["info"].toString());
    
    return colors;
}

QJsonObject ThemePersistence::typographyToJson(const ThemeData::Typography& typography)
{
    QJsonObject json;
    
    json["fontFamily"] = typography.fontFamily;
    json["baseFontSize"] = typography.baseFontSize;
    json["titleFontSize"] = typography.titleFontSize;
    json["smallFontSize"] = typography.smallFontSize;
    json["boldTitles"] = typography.boldTitles;
    
    return json;
}

ThemeData::Typography ThemePersistence::typographyFromJson(const QJsonObject& json)
{
    ThemeData::Typography typography;
    
    typography.fontFamily = json["fontFamily"].toString("Segoe UI, Ubuntu, sans-serif");
    typography.baseFontSize = json["baseFontSize"].toInt(9);
    typography.titleFontSize = json["titleFontSize"].toInt(11);
    typography.smallFontSize = json["smallFontSize"].toInt(8);
    typography.boldTitles = json["boldTitles"].toBool(true);
    
    return typography;
}

QJsonObject ThemePersistence::spacingToJson(const ThemeData::Spacing& spacing)
{
    QJsonObject json;
    
    json["padding"] = spacing.padding;
    json["margin"] = spacing.margin;
    json["borderRadius"] = spacing.borderRadius;
    json["borderWidth"] = spacing.borderWidth;
    
    return json;
}

ThemeData::Spacing ThemePersistence::spacingFromJson(const QJsonObject& json)
{
    ThemeData::Spacing spacing;
    
    spacing.padding = json["padding"].toInt(8);
    spacing.margin = json["margin"].toInt(4);
    spacing.borderRadius = json["borderRadius"].toInt(4);
    spacing.borderWidth = json["borderWidth"].toInt(1);
    
    return spacing;
}

bool ThemePersistence::validateThemeJson(const QJsonObject& json)
{
    // Check required fields
    QStringList requiredFields = {"name", "colors", "typography", "spacing"};
    
    for (const QString& field : requiredFields) {
        if (!json.contains(field)) {
            LOG_ERROR(LogCategories::CONFIG, QString("Theme JSON missing required field: %1").arg(field));
            return false;
        }
    }
    
    // Validate colors object
    if (!json["colors"].isObject()) {
        LOG_ERROR(LogCategories::CONFIG, "Theme colors field is not an object");
        return false;
    }
    
    QJsonObject colors = json["colors"].toObject();
    QStringList requiredColors = {"background", "foreground", "accent", "border"};
    
    for (const QString& color : requiredColors) {
        if (!colors.contains(color)) {
            LOG_ERROR(LogCategories::CONFIG, QString("Theme colors missing required color: %1").arg(color));
            return false;
        }
        
        QColor testColor(colors[color].toString());
        if (!testColor.isValid()) {
            LOG_ERROR(LogCategories::CONFIG, QString("Invalid color value for %1: %2").arg(color).arg(colors[color].toString()));
            return false;
        }
    }
    
    return true;
}

bool ThemePersistence::ensureThemeDirectory()
{
    QString themePath = getThemeStoragePath();
    QDir themeDir;
    
    if (!themeDir.exists(themePath)) {
        bool success = themeDir.mkpath(themePath);
        if (!success) {
            LOG_ERROR(LogCategories::CONFIG, QString("Failed to create theme directory: %1").arg(themePath));
            return false;
        }
        LOG_INFO(LogCategories::CONFIG, QString("Created theme directory: %1").arg(themePath));
    }
    
    // Clean up old backup files (keep only last 7 days)
    cleanupOldBackups();
    
    return true;
}

void ThemePersistence::cleanupOldBackups()
{
    QString dataPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir dataDir(dataPath);
    
    if (!dataDir.exists()) {
        return;
    }
    
    // Find backup files
    QStringList filters;
    filters << "themes_auto_backup_*.json" << "themes_backup_*.json" << "themes_migration_backup_*.json";
    
    QFileInfoList backupFiles = dataDir.entryInfoList(filters, QDir::Files);
    QDateTime cutoffDate = QDateTime::currentDateTime().addDays(-7);
    
    int deletedCount = 0;
    for (const QFileInfo& fileInfo : backupFiles) {
        if (fileInfo.lastModified() < cutoffDate) {
            if (QFile::remove(fileInfo.absoluteFilePath())) {
                deletedCount++;
                LOG_DEBUG(LogCategories::CONFIG, QString("Deleted old backup: %1").arg(fileInfo.fileName()));
            }
        }
    }
    
    if (deletedCount > 0) {
        LOG_INFO(LogCategories::CONFIG, QString("Cleaned up %1 old backup files").arg(deletedCount));
    }
}