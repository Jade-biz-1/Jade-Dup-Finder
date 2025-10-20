#ifndef THEME_PERSISTENCE_H
#define THEME_PERSISTENCE_H

#include <QString>
#include <QStringList>
#include <QJsonObject>
#include <QJsonDocument>
#include <QPair>
#include "theme_manager.h"

class ThemePersistence
{
public:
    static bool saveThemePreference(ThemeManager::Theme theme, const QString& customName = QString());
    static QPair<ThemeManager::Theme, QString> loadThemePreference();
    static bool saveCustomTheme(const QString& name, const ThemeData& theme);
    static ThemeData loadCustomTheme(const QString& name);
    static QStringList getCustomThemeNames();
    static bool deleteCustomTheme(const QString& name);
    static bool customThemeExists(const QString& name);
    
    // Backup and restore
    static bool backupThemes(const QString& backupPath);
    static bool restoreThemes(const QString& backupPath);
    
    // Migration helpers
    static bool migrateFromOldSettings();
    static QString getThemeVersion();
    static void setThemeVersion(const QString& version);
    
    // JSON conversion (made public for theme editor)
    static QJsonObject themeToJson(const ThemeData& theme);
    static ThemeData themeFromJson(const QJsonObject& json);
    static bool validateThemeJson(const QJsonObject& json);
    
private:
    static QString getThemeStoragePath();
    static QString getCustomThemePath(const QString& name);
    static QString getPreferencesKey();
    static QString getBackupFileName();
    

    static QJsonObject colorSchemeToJson(const ThemeData::ColorScheme& colors);
    static ThemeData::ColorScheme colorSchemeFromJson(const QJsonObject& json);
    static QJsonObject typographyToJson(const ThemeData::Typography& typography);
    static ThemeData::Typography typographyFromJson(const QJsonObject& json);
    static QJsonObject spacingToJson(const ThemeData::Spacing& spacing);
    static ThemeData::Spacing spacingFromJson(const QJsonObject& json);
    

    static bool ensureThemeDirectory();
    static void cleanupOldBackups();
    static ThemeData attemptThemeRecovery(const QString& name);
    
    // Constants
    static const QString THEME_FILE_EXTENSION;
    static const QString THEME_VERSION;
    static const QString PREFERENCES_GROUP;
};

#endif // THEME_PERSISTENCE_H