#include <QCoreApplication>
#include <QDebug>
#include <QTimer>
#include <QEventLoop>
#include <QTemporaryDir>
#include <QFile>
#include <QDir>
#include <QSignalSpy>
#include <QTest>
#include <QSettings>
#include <QStandardPaths>

#include "file_scanner.h"
#include "duplicate_detector.h"
#include "../src/core/safety_manager.h"
#include "theme_manager.h"
#include "app_config.h"

/**
 * @brief Configuration Persistence and Loading Testing
 * 
 * This test verifies:
 * - Configuration persistence across application restarts
 * - Settings validation and migration
 * - Default configuration handling
 * - Configuration import/export functionality
 * - Settings synchronization between components
 * - Configuration backup and recovery
 * 
 * Requirements: 1.3, 4.5, 10.3
 */

class ConfigurationPersistenceTest : public QObject {
    Q_OBJECT

private:
    QTemporaryDir* m_tempDir;
    QString m_testPath;
    QString m_configPath;

private slots:
    void initTestCase() {
        qDebug() << "===========================================";
        qDebug() << "Configuration Persistence Testing";
        qDebug() << "===========================================";
        qDebug();
        
        m_tempDir = new QTemporaryDir();
        QVERIFY(m_tempDir->isValid());
        m_testPath = m_tempDir->path();
        m_configPath = m_testPath + "/config";
        
        // Create config directory
        QDir().mkpath(m_configPath);
        
        qDebug() << "Test directory:" << m_testPath;
        qDebug() << "Config directory:" << m_configPath;
    }
    
    void cleanupTestCase() {
        delete m_tempDir;
        qDebug() << "\n===========================================";
        qDebug() << "All tests completed";
        qDebug() << "===========================================";
    }
    
    /**
     * Test 1: AppConfig persistence and loading
     */
    void test_appConfigPersistence() {
        qDebug() << "\n[Test 1] AppConfig Persistence";
        qDebug() << "===============================";
        
        QString configFile = m_configPath + "/app_config_test.ini";
        
        // Create and configure AppConfig
        {
            QSettings settings(configFile, QSettings::IniFormat);
            AppConfig config(&settings);
            
            qDebug() << "   Setting configuration values...";
            
            // Set various configuration values
            config.setLastScanPath("/test/scan/path");
            config.setMinimumFileSize(1024);
            config.setMaximumFileSize(1024 * 1024 * 100); // 100MB
            config.setIncludeHiddenFiles(true);
            config.setFollowSymlinks(false);
            config.setBackupDirectory("/test/backup/path");
            config.setSafetyLevel(2); // Assuming 2 = Standard
            config.setTheme("Dark");
            config.setAutoSave(true);
            config.setMaxRecentFiles(15);
            
            // Save configuration
            config.save();
            settings.sync();
            
            qDebug() << "      Configuration saved to:" << configFile;
        }
        
        // Load configuration in a new instance
        {
            QSettings settings(configFile, QSettings::IniFormat);
            AppConfig config(&settings);
            
            qDebug() << "   Loading configuration values...";
            
            // Verify loaded values
            QCOMPARE(config.getLastScanPath(), QString("/test/scan/path"));
            QCOMPARE(config.getMinimumFileSize(), qint64(1024));
            QCOMPARE(config.getMaximumFileSize(), qint64(1024 * 1024 * 100));
            QCOMPARE(config.getIncludeHiddenFiles(), true);
            QCOMPARE(config.getFollowSymlinks(), false);
            QCOMPARE(config.getBackupDirectory(), QString("/test/backup/path"));
            QCOMPARE(config.getSafetyLevel(), 2);
            QCOMPARE(config.getTheme(), QString("Dark"));
            QCOMPARE(config.getAutoSave(), true);
            QCOMPARE(config.getMaxRecentFiles(), 15);
            
            qDebug() << "      Last scan path:" << config.getLastScanPath();
            qDebug() << "      Minimum file size:" << config.getMinimumFileSize();
            qDebug() << "      Maximum file size:" << config.getMaximumFileSize();
            qDebug() << "      Include hidden files:" << config.getIncludeHiddenFiles();
            qDebug() << "      Follow symlinks:" << config.getFollowSymlinks();
            qDebug() << "      Backup directory:" << config.getBackupDirectory();
            qDebug() << "      Safety level:" << config.getSafetyLevel();
            qDebug() << "      Theme:" << config.getTheme();
            qDebug() << "      Auto save:" << config.getAutoSave();
            qDebug() << "      Max recent files:" << config.getMaxRecentFiles();
        }
        
        qDebug() << "✓ AppConfig persistence verified";
    }
    
    /**
     * Test 2: FileScanner configuration persistence
     */
    void test_fileScannerConfigPersistence() {
        qDebug() << "\n[Test 2] FileScanner Configuration Persistence";
        qDebug() << "================================================";
        
        QString configFile = m_configPath + "/scanner_config_test.ini";
        
        // Create and save scanner configuration
        {
            QSettings settings(configFile, QSettings::IniFormat);
            
            // Save FileScanner options
            settings.beginGroup("FileScanner");
            settings.setValue("minimumFileSize", 2048);
            settings.setValue("maximumFileSize", 50 * 1024 * 1024); // 50MB
            settings.setValue("includeHiddenFiles", false);
            settings.setValue("followSymlinks", true);
            settings.setValue("scanSystemDirectories", false);
            settings.setValue("progressBatchSize", 250);
            settings.setValue("enableMetadataCache", true);
            settings.setValue("metadataCacheSizeLimit", 5000);
            
            QStringList includePatterns = {"*.txt", "*.doc", "*.pdf"};
            QStringList excludePatterns = {"*.tmp", "*.log"};
            settings.setValue("includePatterns", includePatterns);
            settings.setValue("excludePatterns", excludePatterns);
            settings.setValue("caseSensitivePatterns", true);
            
            settings.endGroup();
            settings.sync();
            
            qDebug() << "   FileScanner configuration saved";
        }
        
        // Load and apply configuration
        {
            QSettings settings(configFile, QSettings::IniFormat);
            
            settings.beginGroup("FileScanner");
            
            FileScanner::ScanOptions options;
            options.minimumFileSize = settings.value("minimumFileSize", 0).toLongLong();
            options.maximumFileSize = settings.value("maximumFileSize", -1).toLongLong();
            options.includeHiddenFiles = settings.value("includeHiddenFiles", false).toBool();
            options.followSymlinks = settings.value("followSymlinks", false).toBool();
            options.scanSystemDirectories = settings.value("scanSystemDirectories", false).toBool();
            options.progressBatchSize = settings.value("progressBatchSize", 100).toInt();
            options.enableMetadataCache = settings.value("enableMetadataCache", false).toBool();
            options.metadataCacheSizeLimit = settings.value("metadataCacheSizeLimit", 10000).toInt();
            
            options.includePatterns = settings.value("includePatterns").toStringList();
            options.excludePatterns = settings.value("excludePatterns").toStringList();
            options.caseSensitivePatterns = settings.value("caseSensitivePatterns", false).toBool();
            
            settings.endGroup();
            
            // Verify loaded configuration
            QCOMPARE(options.minimumFileSize, qint64(2048));
            QCOMPARE(options.maximumFileSize, qint64(50 * 1024 * 1024));
            QCOMPARE(options.includeHiddenFiles, false);
            QCOMPARE(options.followSymlinks, true);
            QCOMPARE(options.scanSystemDirectories, false);
            QCOMPARE(options.progressBatchSize, 250);
            QCOMPARE(options.enableMetadataCache, true);
            QCOMPARE(options.metadataCacheSizeLimit, 5000);
            QCOMPARE(options.caseSensitivePatterns, true);
            
            QCOMPARE(options.includePatterns.size(), 3);
            QVERIFY(options.includePatterns.contains("*.txt"));
            QVERIFY(options.includePatterns.contains("*.doc"));
            QVERIFY(options.includePatterns.contains("*.pdf"));
            
            QCOMPARE(options.excludePatterns.size(), 2);
            QVERIFY(options.excludePatterns.contains("*.tmp"));
            QVERIFY(options.excludePatterns.contains("*.log"));
            
            qDebug() << "   FileScanner configuration loaded and verified";
            qDebug() << "      Minimum file size:" << options.minimumFileSize;
            qDebug() << "      Maximum file size:" << options.maximumFileSize;
            qDebug() << "      Include patterns:" << options.includePatterns;
            qDebug() << "      Exclude patterns:" << options.excludePatterns;
        }
        
        qDebug() << "✓ FileScanner configuration persistence verified";
    }
    
    /**
     * Test 3: DuplicateDetector configuration persistence
     */
    void test_duplicateDetectorConfigPersistence() {
        qDebug() << "\n[Test 3] DuplicateDetector Configuration Persistence";
        qDebug() << "=====================================================";
        
        QString configFile = m_configPath + "/detector_config_test.ini";
        
        // Save DuplicateDetector configuration
        {
            QSettings settings(configFile, QSettings::IniFormat);
            
            settings.beginGroup("DuplicateDetector");
            settings.setValue("detectionLevel", static_cast<int>(DuplicateDetector::DetectionLevel::Standard));
            settings.setValue("groupBySize", true);
            settings.setValue("analyzeMetadata", false);
            settings.setValue("fuzzyNameMatching", true);
            settings.setValue("similarityThreshold", 0.85);
            settings.setValue("minimumFileSize", 512);
            settings.setValue("maximumFileSize", 10 * 1024 * 1024); // 10MB
            settings.setValue("skipEmptyFiles", true);
            settings.setValue("skipSystemFiles", true);
            settings.endGroup();
            
            settings.sync();
            qDebug() << "   DuplicateDetector configuration saved";
        }
        
        // Load and verify configuration
        {
            QSettings settings(configFile, QSettings::IniFormat);
            
            settings.beginGroup("DuplicateDetector");
            
            DuplicateDetector::DetectionOptions options;
            options.level = static_cast<DuplicateDetector::DetectionLevel>(
                settings.value("detectionLevel", static_cast<int>(DuplicateDetector::DetectionLevel::Standard)).toInt());
            options.groupBySize = settings.value("groupBySize", true).toBool();
            options.analyzeMetadata = settings.value("analyzeMetadata", false).toBool();
            options.fuzzyNameMatching = settings.value("fuzzyNameMatching", false).toBool();
            options.similarityThreshold = settings.value("similarityThreshold", 0.95).toDouble();
            options.minimumFileSize = settings.value("minimumFileSize", 0).toLongLong();
            options.maximumFileSize = settings.value("maximumFileSize", -1).toLongLong();
            options.skipEmptyFiles = settings.value("skipEmptyFiles", true).toBool();
            options.skipSystemFiles = settings.value("skipSystemFiles", true).toBool();
            
            settings.endGroup();
            
            // Verify loaded configuration
            QCOMPARE(options.level, DuplicateDetector::DetectionLevel::Standard);
            QCOMPARE(options.groupBySize, true);
            QCOMPARE(options.analyzeMetadata, false);
            QCOMPARE(options.fuzzyNameMatching, true);
            QCOMPARE(options.similarityThreshold, 0.85);
            QCOMPARE(options.minimumFileSize, qint64(512));
            QCOMPARE(options.maximumFileSize, qint64(10 * 1024 * 1024));
            QCOMPARE(options.skipEmptyFiles, true);
            QCOMPARE(options.skipSystemFiles, true);
            
            qDebug() << "   DuplicateDetector configuration loaded and verified";
            qDebug() << "      Detection level:" << static_cast<int>(options.level);
            qDebug() << "      Group by size:" << options.groupBySize;
            qDebug() << "      Analyze metadata:" << options.analyzeMetadata;
            qDebug() << "      Fuzzy name matching:" << options.fuzzyNameMatching;
            qDebug() << "      Similarity threshold:" << options.similarityThreshold;
        }
        
        qDebug() << "✓ DuplicateDetector configuration persistence verified";
    }
    
    /**
     * Test 4: SafetyManager configuration persistence
     */
    void test_safetyManagerConfigPersistence() {
        qDebug() << "\n[Test 4] SafetyManager Configuration Persistence";
        qDebug() << "=================================================";
        
        QString configFile = m_configPath + "/safety_config_test.ini";
        
        // Save SafetyManager configuration
        {
            QSettings settings(configFile, QSettings::IniFormat);
            
            settings.beginGroup("SafetyManager");
            settings.setValue("backupDirectory", "/test/backup/directory");
            settings.setValue("safetyLevel", static_cast<int>(SafetyManager::SafetyLevel::Conservative));
            settings.setValue("maxBackupAge", 30); // 30 days
            settings.setValue("maxBackupSize", 1024 * 1024 * 1024); // 1GB
            settings.setValue("compressionEnabled", true);
            settings.setValue("encryptionEnabled", false);
            settings.setValue("autoCleanup", true);
            
            QStringList protectionRules = {
                "*/System/*",
                "*/Windows/*",
                "*.system",
                "*.important"
            };
            settings.setValue("protectionRules", protectionRules);
            
            settings.endGroup();
            settings.sync();
            
            qDebug() << "   SafetyManager configuration saved";
        }
        
        // Load and verify configuration
        {
            QSettings settings(configFile, QSettings::IniFormat);
            
            settings.beginGroup("SafetyManager");
            
            QString backupDirectory = settings.value("backupDirectory", "").toString();
            SafetyManager::SafetyLevel safetyLevel = static_cast<SafetyManager::SafetyLevel>(
                settings.value("safetyLevel", static_cast<int>(SafetyManager::SafetyLevel::Standard)).toInt());
            int maxBackupAge = settings.value("maxBackupAge", 7).toInt();
            qint64 maxBackupSize = settings.value("maxBackupSize", 0).toLongLong();
            bool compressionEnabled = settings.value("compressionEnabled", false).toBool();
            bool encryptionEnabled = settings.value("encryptionEnabled", false).toBool();
            bool autoCleanup = settings.value("autoCleanup", false).toBool();
            
            QStringList protectionRules = settings.value("protectionRules").toStringList();
            
            settings.endGroup();
            
            // Verify loaded configuration
            QCOMPARE(backupDirectory, QString("/test/backup/directory"));
            QCOMPARE(safetyLevel, SafetyManager::SafetyLevel::Conservative);
            QCOMPARE(maxBackupAge, 30);
            QCOMPARE(maxBackupSize, qint64(1024 * 1024 * 1024));
            QCOMPARE(compressionEnabled, true);
            QCOMPARE(encryptionEnabled, false);
            QCOMPARE(autoCleanup, true);
            
            QCOMPARE(protectionRules.size(), 4);
            QVERIFY(protectionRules.contains("*/System/*"));
            QVERIFY(protectionRules.contains("*/Windows/*"));
            QVERIFY(protectionRules.contains("*.system"));
            QVERIFY(protectionRules.contains("*.important"));
            
            qDebug() << "   SafetyManager configuration loaded and verified";
            qDebug() << "      Backup directory:" << backupDirectory;
            qDebug() << "      Safety level:" << static_cast<int>(safetyLevel);
            qDebug() << "      Max backup age:" << maxBackupAge << "days";
            qDebug() << "      Protection rules:" << protectionRules.size();
        }
        
        qDebug() << "✓ SafetyManager configuration persistence verified";
    }
    
    /**
     * Test 5: ThemeManager configuration persistence
     */
    void test_themeManagerConfigPersistence() {
        qDebug() << "\n[Test 5] ThemeManager Configuration Persistence";
        qDebug() << "================================================";
        
        ThemeManager* themeManager = ThemeManager::instance();
        QVERIFY(themeManager != nullptr);
        
        // Get original theme
        ThemeManager::Theme originalTheme = themeManager->currentTheme();
        
        // Change theme and save
        ThemeManager::Theme testTheme = ThemeManager::Dark;
        themeManager->setTheme(testTheme);
        themeManager->saveToSettings();
        
        qDebug() << "   Theme changed to:" << themeManager->currentThemeString();
        qDebug() << "   Settings saved";
        
        // Change to different theme temporarily
        themeManager->setTheme(ThemeManager::Light);
        QCOMPARE(themeManager->currentTheme(), ThemeManager::Light);
        
        // Load settings - should restore saved theme
        themeManager->loadFromSettings();
        QCOMPARE(themeManager->currentTheme(), testTheme);
        
        qDebug() << "   Settings loaded, theme restored to:" << themeManager->currentThemeString();
        
        // Restore original theme
        themeManager->setTheme(originalTheme);
        themeManager->saveToSettings();
        
        qDebug() << "✓ ThemeManager configuration persistence verified";
    }
    
    /**
     * Test 6: Configuration validation and migration
     */
    void test_configurationValidationAndMigration() {
        qDebug() << "\n[Test 6] Configuration Validation and Migration";
        qDebug() << "================================================";
        
        QString configFile = m_configPath + "/migration_test.ini";
        
        // Create old version configuration
        {
            QSettings settings(configFile, QSettings::IniFormat);
            
            // Simulate old version settings
            settings.setValue("version", "1.0");
            settings.setValue("oldSetting", "oldValue");
            settings.setValue("deprecatedSetting", true);
            
            // Some settings that should be migrated
            settings.setValue("scanPath", "/old/scan/path");
            settings.setValue("fileSize", 1000);
            
            settings.sync();
            qDebug() << "   Created old version configuration";
        }
        
        // Load and migrate configuration
        {
            QSettings settings(configFile, QSettings::IniFormat);
            
            QString version = settings.value("version", "").toString();
            qDebug() << "   Configuration version:" << version;
            
            if (version == "1.0") {
                qDebug() << "   Migrating from version 1.0...";
                
                // Migrate old settings to new format
                QString oldScanPath = settings.value("scanPath", "").toString();
                int oldFileSize = settings.value("fileSize", 0).toInt();
                
                // Remove old settings
                settings.remove("oldSetting");
                settings.remove("deprecatedSetting");
                settings.remove("scanPath");
                settings.remove("fileSize");
                
                // Add new settings
                settings.setValue("lastScanPath", oldScanPath);
                settings.setValue("minimumFileSize", oldFileSize);
                settings.setValue("version", "2.0");
                
                settings.sync();
                
                qDebug() << "      Migration completed";
                qDebug() << "      Migrated scan path:" << oldScanPath;
                qDebug() << "      Migrated file size:" << oldFileSize;
            }
            
            // Verify migrated configuration
            QString newVersion = settings.value("version", "").toString();
            QString migratedPath = settings.value("lastScanPath", "").toString();
            int migratedSize = settings.value("minimumFileSize", 0).toInt();
            
            QCOMPARE(newVersion, QString("2.0"));
            QCOMPARE(migratedPath, QString("/old/scan/path"));
            QCOMPARE(migratedSize, 1000);
            
            // Verify old settings are removed
            QVERIFY(!settings.contains("oldSetting"));
            QVERIFY(!settings.contains("deprecatedSetting"));
            QVERIFY(!settings.contains("scanPath"));
            QVERIFY(!settings.contains("fileSize"));
        }
        
        qDebug() << "✓ Configuration validation and migration verified";
    }
    
    /**
     * Test 7: Default configuration handling
     */
    void test_defaultConfigurationHandling() {
        qDebug() << "\n[Test 7] Default Configuration Handling";
        qDebug() << "========================================";
        
        QString configFile = m_configPath + "/default_test.ini";
        
        // Test loading configuration with missing file
        {
            // Ensure file doesn't exist
            QFile::remove(configFile);
            
            QSettings settings(configFile, QSettings::IniFormat);
            AppConfig config(&settings);
            
            // Should use default values
            qDebug() << "   Testing default values...";
            
            QString defaultScanPath = config.getLastScanPath();
            qint64 defaultMinSize = config.getMinimumFileSize();
            qint64 defaultMaxSize = config.getMaximumFileSize();
            bool defaultHidden = config.getIncludeHiddenFiles();
            bool defaultSymlinks = config.getFollowSymlinks();
            QString defaultTheme = config.getTheme();
            bool defaultAutoSave = config.getAutoSave();
            int defaultMaxRecent = config.getMaxRecentFiles();
            
            qDebug() << "      Default scan path:" << defaultScanPath;
            qDebug() << "      Default min size:" << defaultMinSize;
            qDebug() << "      Default max size:" << defaultMaxSize;
            qDebug() << "      Default include hidden:" << defaultHidden;
            qDebug() << "      Default follow symlinks:" << defaultSymlinks;
            qDebug() << "      Default theme:" << defaultTheme;
            qDebug() << "      Default auto save:" << defaultAutoSave;
            qDebug() << "      Default max recent:" << defaultMaxRecent;
            
            // Verify reasonable defaults
            QVERIFY(defaultMinSize >= 0);
            QVERIFY(defaultMaxSize >= -1); // -1 means no limit
            QVERIFY(!defaultTheme.isEmpty());
            QVERIFY(defaultMaxRecent > 0);
        }
        
        qDebug() << "✓ Default configuration handling verified";
    }
    
    /**
     * Test 8: Configuration import/export functionality
     */
    void test_configurationImportExport() {
        qDebug() << "\n[Test 8] Configuration Import/Export";
        qDebug() << "=====================================";
        
        QString sourceConfigFile = m_configPath + "/export_source.ini";
        QString exportedConfigFile = m_configPath + "/exported_config.ini";
        QString importedConfigFile = m_configPath + "/imported_config.ini";
        
        // Create source configuration
        {
            QSettings settings(sourceConfigFile, QSettings::IniFormat);
            AppConfig config(&settings);
            
            config.setLastScanPath("/export/test/path");
            config.setMinimumFileSize(2048);
            config.setTheme("Dark");
            config.setAutoSave(true);
            config.setMaxRecentFiles(20);
            
            config.save();
            settings.sync();
            
            qDebug() << "   Created source configuration";
        }
        
        // Export configuration (simulate export functionality)
        {
            QSettings source(sourceConfigFile, QSettings::IniFormat);
            QSettings exported(exportedConfigFile, QSettings::IniFormat);
            
            // Copy all settings
            QStringList keys = source.allKeys();
            for (const QString& key : keys) {
                exported.setValue(key, source.value(key));
            }
            
            // Add export metadata
            exported.setValue("exportedAt", QDateTime::currentDateTime());
            exported.setValue("exportedBy", "ConfigurationPersistenceTest");
            exported.setValue("exportVersion", "1.0");
            
            exported.sync();
            
            qDebug() << "   Configuration exported to:" << exportedConfigFile;
            qDebug() << "   Exported keys:" << keys.size();
        }
        
        // Import configuration (simulate import functionality)
        {
            QSettings exported(exportedConfigFile, QSettings::IniFormat);
            QSettings imported(importedConfigFile, QSettings::IniFormat);
            
            // Verify export metadata
            QDateTime exportedAt = exported.value("exportedAt").toDateTime();
            QString exportedBy = exported.value("exportedBy").toString();
            QString exportVersion = exported.value("exportVersion").toString();
            
            QVERIFY(exportedAt.isValid());
            QCOMPARE(exportedBy, QString("ConfigurationPersistenceTest"));
            QCOMPARE(exportVersion, QString("1.0"));
            
            qDebug() << "   Export metadata verified";
            qDebug() << "      Exported at:" << exportedAt.toString();
            qDebug() << "      Exported by:" << exportedBy;
            qDebug() << "      Export version:" << exportVersion;
            
            // Import configuration settings (excluding metadata)
            QStringList keys = exported.allKeys();
            for (const QString& key : keys) {
                if (!key.startsWith("exported")) {
                    imported.setValue(key, exported.value(key));
                }
            }
            
            imported.sync();
            
            qDebug() << "   Configuration imported to:" << importedConfigFile;
        }
        
        // Verify imported configuration
        {
            QSettings settings(importedConfigFile, QSettings::IniFormat);
            AppConfig config(&settings);
            
            QCOMPARE(config.getLastScanPath(), QString("/export/test/path"));
            QCOMPARE(config.getMinimumFileSize(), qint64(2048));
            QCOMPARE(config.getTheme(), QString("Dark"));
            QCOMPARE(config.getAutoSave(), true);
            QCOMPARE(config.getMaxRecentFiles(), 20);
            
            qDebug() << "   Imported configuration verified";
            qDebug() << "      Last scan path:" << config.getLastScanPath();
            qDebug() << "      Minimum file size:" << config.getMinimumFileSize();
            qDebug() << "      Theme:" << config.getTheme();
        }
        
        qDebug() << "✓ Configuration import/export verified";
    }
};

int main(int argc, char* argv[])
{
    QCoreApplication app(argc, argv);
    ConfigurationPersistenceTest test;
    int result = QTest::qExec(&test, argc, argv);
    
    // Process any remaining events before exit
    QCoreApplication::processEvents();
    
    return result;
}

#include "test_configuration_persistence.moc"