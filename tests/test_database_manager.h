#pragma once

#include <QString>
#include <QStringList>
#include <QMap>
#include <QVariant>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlError>
#include <QJsonObject>
#include <QJsonArray>
#include <QTemporaryFile>

/**
 * @brief Test database management system for DupFinder testing
 * 
 * Provides utilities for creating, populating, and managing test databases
 * including configuration databases, metadata storage, and test data scenarios.
 */
class TestDatabaseManager {
public:
    /**
     * @brief Database configuration specification
     */
    struct DatabaseConfig {
        QString name;
        QString type = "QSQLITE";           // Database driver type
        QString hostName;                   // For network databases
        int port = -1;                      // For network databases
        QString databaseName;               // Database name or file path
        QString userName;                   // For authenticated databases
        QString password;                   // For authenticated databases
        QString connectionName;             // Qt connection name
        QMap<QString, QString> options;     // Additional connection options
        bool isTemporary = true;            // Whether to use temporary file
        bool autoCleanup = true;            // Whether to cleanup on destruction
    };

    /**
     * @brief Table schema definition
     */
    struct TableSchema {
        QString tableName;
        QStringList columns;                // Column definitions (name type constraints)
        QStringList primaryKeys;            // Primary key columns
        QStringList indexes;                // Index definitions
        QStringList foreignKeys;            // Foreign key constraints
        QMap<QString, QVariant> options;    // Table-specific options
    };

    /**
     * @brief Test data specification for database population
     */
    struct TestDataSpec {
        QString tableName;
        QList<QMap<QString, QVariant>> records;  // List of record data
        bool clearExisting = true;               // Clear table before inserting
        bool useTransactions = true;             // Use transactions for bulk insert
        int batchSize = 1000;                   // Batch size for large datasets
    };

    /**
     * @brief Predefined database scenarios for testing
     */
    enum class DatabaseScenario {
        Empty,                    ///< Empty database with schema only
        Minimal,                  ///< Minimal test data
        Small,                    ///< Small dataset (< 1000 records)
        Medium,                   ///< Medium dataset (1000-10000 records)
        Large,                    ///< Large dataset (10000+ records)
        Configuration,            ///< DupFinder configuration data
        ScanResults,             ///< File scan results and metadata
        UserPreferences,         ///< User settings and preferences
        PerformanceTest,         ///< Large dataset for performance testing
        CorruptedData,           ///< Data with integrity issues
        EdgeCases,               ///< Edge case data scenarios
        Migration,               ///< Database migration scenarios
        Concurrent               ///< Multi-connection test scenario
    };

    TestDatabaseManager();
    ~TestDatabaseManager();

    // Database lifecycle management
    QString createTestDatabase(const DatabaseConfig& config);
    bool connectToDatabase(const QString& connectionName);
    void disconnectDatabase(const QString& connectionName);
    void cleanupDatabase(const QString& connectionName);
    void cleanupAllDatabases();

    // Schema management
    bool createSchema(const QString& connectionName, const QList<TableSchema>& tables);
    bool createTable(const QString& connectionName, const TableSchema& table);
    bool dropTable(const QString& connectionName, const QString& tableName);
    bool truncateTable(const QString& connectionName, const QString& tableName);
    QStringList getTableNames(const QString& connectionName);

    // Data population and management
    bool populateDatabase(const QString& connectionName, const QList<TestDataSpec>& dataSpecs);
    bool insertTestData(const QString& connectionName, const TestDataSpec& dataSpec);
    bool executeQuery(const QString& connectionName, const QString& query, const QMap<QString, QVariant>& parameters = {});
    QList<QMap<QString, QVariant>> selectData(const QString& connectionName, const QString& query, const QMap<QString, QVariant>& parameters = {});

    // Scenario-based database creation
    QString createScenarioDatabase(DatabaseScenario scenario, const QString& baseName = "");
    QList<TableSchema> getScenarioSchema(DatabaseScenario scenario);
    QList<TestDataSpec> getScenarioData(DatabaseScenario scenario);

    // DupFinder-specific database utilities
    QString createConfigurationDatabase();
    QString createScanResultsDatabase();
    QString createUserPreferencesDatabase();
    bool populateConfigurationData(const QString& connectionName, const QJsonObject& config);
    bool populateScanResults(const QString& connectionName, const QStringList& filePaths, const QMap<QString, QString>& hashes);

    // Database validation and verification
    bool validateDatabaseSchema(const QString& connectionName, const QList<TableSchema>& expectedSchema);
    bool validateTableData(const QString& connectionName, const QString& tableName, int expectedRecordCount = -1);
    QMap<QString, QVariant> getDatabaseStatistics(const QString& connectionName);
    QString generateDatabaseReport(const QString& connectionName);

    // Transaction management
    bool beginTransaction(const QString& connectionName);
    bool commitTransaction(const QString& connectionName);
    bool rollbackTransaction(const QString& connectionName);

    // Backup and restore
    bool backupDatabase(const QString& connectionName, const QString& backupPath);
    bool restoreDatabase(const QString& connectionName, const QString& backupPath);
    QString exportDatabaseToSql(const QString& connectionName);
    bool importDatabaseFromSql(const QString& connectionName, const QString& sqlContent);

    // Performance and monitoring
    void enableQueryLogging(const QString& connectionName, bool enable = true);
    QStringList getExecutedQueries(const QString& connectionName);
    void clearQueryLog(const QString& connectionName);
    qint64 measureQueryPerformance(const QString& connectionName, const QString& query, int iterations = 1);

    // Utility functions
    static QString generateConnectionName(const QString& baseName = "test_db");
    static QString generateTemporaryDatabasePath(const QString& baseName = "test");
    static QSqlError getLastError(const QString& connectionName);
    static bool isDatabaseConnected(const QString& connectionName);

    // Configuration helpers
    DatabaseConfig getDefaultSQLiteConfig(const QString& databaseName = "");
    DatabaseConfig getInMemoryDatabaseConfig(const QString& databaseName = "");

private:
    QMap<QString, DatabaseConfig> m_databaseConfigs;
    QMap<QString, QStringList> m_queryLogs;
    QStringList m_managedConnections;
    QString m_tempDatabaseDirectory;

    // Internal helper methods
    bool setupDatabaseConnection(const DatabaseConfig& config);
    QString createTemporaryDatabaseFile(const QString& baseName);
    void registerManagedConnection(const QString& connectionName);
    void unregisterManagedConnection(const QString& connectionName);

    // Schema creation helpers
    QString generateCreateTableSql(const TableSchema& table, const QString& driverType);
    QString generateColumnDefinition(const QString& column, const QString& driverType);
    QString generateIndexSql(const QString& tableName, const QString& indexDef, const QString& driverType);

    // Data population helpers
    bool insertRecordBatch(const QString& connectionName, const QString& tableName, 
                          const QList<QMap<QString, QVariant>>& records);
    QString generateInsertSql(const QString& tableName, const QStringList& columns);
    void bindQueryParameters(QSqlQuery& query, const QMap<QString, QVariant>& parameters);

    // Validation helpers
    bool compareTableSchema(const QString& connectionName, const TableSchema& expected);
    QStringList getTableColumns(const QString& connectionName, const QString& tableName);
    int getTableRecordCount(const QString& connectionName, const QString& tableName);

    // Scenario data generators
    QList<TestDataSpec> generateConfigurationData();
    QList<TestDataSpec> generateScanResultsData(int fileCount = 100);
    QList<TestDataSpec> generateUserPreferencesData();
    QList<TestDataSpec> generatePerformanceTestData(int recordCount = 10000);
    QList<TestDataSpec> generateCorruptedData();

    // DupFinder schema definitions
    QList<TableSchema> getDupFinderConfigSchema();
    QList<TableSchema> getDupFinderScanResultsSchema();
    QList<TableSchema> getDupFinderUserPreferencesSchema();
};

/**
 * @brief RAII wrapper for database transactions
 */
class DatabaseTransaction {
public:
    explicit DatabaseTransaction(const QString& connectionName);
    ~DatabaseTransaction();

    bool commit();
    void rollback();
    bool isActive() const;

private:
    QString m_connectionName;
    bool m_committed;
    bool m_active;
};

/**
 * @brief Macros for convenient database testing
 */
#define CREATE_TEST_DATABASE(scenario) \
    TestDatabaseManager dbManager; \
    QString dbConnection = dbManager.createScenarioDatabase(TestDatabaseManager::DatabaseScenario::scenario)

#define EXECUTE_DB_QUERY(connection, query) \
    do { \
        if (!dbManager.executeQuery(connection, query)) { \
            QFAIL(QString("Database query failed: %1").arg(query).toUtf8().constData()); \
        } \
    } while(0)

#define VERIFY_DB_RECORD_COUNT(connection, table, expectedCount) \
    do { \
        QString query = QString("SELECT COUNT(*) as count FROM %1").arg(table); \
        auto results = dbManager.selectData(connection, query); \
        QVERIFY(!results.isEmpty()); \
        int actualCount = results.first()["count"].toInt(); \
        QCOMPARE(actualCount, expectedCount); \
    } while(0)