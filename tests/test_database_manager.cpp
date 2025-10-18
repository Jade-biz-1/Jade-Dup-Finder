#include "test_database_manager.h"
#include <QStandardPaths>
#include <QDir>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlError>
#include <QSqlRecord>
#include <QDebug>
#include <QDateTime>
#include <QUuid>
#include <QJsonDocument>
#include <QTextStream>

TestDatabaseManager::TestDatabaseManager() {
    // Set up temporary database directory
    QString tempLocation = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    m_tempDatabaseDirectory = QDir(tempLocation).absoluteFilePath("dupfinder_test_databases");
    QDir().mkpath(m_tempDatabaseDirectory);
}

TestDatabaseManager::~TestDatabaseManager() {
    cleanupAllDatabases();
}

QString TestDatabaseManager::createTestDatabase(const DatabaseConfig& config) {
    QString connectionName = config.connectionName.isEmpty() ? 
        generateConnectionName(config.name) : config.connectionName;
    
    // Store configuration
    m_databaseConfigs[connectionName] = config;
    
    // Setup database connection
    if (setupDatabaseConnection(config)) {
        registerManagedConnection(connectionName);
        return connectionName;
    }
    
    return QString();
}

bool TestDatabaseManager::connectToDatabase(const QString& connectionName) {
    if (!m_databaseConfigs.contains(connectionName)) {
        qWarning() << "Database configuration not found:" << connectionName;
        return false;
    }
    
    QSqlDatabase db = QSqlDatabase::database(connectionName);
    if (!db.isValid()) {
        return setupDatabaseConnection(m_databaseConfigs[connectionName]);
    }
    
    return db.isOpen() || db.open();
}

void TestDatabaseManager::disconnectDatabase(const QString& connectionName) {
    QSqlDatabase db = QSqlDatabase::database(connectionName);
    if (db.isValid() && db.isOpen()) {
        db.close();
    }
}

void TestDatabaseManager::cleanupDatabase(const QString& connectionName) {
    disconnectDatabase(connectionName);
    
    if (m_databaseConfigs.contains(connectionName)) {
        DatabaseConfig config = m_databaseConfigs[connectionName];
        
        // Remove database file if it's a file-based database
        if (config.type == "QSQLITE" && !config.databaseName.isEmpty()) {
            QFile::remove(config.databaseName);
        }
        
        m_databaseConfigs.remove(connectionName);
    }
    
    // Remove Qt database connection
    QSqlDatabase::removeDatabase(connectionName);
    unregisterManagedConnection(connectionName);
}

void TestDatabaseManager::cleanupAllDatabases() {
    QStringList connections = m_managedConnections;
    for (const QString& connectionName : connections) {
        cleanupDatabase(connectionName);
    }
}

bool TestDatabaseManager::createSchema(const QString& connectionName, const QList<TableSchema>& tables) {
    if (!connectToDatabase(connectionName)) {
        return false;
    }
    
    QSqlDatabase db = QSqlDatabase::database(connectionName);
    
    for (const TableSchema& table : tables) {
        if (!createTable(connectionName, table)) {
            qWarning() << "Failed to create table:" << table.tableName;
            return false;
        }
    }
    
    return true;
}

bool TestDatabaseManager::createTable(const QString& connectionName, const TableSchema& table) {
    if (!connectToDatabase(connectionName)) {
        return false;
    }
    
    QSqlDatabase db = QSqlDatabase::database(connectionName);
    QString sql = generateCreateTableSql(table, db.driverName());
    
    QSqlQuery query(db);
    if (!query.exec(sql)) {
        qWarning() << "Failed to create table" << table.tableName << ":" << query.lastError().text();
        return false;
    }
    
    // Create indexes
    for (const QString& indexDef : table.indexes) {
        QString indexSql = generateIndexSql(table.tableName, indexDef, db.driverName());
        if (!query.exec(indexSql)) {
            qWarning() << "Failed to create index:" << query.lastError().text();
        }
    }
    
    return true;
}

bool TestDatabaseManager::dropTable(const QString& connectionName, const QString& tableName) {
    if (!connectToDatabase(connectionName)) {
        return false;
    }
    
    QSqlDatabase db = QSqlDatabase::database(connectionName);
    QSqlQuery query(db);
    
    QString sql = QString("DROP TABLE IF EXISTS %1").arg(tableName);
    return query.exec(sql);
}

bool TestDatabaseManager::truncateTable(const QString& connectionName, const QString& tableName) {
    if (!connectToDatabase(connectionName)) {
        return false;
    }
    
    QSqlDatabase db = QSqlDatabase::database(connectionName);
    QSqlQuery query(db);
    
    QString sql = QString("DELETE FROM %1").arg(tableName);
    return query.exec(sql);
}

QStringList TestDatabaseManager::getTableNames(const QString& connectionName) {
    if (!connectToDatabase(connectionName)) {
        return QStringList();
    }
    
    QSqlDatabase db = QSqlDatabase::database(connectionName);
    return db.tables();
}

bool TestDatabaseManager::populateDatabase(const QString& connectionName, const QList<TestDataSpec>& dataSpecs) {
    for (const TestDataSpec& spec : dataSpecs) {
        if (!insertTestData(connectionName, spec)) {
            return false;
        }
    }
    return true;
}

bool TestDatabaseManager::insertTestData(const QString& connectionName, const TestDataSpec& dataSpec) {
    if (!connectToDatabase(connectionName)) {
        return false;
    }
    
    if (dataSpec.records.isEmpty()) {
        return true; // Nothing to insert
    }
    
    // Clear existing data if requested
    if (dataSpec.clearExisting) {
        truncateTable(connectionName, dataSpec.tableName);
    }
    
    // Insert records in batches
    QList<QMap<QString, QVariant>> batch;
    for (const QMap<QString, QVariant>& record : dataSpec.records) {
        batch.append(record);
        
        if (batch.size() >= dataSpec.batchSize) {
            if (!insertRecordBatch(connectionName, dataSpec.tableName, batch)) {
                return false;
            }
            batch.clear();
        }
    }
    
    // Insert remaining records
    if (!batch.isEmpty()) {
        return insertRecordBatch(connectionName, dataSpec.tableName, batch);
    }
    
    return true;
}

bool TestDatabaseManager::executeQuery(const QString& connectionName, const QString& query, const QMap<QString, QVariant>& parameters) {
    if (!connectToDatabase(connectionName)) {
        return false;
    }
    
    QSqlDatabase db = QSqlDatabase::database(connectionName);
    QSqlQuery sqlQuery(db);
    
    if (!sqlQuery.prepare(query)) {
        qWarning() << "Failed to prepare query:" << sqlQuery.lastError().text();
        return false;
    }
    
    // Bind parameters
    bindQueryParameters(sqlQuery, parameters);
    
    bool success = sqlQuery.exec();
    if (!success) {
        qWarning() << "Query execution failed:" << sqlQuery.lastError().text();
    }
    
    // Log query if enabled
    if (m_queryLogs.contains(connectionName)) {
        m_queryLogs[connectionName].append(query);
    }
    
    return success;
}

QList<QMap<QString, QVariant>> TestDatabaseManager::selectData(const QString& connectionName, const QString& query, const QMap<QString, QVariant>& parameters) {
    QList<QMap<QString, QVariant>> results;
    
    if (!connectToDatabase(connectionName)) {
        return results;
    }
    
    QSqlDatabase db = QSqlDatabase::database(connectionName);
    QSqlQuery sqlQuery(db);
    
    if (!sqlQuery.prepare(query)) {
        qWarning() << "Failed to prepare select query:" << sqlQuery.lastError().text();
        return results;
    }
    
    // Bind parameters
    bindQueryParameters(sqlQuery, parameters);
    
    if (!sqlQuery.exec()) {
        qWarning() << "Select query execution failed:" << sqlQuery.lastError().text();
        return results;
    }
    
    // Process results
    while (sqlQuery.next()) {
        QMap<QString, QVariant> record;
        QSqlRecord sqlRecord = sqlQuery.record();
        
        for (int i = 0; i < sqlRecord.count(); ++i) {
            QString fieldName = sqlRecord.fieldName(i);
            QVariant value = sqlQuery.value(i);
            record[fieldName] = value;
        }
        
        results.append(record);
    }
    
    // Log query if enabled
    if (m_queryLogs.contains(connectionName)) {
        m_queryLogs[connectionName].append(query);
    }
    
    return results;
}

QString TestDatabaseManager::createScenarioDatabase(DatabaseScenario scenario, const QString& baseName) {
    QString scenarioName = baseName.isEmpty() ? 
        QString("scenario_%1").arg(static_cast<int>(scenario)) : baseName;
    
    DatabaseConfig config = getDefaultSQLiteConfig(scenarioName);
    QString connectionName = createTestDatabase(config);
    
    if (connectionName.isEmpty()) {
        return QString();
    }
    
    // Create schema for scenario
    QList<TableSchema> schema = getScenarioSchema(scenario);
    if (!createSchema(connectionName, schema)) {
        cleanupDatabase(connectionName);
        return QString();
    }
    
    // Populate with scenario data
    QList<TestDataSpec> data = getScenarioData(scenario);
    if (!populateDatabase(connectionName, data)) {
        qWarning() << "Failed to populate scenario database with data";
        // Don't fail completely - schema might still be useful
    }
    
    return connectionName;
}

QList<TestDatabaseManager::TableSchema> TestDatabaseManager::getScenarioSchema(DatabaseScenario scenario) {
    switch (scenario) {
        case DatabaseScenario::Configuration:
            return getDupFinderConfigSchema();
        case DatabaseScenario::ScanResults:
            return getDupFinderScanResultsSchema();
        case DatabaseScenario::UserPreferences:
            return getDupFinderUserPreferencesSchema();
        default:
            // Return basic schema for other scenarios
            return getDupFinderConfigSchema();
    }
}

QList<TestDatabaseManager::TestDataSpec> TestDatabaseManager::getScenarioData(DatabaseScenario scenario) {
    switch (scenario) {
        case DatabaseScenario::Configuration:
            return generateConfigurationData();
        case DatabaseScenario::ScanResults:
            return generateScanResultsData();
        case DatabaseScenario::UserPreferences:
            return generateUserPreferencesData();
        case DatabaseScenario::PerformanceTest:
            return generatePerformanceTestData();
        case DatabaseScenario::CorruptedData:
            return generateCorruptedData();
        default:
            return QList<TestDataSpec>(); // Empty data for other scenarios
    }
}

QString TestDatabaseManager::createConfigurationDatabase() {
    return createScenarioDatabase(DatabaseScenario::Configuration, "config");
}

QString TestDatabaseManager::createScanResultsDatabase() {
    return createScenarioDatabase(DatabaseScenario::ScanResults, "scan_results");
}

QString TestDatabaseManager::createUserPreferencesDatabase() {
    return createScenarioDatabase(DatabaseScenario::UserPreferences, "user_prefs");
}

bool TestDatabaseManager::populateScanResults(const QString& connectionName, const QStringList& filePaths, const QMap<QString, QString>& hashes) {
    TestDataSpec spec;
    spec.tableName = "scan_results";
    spec.clearExisting = false; // Append to existing data
    
    for (const QString& filePath : filePaths) {
        QMap<QString, QVariant> record;
        record["file_path"] = filePath;
        record["file_hash"] = hashes.value(filePath, "");
        record["file_size"] = QFileInfo(filePath).size();
        record["scan_date"] = QDateTime::currentDateTime();
        
        spec.records.append(record);
    }
    
    return insertTestData(connectionName, spec);
}

QMap<QString, QVariant> TestDatabaseManager::getDatabaseStatistics(const QString& connectionName) {
    QMap<QString, QVariant> stats;
    
    QStringList tables = getTableNames(connectionName);
    stats["table_count"] = tables.size();
    
    qint64 totalRecords = 0;
    for (const QString& tableName : tables) {
        int recordCount = getTableRecordCount(connectionName, tableName);
        stats[QString("table_%1_records").arg(tableName)] = recordCount;
        totalRecords += recordCount;
    }
    
    stats["total_records"] = totalRecords;
    stats["connection_name"] = connectionName;
    
    return stats;
}

QString TestDatabaseManager::generateDatabaseReport(const QString& connectionName) {
    QString report;
    QTextStream stream(&report);
    
    stream << "Database Report\n";
    stream << "===============\n";
    stream << "Connection: " << connectionName << "\n";
    stream << "Generated: " << QDateTime::currentDateTime().toString() << "\n\n";
    
    QMap<QString, QVariant> stats = getDatabaseStatistics(connectionName);
    stream << "Statistics:\n";
    stream << "- Tables: " << stats["table_count"].toInt() << "\n";
    stream << "- Total Records: " << stats["total_records"].toLongLong() << "\n\n";
    
    QStringList tables = getTableNames(connectionName);
    stream << "Tables:\n";
    for (const QString& tableName : tables) {
        int recordCount = stats[QString("table_%1_records").arg(tableName)].toInt();
        stream << "- " << tableName << ": " << recordCount << " records\n";
    }
    
    return report;
}

// Static utility functions
QString TestDatabaseManager::generateConnectionName(const QString& baseName) {
    QString timestamp = QString::number(QDateTime::currentMSecsSinceEpoch());
    QString uuid = QUuid::createUuid().toString(QUuid::WithoutBraces).left(8);
    return QString("%1_%2_%3").arg(baseName).arg(timestamp).arg(uuid);
}

QString TestDatabaseManager::generateTemporaryDatabasePath(const QString& baseName) {
    QString tempDir = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    QString fileName = QString("%1_%2.db")
                      .arg(baseName)
                      .arg(QDateTime::currentMSecsSinceEpoch());
    return QDir(tempDir).absoluteFilePath(fileName);
}

bool TestDatabaseManager::isDatabaseConnected(const QString& connectionName) {
    QSqlDatabase db = QSqlDatabase::database(connectionName);
    return db.isValid() && db.isOpen();
}

TestDatabaseManager::DatabaseConfig TestDatabaseManager::getDefaultSQLiteConfig(const QString& databaseName) {
    DatabaseConfig config;
    config.name = databaseName.isEmpty() ? "test_db" : databaseName;
    config.type = "QSQLITE";
    config.databaseName = generateTemporaryDatabasePath(config.name);
    config.connectionName = generateConnectionName(config.name);
    config.isTemporary = true;
    config.autoCleanup = true;
    
    return config;
}

TestDatabaseManager::DatabaseConfig TestDatabaseManager::getInMemoryDatabaseConfig(const QString& databaseName) {
    DatabaseConfig config;
    config.name = databaseName.isEmpty() ? "memory_db" : databaseName;
    config.type = "QSQLITE";
    config.databaseName = ":memory:";
    config.connectionName = generateConnectionName(config.name);
    config.isTemporary = true;
    config.autoCleanup = true;
    
    return config;
}

// Private helper methods
bool TestDatabaseManager::setupDatabaseConnection(const DatabaseConfig& config) {
    QSqlDatabase db = QSqlDatabase::addDatabase(config.type, config.connectionName);
    
    if (config.type == "QSQLITE") {
        db.setDatabaseName(config.databaseName);
    } else {
        db.setHostName(config.hostName);
        db.setPort(config.port);
        db.setDatabaseName(config.databaseName);
        db.setUserName(config.userName);
        db.setPassword(config.password);
    }
    
    // Set connection options
    for (auto it = config.options.begin(); it != config.options.end(); ++it) {
        db.setConnectOptions(QString("%1=%2").arg(it.key()).arg(it.value()));
    }
    
    if (!db.open()) {
        qWarning() << "Failed to open database:" << db.lastError().text();
        QSqlDatabase::removeDatabase(config.connectionName);
        return false;
    }
    
    return true;
}

void TestDatabaseManager::registerManagedConnection(const QString& connectionName) {
    if (!m_managedConnections.contains(connectionName)) {
        m_managedConnections.append(connectionName);
    }
}

void TestDatabaseManager::unregisterManagedConnection(const QString& connectionName) {
    m_managedConnections.removeAll(connectionName);
}

QString TestDatabaseManager::generateCreateTableSql(const TableSchema& table, const QString& driverType) {
    Q_UNUSED(driverType)
    
    QString sql = QString("CREATE TABLE %1 (").arg(table.tableName);
    
    // Add columns
    QStringList columnDefs;
    for (const QString& column : table.columns) {
        columnDefs.append(column);
    }
    
    sql += columnDefs.join(", ");
    
    // Add primary key constraint
    if (!table.primaryKeys.isEmpty()) {
        sql += QString(", PRIMARY KEY (%1)").arg(table.primaryKeys.join(", "));
    }
    
    sql += ")";
    
    return sql;
}

QString TestDatabaseManager::generateIndexSql(const QString& tableName, const QString& indexDef, const QString& driverType) {
    Q_UNUSED(driverType)
    
    // Simple index creation - in real implementation, parse indexDef properly
    return QString("CREATE INDEX idx_%1_%2 ON %1 (%2)")
           .arg(tableName)
           .arg(indexDef);
}

bool TestDatabaseManager::insertRecordBatch(const QString& connectionName, const QString& tableName, const QList<QMap<QString, QVariant>>& records) {
    if (records.isEmpty()) {
        return true;
    }
    
    // Get column names from first record
    QStringList columns = records.first().keys();
    QString sql = generateInsertSql(tableName, columns);
    
    QSqlDatabase db = QSqlDatabase::database(connectionName);
    QSqlQuery query(db);
    
    if (!query.prepare(sql)) {
        qWarning() << "Failed to prepare insert query:" << query.lastError().text();
        return false;
    }
    
    // Insert each record
    for (const QMap<QString, QVariant>& record : records) {
        // Bind values in column order
        for (const QString& column : columns) {
            query.addBindValue(record.value(column));
        }
        
        if (!query.exec()) {
            qWarning() << "Failed to insert record:" << query.lastError().text();
            return false;
        }
    }
    
    return true;
}

QString TestDatabaseManager::generateInsertSql(const QString& tableName, const QStringList& columns) {
    QString columnList = columns.join(", ");
    QString placeholders = QString("?").repeated(columns.size());
    placeholders = placeholders.split("").join(", ?").mid(2); // Convert to "?, ?, ?"
    
    return QString("INSERT INTO %1 (%2) VALUES (%3)")
           .arg(tableName)
           .arg(columnList)
           .arg(placeholders);
}

void TestDatabaseManager::bindQueryParameters(QSqlQuery& query, const QMap<QString, QVariant>& parameters) {
    for (auto it = parameters.begin(); it != parameters.end(); ++it) {
        query.bindValue(it.key(), it.value());
    }
}

int TestDatabaseManager::getTableRecordCount(const QString& connectionName, const QString& tableName) {
    QString query = QString("SELECT COUNT(*) as count FROM %1").arg(tableName);
    auto results = selectData(connectionName, query);
    
    if (!results.isEmpty()) {
        return results.first()["count"].toInt();
    }
    
    return 0;
}

// Schema definitions for DupFinder
QList<TestDatabaseManager::TableSchema> TestDatabaseManager::getDupFinderConfigSchema() {
    QList<TableSchema> schema;
    
    TableSchema settingsTable;
    settingsTable.tableName = "settings";
    settingsTable.columns = {
        "id INTEGER PRIMARY KEY AUTOINCREMENT",
        "key TEXT NOT NULL",
        "value TEXT",
        "category TEXT",
        "created_date DATETIME DEFAULT CURRENT_TIMESTAMP"
    };
    settingsTable.indexes = {"key", "category"};
    schema.append(settingsTable);
    
    return schema;
}

QList<TestDatabaseManager::TableSchema> TestDatabaseManager::getDupFinderScanResultsSchema() {
    QList<TableSchema> schema;
    
    TableSchema scanResultsTable;
    scanResultsTable.tableName = "scan_results";
    scanResultsTable.columns = {
        "id INTEGER PRIMARY KEY AUTOINCREMENT",
        "file_path TEXT NOT NULL",
        "file_hash TEXT",
        "file_size INTEGER",
        "scan_date DATETIME DEFAULT CURRENT_TIMESTAMP"
    };
    scanResultsTable.indexes = {"file_hash", "file_path"};
    schema.append(scanResultsTable);
    
    return schema;
}

QList<TestDatabaseManager::TableSchema> TestDatabaseManager::getDupFinderUserPreferencesSchema() {
    QList<TableSchema> schema;
    
    TableSchema preferencesTable;
    preferencesTable.tableName = "user_preferences";
    preferencesTable.columns = {
        "id INTEGER PRIMARY KEY AUTOINCREMENT",
        "preference_key TEXT NOT NULL",
        "preference_value TEXT",
        "user_id TEXT",
        "updated_date DATETIME DEFAULT CURRENT_TIMESTAMP"
    };
    preferencesTable.indexes = {"preference_key", "user_id"};
    schema.append(preferencesTable);
    
    return schema;
}

// Data generators
QList<TestDatabaseManager::TestDataSpec> TestDatabaseManager::generateConfigurationData() {
    QList<TestDataSpec> dataSpecs;
    
    TestDataSpec settingsData;
    settingsData.tableName = "settings";
    
    // Generate sample configuration data
    QList<QMap<QString, QVariant>> records = {
        {{"key", "scan_threads"}, {"value", "4"}, {"category", "performance"}},
        {{"key", "hash_algorithm"}, {"value", "MD5"}, {"category", "scanning"}},
        {{"key", "auto_backup"}, {"value", "true"}, {"category", "safety"}},
        {{"key", "theme"}, {"value", "dark"}, {"category", "ui"}},
        {{"key", "language"}, {"value", "en"}, {"category", "ui"}}
    };
    
    settingsData.records = records;
    dataSpecs.append(settingsData);
    
    return dataSpecs;
}

QList<TestDatabaseManager::TestDataSpec> TestDatabaseManager::generateScanResultsData(int fileCount) {
    QList<TestDataSpec> dataSpecs;
    
    TestDataSpec scanData;
    scanData.tableName = "scan_results";
    
    // Generate sample scan results
    for (int i = 0; i < fileCount; ++i) {
        QMap<QString, QVariant> record;
        record["file_path"] = QString("/test/path/file_%1.txt").arg(i);
        record["file_hash"] = QString("hash_%1").arg(i);
        record["file_size"] = 1024 + (i * 100);
        record["scan_date"] = QDateTime::currentDateTime().addSecs(-i * 60);
        
        scanData.records.append(record);
    }
    
    dataSpecs.append(scanData);
    return dataSpecs;
}

QList<TestDatabaseManager::TestDataSpec> TestDatabaseManager::generateUserPreferencesData() {
    QList<TestDataSpec> dataSpecs;
    
    TestDataSpec prefsData;
    prefsData.tableName = "user_preferences";
    
    QList<QMap<QString, QVariant>> records = {
        {{"preference_key", "window_geometry"}, {"preference_value", "800x600+100+100"}, {"user_id", "test_user"}},
        {{"preference_key", "last_scan_path"}, {"preference_value", "/home/user/Documents"}, {"user_id", "test_user"}},
        {{"preference_key", "show_hidden_files"}, {"preference_value", "false"}, {"user_id", "test_user"}}
    };
    
    prefsData.records = records;
    dataSpecs.append(prefsData);
    
    return dataSpecs;
}

QList<TestDatabaseManager::TestDataSpec> TestDatabaseManager::generatePerformanceTestData(int recordCount) {
    // Generate large dataset for performance testing
    return generateScanResultsData(recordCount);
}

QList<TestDatabaseManager::TestDataSpec> TestDatabaseManager::generateCorruptedData() {
    QList<TestDataSpec> dataSpecs;
    
    TestDataSpec corruptedData;
    corruptedData.tableName = "settings";
    
    // Generate data with potential issues
    QList<QMap<QString, QVariant>> records = {
        {{"key", ""}, {"value", "empty_key"}, {"category", "test"}}, // Empty key
        {{"key", "null_value"}, {"value", QVariant()}, {"category", "test"}}, // Null value
        {{"key", "very_long_key_" + QString("x").repeated(1000)}, {"value", "long_key"}, {"category", "test"}} // Very long key
    };
    
    corruptedData.records = records;
    dataSpecs.append(corruptedData);
    
    return dataSpecs;
}