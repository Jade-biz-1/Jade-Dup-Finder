# Test Data Management System

This document describes the comprehensive test data management system for DupFinder, which provides utilities for creating realistic test datasets, managing test environments, and handling test databases.

## Overview

The test data management system consists of three main components:

1. **TestDataGenerator** - Creates realistic file hierarchies and test datasets
2. **TestEnvironmentIsolator** - Provides isolated test environments with resource monitoring
3. **TestDatabaseManager** - Manages test databases with schema creation and data population

## Components

### 1. TestDataGenerator

The `TestDataGenerator` class provides comprehensive utilities for creating realistic test data including files, directories, and various test scenarios.

#### Key Features

- **File Generation**: Create files with specific content, sizes, and metadata
- **Directory Structures**: Generate complex directory hierarchies with configurable depth and width
- **Duplicate Scenarios**: Create duplicate files for testing detection algorithms
- **Realistic Content**: Generate appropriate content for different file types (text, images, documents)
- **Special Cases**: Handle edge cases like long paths, special characters, hidden files, and symlinks
- **Performance Datasets**: Create large datasets for performance and stress testing

#### Usage Examples

```cpp
#include "test_data_generator.h"

// Create a test data generator
TestDataGenerator generator;

// Generate a single file
TestDataGenerator::FileSpec fileSpec;
fileSpec.fileName = "test_document.txt";
fileSpec.sizeBytes = 1024;
fileSpec.content = "Test content";
QString filePath = generator.generateTestFile(fileSpec);

// Generate a directory structure
TestDataGenerator::DirectorySpec dirSpec;
dirSpec.name = "test_dataset";
dirSpec.depth = 3;
dirSpec.filesPerDirectory = 10;
dirSpec.subdirectories = 5;
dirSpec.duplicateRatio = 0.2; // 20% duplicates
QString dirPath = generator.generateTestDirectory(dirSpec);

// Generate scenario-based datasets
QStringList files = generator.generateFileHierarchy(
    TestDataGenerator::TestScenario::MediumDataset
);

// Create duplicate files
QStringList duplicates = generator.createDuplicateSet(originalFile, 3);

// Generate performance dataset
QString perfDataset = generator.generatePerformanceDataset(100 * 1024 * 1024); // 100MB

// Cleanup (automatic on destruction)
generator.cleanupGeneratedData();
```

#### Predefined Scenarios

| Scenario | Description | Files | Structure |
|----------|-------------|-------|-----------|
| `EmptyDirectory` | Empty directory for edge cases | 0 | Minimal |
| `SingleFile` | Single file scenario | 1 | Flat |
| `SmallDataset` | Small test dataset | ~30 | 2 levels deep |
| `MediumDataset` | Medium test dataset | ~375 | 3 levels deep |
| `LargeDataset` | Large test dataset | ~1600 | 4 levels deep |
| `DeepHierarchy` | Deep directory structure | ~50 | 10 levels deep |
| `WideHierarchy` | Wide directory structure | ~320 | 2 levels, 15 subdirs |
| `DuplicateHeavy` | High duplicate ratio | ~320 | 70% duplicates |
| `PerformanceStress` | Large files for performance | ~1500 | 1-10MB files |
| `EdgeCases` | Special characters, symlinks | ~45 | Various edge cases |

### 2. TestEnvironmentIsolator

The `TestEnvironmentIsolator` class provides isolated test environments to prevent test interference and enable resource monitoring.

#### Key Features

- **Environment Isolation**: Create isolated directories for each test
- **Resource Monitoring**: Track memory and disk usage during test execution
- **Process Isolation**: Run tests in separate processes with controlled environments
- **Automatic Cleanup**: Clean up environments after test completion
- **Performance Tracking**: Measure execution time and resource consumption

#### Usage Examples

```cpp
#include "test_data_generator.h"

// Create environment isolator
TestEnvironmentIsolator isolator;

// Create isolated environment
QString envId = isolator.createIsolatedEnvironment("my_test");
QString envPath = isolator.getEnvironmentPath(envId);

// Start resource monitoring
isolator.startResourceMonitoring(envId);

// Perform test operations in isolated environment
TestDataGenerator generator;
generator.setTemporaryDirectory(envPath + "/data");
// ... test operations ...

// Stop monitoring and get resource usage
isolator.stopResourceMonitoring(envId);
QMap<QString, QVariant> usage = isolator.getResourceUsage(envId);

qDebug() << "Memory used:" << usage["memoryUsed"].toLongLong() << "bytes";
qDebug() << "Execution time:" << usage["executionTimeMs"].toLongLong() << "ms";

// Cleanup environment
isolator.destroyIsolatedEnvironment(envId);
```

### 3. TestDatabaseManager

The `TestDatabaseManager` class provides comprehensive database management for testing, including schema creation, data population, and scenario-based database generation.

#### Key Features

- **Database Creation**: Create SQLite, MySQL, PostgreSQL test databases
- **Schema Management**: Define and create table schemas with constraints
- **Data Population**: Populate databases with realistic test data
- **Scenario Databases**: Pre-configured databases for common test scenarios
- **Transaction Support**: Manage database transactions for test isolation
- **Performance Testing**: Generate large datasets for database performance testing

#### Usage Examples

```cpp
#include "test_database_manager.h"

// Create database manager
TestDatabaseManager dbManager;

// Create scenario-based database
QString dbConnection = dbManager.createScenarioDatabase(
    TestDatabaseManager::DatabaseScenario::Configuration
);

// Verify database connection
if (TestDatabaseManager::isDatabaseConnected(dbConnection)) {
    // Execute queries
    QString query = "SELECT COUNT(*) as count FROM settings";
    auto results = dbManager.selectData(dbConnection, query);
    
    // Insert test data
    TestDatabaseManager::TestDataSpec dataSpec;
    dataSpec.tableName = "settings";
    dataSpec.records = {
        {{"key", "test_setting"}, {"value", "test_value"}, {"category", "test"}}
    };
    dbManager.insertTestData(dbConnection, dataSpec);
}

// Generate database report
QString report = dbManager.generateDatabaseReport(dbConnection);
qDebug() << report;

// Cleanup (automatic on destruction)
dbManager.cleanupDatabase(dbConnection);
```

#### Database Scenarios

| Scenario | Description | Tables | Use Case |
|----------|-------------|--------|----------|
| `Configuration` | App configuration data | settings | Configuration testing |
| `ScanResults` | File scan results | scan_results | Scan algorithm testing |
| `UserPreferences` | User settings | user_preferences | UI preference testing |
| `PerformanceTest` | Large dataset | scan_results (10k+ records) | Performance testing |
| `CorruptedData` | Data integrity issues | Various with bad data | Error handling testing |
| `Empty` | Schema only | Empty tables | Schema validation |

## Integration with Test Framework

The test data management system integrates seamlessly with the enhanced test framework:

### Using with TestBase

```cpp
#include "test_base.h"
#include "test_data_generator.h"

DECLARE_TEST_CLASS(MyDataTest, Unit, High, "data", "files")

private slots:
    TEST_METHOD(test_fileProcessing_largeFiles_handlesCorrectly) {
        // Create test data in isolated environment
        TestDataGenerator generator;
        generator.setTemporaryDirectory(createTestDirectory("large_files"));
        
        // Generate large files for testing
        QString largeFile = generator.generateLargeFileSet(5, 10 * 1024 * 1024); // 5 x 10MB files
        
        // Test file processing logic
        // ... test implementation ...
        
        // Cleanup is automatic via TestBase
    }

END_TEST_CLASS()
```

### Using with Database Tests

```cpp
DECLARE_TEST_CLASS(MyDatabaseTest, Integration, High, "database", "persistence")

private slots:
    TEST_METHOD(test_scanResults_persistence_storesCorrectly) {
        // Create test database
        CREATE_TEST_DATABASE(ScanResults);
        
        // Verify database setup
        VERIFY_DB_RECORD_COUNT(dbConnection, "scan_results", 100);
        
        // Test database operations
        EXECUTE_DB_QUERY(dbConnection, "INSERT INTO scan_results (file_path, file_hash) VALUES ('test', 'hash')");
        
        // Verify results
        VERIFY_DB_RECORD_COUNT(dbConnection, "scan_results", 101);
    }

END_TEST_CLASS()
```

## Configuration

### Test Data Configuration

Create a `test_data_config.json` file to configure default test data generation:

```json
{
  "dataGeneration": {
    "defaultTempDirectory": "/tmp/dupfinder_test_data",
    "cleanupOnExit": true,
    "maxDatasetSize": "1GB",
    "defaultFileTypes": ["txt", "jpg", "pdf", "doc"],
    "performanceTestSizes": {
      "small": "10MB",
      "medium": "100MB", 
      "large": "1GB"
    }
  },
  "environmentIsolation": {
    "enableResourceMonitoring": true,
    "isolationDirectory": "/tmp/dupfinder_isolation",
    "maxEnvironments": 10,
    "cleanupDelaySeconds": 30
  },
  "databaseTesting": {
    "defaultDriver": "QSQLITE",
    "inMemoryByDefault": false,
    "enableQueryLogging": true,
    "maxConnections": 5
  }
}
```

### Environment Variables

The test data management system respects these environment variables:

- `DUPFINDER_TEST_DATA_DIR`: Override default test data directory
- `DUPFINDER_TEST_DB_DIR`: Override default test database directory
- `DUPFINDER_TEST_CLEANUP`: Set to "false" to disable automatic cleanup
- `DUPFINDER_TEST_ISOLATION`: Set to "false" to disable environment isolation
- `DUPFINDER_TEST_PERFORMANCE`: Set to "true" to enable performance datasets

## Best Practices

### 1. Test Data Organization

```cpp
// Good: Use descriptive names and organize by test purpose
TestDataGenerator generator;
QString scanTestData = generator.generateFileHierarchy(
    TestDataGenerator::TestScenario::MediumDataset, 
    "scan_algorithm_tests"
);

// Good: Use appropriate scenarios for test requirements
QString duplicateTestData = generator.generateFileHierarchy(
    TestDataGenerator::TestScenario::DuplicateHeavy,
    "duplicate_detection_tests"
);
```

### 2. Environment Isolation

```cpp
// Good: Use isolation for tests that modify file system
TestEnvironmentIsolator isolator;
QString envId = isolator.createIsolatedEnvironment("file_modification_test");

// Perform file modifications in isolated environment
// ... test operations ...

// Cleanup is automatic
isolator.destroyIsolatedEnvironment(envId);
```

### 3. Database Testing

```cpp
// Good: Use transactions for test isolation
{
    DatabaseTransaction transaction(dbConnection);
    
    // Perform database operations
    dbManager.executeQuery(dbConnection, "INSERT INTO ...");
    
    // Test the operations
    // ...
    
    // Rollback automatically on scope exit (unless committed)
}
```

### 4. Performance Considerations

```cpp
// Good: Use appropriate dataset sizes for performance tests
void test_performance_largeDataset() {
    skipIfCI("Performance test too slow for CI");
    
    TestDataGenerator generator;
    startPerformanceMeasurement("large_dataset_processing");
    
    QString dataset = generator.generatePerformanceDataset(100 * 1024 * 1024);
    // ... performance test ...
    
    qint64 elapsed = stopPerformanceMeasurement("large_dataset_processing");
    recordPerformanceMetric("processing_rate", dataSize / elapsed, "bytes/ms");
}
```

### 5. Cleanup Management

```cpp
// Good: Explicit cleanup for critical tests
void test_criticalOperation() {
    TestDataGenerator generator;
    generator.setCleanupOnDestruction(false); // Manual cleanup
    
    QString testData = generator.generateTestDirectory(spec);
    
    try {
        // Critical test operations
        // ...
        
        // Explicit cleanup on success
        generator.cleanupGeneratedData();
    } catch (...) {
        // Ensure cleanup on failure
        generator.cleanupGeneratedData();
        throw;
    }
}
```

## Troubleshooting

### Common Issues

1. **Insufficient Disk Space**
   ```cpp
   // Check available space before generating large datasets
   QStorageInfo storage(testDirectory);
   if (storage.bytesAvailable() < requiredSize) {
       QSKIP("Insufficient disk space for test");
   }
   ```

2. **Permission Issues**
   ```cpp
   // Verify write permissions
   QFileInfo dirInfo(testDirectory);
   if (!dirInfo.isWritable()) {
       QSKIP("No write permission for test directory");
   }
   ```

3. **Database Connection Failures**
   ```cpp
   // Check database connection before tests
   if (!TestDatabaseManager::isDatabaseConnected(connection)) {
       QFAIL("Database connection failed");
   }
   ```

### Debug Mode

Enable verbose logging for debugging:

```cpp
// Enable debug logging
TestConfig::instance().globalConfig().verboseOutput = true;

// Enable database query logging
dbManager.enableQueryLogging(connection, true);

// Check generated data
QMap<QString, qint64> analysis = generator.analyzeGeneratedData(dataPath);
qDebug() << "Generated data analysis:" << analysis;
```

## Performance Metrics

The test data management system tracks various performance metrics:

### File Generation Metrics
- Files per second generation rate
- Data generation throughput (MB/s)
- Directory creation time
- Duplicate file creation efficiency

### Database Metrics
- Records per second insertion rate
- Query execution time
- Schema creation time
- Database size and growth

### Environment Metrics
- Memory usage during test execution
- Disk space consumption
- Environment setup/teardown time
- Resource cleanup efficiency

## Future Enhancements

Planned improvements for the test data management system:

1. **Advanced Content Generation**
   - Realistic document content with proper formatting
   - Image generation with specific characteristics
   - Media file generation with metadata

2. **Network-Based Testing**
   - Remote file system simulation
   - Network database testing
   - Distributed test data generation

3. **Cloud Integration**
   - Cloud storage test scenarios
   - Scalable test data generation
   - Remote test environment provisioning

4. **AI-Powered Data Generation**
   - Machine learning-based realistic data generation
   - Intelligent duplicate scenario creation
   - Adaptive performance dataset sizing

5. **Enhanced Monitoring**
   - Real-time resource monitoring
   - Performance regression detection
   - Automated performance baseline management