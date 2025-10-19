#include "test_base.h"
#include "test_data_generator.h"
#include "test_database_manager.h"
#include <QTest>
#include <QDebug>
#include <QDir>
#include <QFileInfo>

/**
 * @brief Example test class demonstrating the test data management system
 * 
 * This example shows how to use the TestDataGenerator, TestEnvironmentIsolator,
 * and TestDatabaseManager for comprehensive test data management.
 */
DECLARE_TEST_CLASS(TestDataManagementExample, Unit, High, "data-management", "framework", "example")

private slots:
    void initTestCase() {
        TestBase::initTestCase();
        logTestInfo("Setting up test data management example");
    }

    void cleanupTestCase() {
        logTestInfo("Cleaning up test data management example");
        TestBase::cleanupTestCase();
    }

    // Test file and directory generation
    TEST_METHOD(test_dataGenerator_createTestFiles_generatesCorrectStructure) {
        logTestStep("Testing file and directory generation");
        
        TestDataGenerator generator;
        
        // Test single file generation
        TestDataGenerator::FileSpec fileSpec;
        fileSpec.fileName = "test_document.txt";
        fileSpec.extension = "txt";
        fileSpec.sizeBytes = 1024;
        fileSpec.content = "This is a test document for validation.";
        
        QString filePath = generator.generateTestFile(fileSpec);
        TEST_VERIFY_WITH_MSG(!filePath.isEmpty(), "File should be generated successfully");
        TEST_VERIFY_WITH_MSG(QFile::exists(filePath), "Generated file should exist");
        
        // Verify file content
        QFile file(filePath);
        TEST_VERIFY_WITH_MSG(file.open(QIODevice::ReadOnly), "File should be readable");
        QString content = file.readAll();
        TEST_COMPARE_WITH_MSG(content, fileSpec.content, "File content should match specification");
        file.close();
        
        // Test directory generation
        TestDataGenerator::DirectorySpec dirSpec;
        dirSpec.name = "test_directory";
        dirSpec.depth = 2;
        dirSpec.filesPerDirectory = 5;
        dirSpec.subdirectories = 3;
        dirSpec.fileTypes = {"txt", "dat", "log"};
        dirSpec.minFileSize = 512;
        dirSpec.maxFileSize = 2048;
        
        QString dirPath = generator.generateTestDirectory(dirSpec);
        TEST_VERIFY_WITH_MSG(!dirPath.isEmpty(), "Directory should be generated successfully");
        TEST_VERIFY_WITH_MSG(QDir(dirPath).exists(), "Generated directory should exist");
        
        // Verify directory structure
        bool structureValid = generator.verifyGeneratedStructure(dirPath, dirSpec);
        TEST_VERIFY_WITH_MSG(structureValid, "Generated directory structure should match specification");
        
        logTestStep("File and directory generation test completed successfully");
    }

    TEST_METHOD(test_dataGenerator_createDuplicateFiles_generatesDuplicatesCorrectly) {
        logTestStep("Testing duplicate file generation");
        
        TestDataGenerator generator;
        
        // Create original file
        TestDataGenerator::FileSpec originalSpec;
        originalSpec.fileName = "original_file.txt";
        originalSpec.content = "This is the original file content for duplication testing.";
        
        QString originalPath = generator.generateTestFile(originalSpec);
        TEST_VERIFY_WITH_MSG(!originalPath.isEmpty(), "Original file should be created");
        
        // Create duplicates
        QStringList duplicates = generator.createDuplicateSet(originalPath, 3);
        TEST_COMPARE_WITH_MSG(duplicates.size(), 3, "Should create 3 duplicate files");
        
        // Verify duplicates have same content
        QFile originalFile(originalPath);
        TEST_VERIFY_WITH_MSG(originalFile.open(QIODevice::ReadOnly), "Original file should be readable");
        QByteArray originalContent = originalFile.readAll();
        originalFile.close();
        
        for (const QString& duplicatePath : duplicates) {
            TEST_VERIFY_WITH_MSG(QFile::exists(duplicatePath), "Duplicate file should exist");
            
            QFile duplicateFile(duplicatePath);
            TEST_VERIFY_WITH_MSG(duplicateFile.open(QIODevice::ReadOnly), "Duplicate file should be readable");
            QByteArray duplicateContent = duplicateFile.readAll();
            duplicateFile.close();
            
            TEST_COMPARE_WITH_MSG(duplicateContent, originalContent, "Duplicate content should match original");
        }
        
        logTestStep("Duplicate file generation test completed successfully");
    }

    TEST_METHOD(test_dataGenerator_scenarioGeneration_createsRealisticDatasets) {
        logTestStep("Testing scenario-based dataset generation");
        
        TestDataGenerator generator;
        
        // Test different scenarios
        QList<TestDataGenerator::TestScenario> scenarios = {
            TestDataGenerator::TestScenario::SmallDataset,
            TestDataGenerator::TestScenario::MediumDataset,
            TestDataGenerator::TestScenario::DeepHierarchy,
            TestDataGenerator::TestScenario::DuplicateHeavy
        };
        
        for (TestDataGenerator::TestScenario scenario : scenarios) {
            logTestStep(QString("Testing scenario: %1").arg(static_cast<int>(scenario)));
            
            QStringList files = generator.generateFileHierarchy(scenario);
            TEST_VERIFY_WITH_MSG(!files.isEmpty(), "Scenario should generate files");
            
            // Analyze generated data
            if (!files.isEmpty()) {
                QFileInfo firstFile(files.first());
                QString scenarioPath = firstFile.absolutePath();
                
                QMap<QString, qint64> analysis = generator.analyzeGeneratedData(scenarioPath);
                TEST_VERIFY_WITH_MSG(analysis["fileCount"].toLongLong() > 0, "Should have generated files");
                TEST_VERIFY_WITH_MSG(analysis["totalSize"].toLongLong() > 0, "Should have non-zero total size");
                
                logTestInfo(QString("Scenario analysis - Files: %1, Total Size: %2 bytes")
                           .arg(analysis["fileCount"].toLongLong())
                           .arg(analysis["totalSize"].toLongLong()));
            }
        }
        
        logTestStep("Scenario generation test completed successfully");
    }

    TEST_METHOD(test_dataGenerator_performanceDataset_handlesLargeDataGeneration) {
        logTestStep("Testing performance dataset generation");
        
        // Skip this test in CI to avoid long execution times
        skipIfCI("Performance dataset generation test is too slow for CI");
        
        TestDataGenerator generator;
        startPerformanceMeasurement("large_dataset_generation");
        
        // Generate performance dataset (smaller size for testing)
        QString datasetPath = generator.generatePerformanceDataset(10 * 1024 * 1024); // 10MB
        
        qint64 generationTime = stopPerformanceMeasurement("large_dataset_generation");
        
        TEST_VERIFY_WITH_MSG(!datasetPath.isEmpty(), "Performance dataset should be generated");
        TEST_VERIFY_WITH_MSG(QDir(datasetPath).exists(), "Performance dataset directory should exist");
        
        // Analyze the generated dataset
        QMap<QString, qint64> analysis = generator.analyzeGeneratedData(datasetPath);
        qint64 totalSize = analysis["totalSize"].toLongLong();
        qint64 fileCount = analysis["fileCount"].toLongLong();
        
        TEST_VERIFY_WITH_MSG(totalSize > 5 * 1024 * 1024, "Dataset should be at least 5MB"); // Allow some variance
        TEST_VERIFY_WITH_MSG(fileCount > 10, "Dataset should have multiple files");
        
        logTestInfo(QString("Generated %1 files with total size %2 bytes in %3ms")
                   .arg(fileCount).arg(totalSize).arg(generationTime));
        
        recordPerformanceMetric("dataset_generation_rate", totalSize / (generationTime + 1), "bytes/ms");
        
        logTestStep("Performance dataset generation test completed successfully");
    }

    TEST_METHOD(test_environmentIsolator_createIsolatedEnvironment_providesIsolation) {
        logTestStep("Testing environment isolation");
        
        TestEnvironmentIsolator isolator;
        
        // Create isolated environment
        QString envId = isolator.createIsolatedEnvironment("test_isolation");
        TEST_VERIFY_WITH_MSG(!envId.isEmpty(), "Should create isolated environment");
        
        QString envPath = isolator.getEnvironmentPath(envId);
        TEST_VERIFY_WITH_MSG(!envPath.isEmpty(), "Should provide environment path");
        TEST_VERIFY_WITH_MSG(QDir(envPath).exists(), "Environment directory should exist");
        
        // Test resource monitoring
        isolator.startResourceMonitoring(envId);
        
        // Simulate some work in the isolated environment
        TestDataGenerator generator;
        generator.setTemporaryDirectory(QDir(envPath).absoluteFilePath("temp"));
        
        TestDataGenerator::DirectorySpec spec;
        spec.name = "isolated_test_data";
        spec.filesPerDirectory = 10;
        spec.subdirectories = 2;
        
        QString testDataPath = generator.generateTestDirectory(spec);
        TEST_VERIFY_WITH_MSG(!testDataPath.isEmpty(), "Should generate test data in isolated environment");
        
        // Stop monitoring and get resource usage
        isolator.stopResourceMonitoring(envId);
        QMap<QString, QVariant> resourceUsage = isolator.getResourceUsage(envId);
        
        TEST_VERIFY_WITH_MSG(resourceUsage.contains("executionTimeMs"), "Should track execution time");
        TEST_VERIFY_WITH_MSG(resourceUsage["executionTimeMs"].toLongLong() > 0, "Should have positive execution time");
        
        // Cleanup
        isolator.destroyIsolatedEnvironment(envId);
        TEST_VERIFY_WITH_MSG(!QDir(envPath).exists(), "Environment should be cleaned up");
        
        logTestStep("Environment isolation test completed successfully");
    }

    TEST_METHOD(test_databaseManager_createTestDatabase_setupsCorrectSchema) {
        logTestStep("Testing database management");
        
        TestDatabaseManager dbManager;
        
        // Create test database
        QString connectionName = dbManager.createScenarioDatabase(
            TestDatabaseManager::DatabaseScenario::Configuration
        );
        TEST_VERIFY_WITH_MSG(!connectionName.isEmpty(), "Should create test database");
        TEST_VERIFY_WITH_MSG(TestDatabaseManager::isDatabaseConnected(connectionName), 
                           "Database should be connected");
        
        // Verify schema
        QStringList tables = dbManager.getTableNames(connectionName);
        TEST_VERIFY_WITH_MSG(!tables.isEmpty(), "Database should have tables");
        
        logTestInfo(QString("Created database with tables: %1").arg(tables.join(", ")));
        
        // Test data population
        TestDatabaseManager::TestDataSpec dataSpec;
        dataSpec.tableName = "settings"; // Assuming settings table exists
        
        QMap<QString, QVariant> record1;
        record1["key"] = "test_setting_1";
        record1["value"] = "test_value_1";
        record1["category"] = "test";
        
        QMap<QString, QVariant> record2;
        record2["key"] = "test_setting_2";
        record2["value"] = "test_value_2";
        record2["category"] = "test";
        
        dataSpec.records = {record1, record2};
        
        bool insertSuccess = dbManager.insertTestData(connectionName, dataSpec);
        if (insertSuccess) {
            // Verify data insertion
            QString query = "SELECT COUNT(*) as count FROM settings WHERE category = 'test'";
            auto results = dbManager.selectData(connectionName, query);
            TEST_VERIFY_WITH_MSG(!results.isEmpty(), "Should return query results");
            
            int recordCount = results.first()["count"].toInt();
            TEST_COMPARE_WITH_MSG(recordCount, 2, "Should have inserted 2 test records");
        } else {
            logTestWarning("Could not insert test data - table may not exist in this scenario");
        }
        
        // Generate database report
        QString report = dbManager.generateDatabaseReport(connectionName);
        TEST_VERIFY_WITH_MSG(!report.isEmpty(), "Should generate database report");
        logTestInfo("Database report generated successfully");
        
        logTestStep("Database management test completed successfully");
    }

    TEST_METHOD(test_dataManagement_integratedWorkflow_worksEndToEnd) {
        logTestStep("Testing integrated data management workflow");
        
        // Create isolated environment
        TestEnvironmentIsolator isolator;
        QString envId = isolator.createIsolatedEnvironment("integrated_test");
        QString envPath = isolator.getEnvironmentPath(envId);
        
        // Start resource monitoring
        isolator.startResourceMonitoring(envId);
        
        // Generate test file system
        TestDataGenerator fileGenerator;
        fileGenerator.setTemporaryDirectory(QDir(envPath).absoluteFilePath("files"));
        
        QStringList generatedFiles = fileGenerator.generateFileHierarchy(
            TestDataGenerator::TestScenario::SmallDataset
        );
        TEST_VERIFY_WITH_MSG(!generatedFiles.isEmpty(), "Should generate test files");
        
        // Create test database
        TestDatabaseManager dbManager;
        QString dbConnection = dbManager.createScenarioDatabase(
            TestDatabaseManager::DatabaseScenario::ScanResults
        );
        TEST_VERIFY_WITH_MSG(!dbConnection.isEmpty(), "Should create test database");
        
        // Simulate populating database with file scan results
        if (!generatedFiles.isEmpty()) {
            QMap<QString, QString> fileHashes;
            for (const QString& filePath : generatedFiles.mid(0, 5)) { // Test with first 5 files
                QString hash = TestDataGenerator::calculateFileHash(filePath);
                if (!hash.isEmpty()) {
                    fileHashes[filePath] = hash;
                }
            }
            
            if (!fileHashes.isEmpty()) {
                bool populateSuccess = dbManager.populateScanResults(dbConnection, 
                                                                   fileHashes.keys(), 
                                                                   fileHashes);
                if (populateSuccess) {
                    logTestInfo(QString("Populated database with %1 file records").arg(fileHashes.size()));
                } else {
                    logTestWarning("Could not populate scan results - table may not exist");
                }
            }
        }
        
        // Stop monitoring and analyze resource usage
        isolator.stopResourceMonitoring(envId);
        QMap<QString, QVariant> resourceUsage = isolator.getResourceUsage(envId);
        
        qint64 executionTime = resourceUsage["executionTimeMs"].toLongLong();
        TEST_VERIFY_WITH_MSG(executionTime > 0, "Should track execution time");
        
        recordPerformanceMetric("integrated_workflow_time", executionTime, "ms");
        
        // Generate comprehensive report
        QMap<QString, qint64> fileAnalysis = fileGenerator.analyzeGeneratedData(
            QDir(envPath).absoluteFilePath("files")
        );
        QString dbReport = dbManager.generateDatabaseReport(dbConnection);
        
        logTestInfo("Integrated workflow analysis:");
        logTestInfo(QString("- Files generated: %1").arg(fileAnalysis["fileCount"].toLongLong()));
        logTestInfo(QString("- Total file size: %1 bytes").arg(fileAnalysis["totalSize"].toLongLong()));
        logTestInfo(QString("- Execution time: %1 ms").arg(executionTime));
        
        // Cleanup
        isolator.destroyIsolatedEnvironment(envId);
        
        logTestStep("Integrated data management workflow test completed successfully");
    }

END_TEST_CLASS()

/**
 * @brief Main function for running the test data management example
 */
int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);
    
    qDebug() << "========================================";
    qDebug() << "Test Data Management System Example";
    qDebug() << "========================================";
    
    // Load test configuration
    TestConfig::instance().loadConfiguration();
    
    // Create and run the test
    TestDataManagementExample test;
    
    if (test.shouldRunTest()) {
        int result = QTest::qExec(&test, argc, argv);
        
        if (result == 0) {
            qDebug() << "✅ Test data management example PASSED";
        } else {
            qDebug() << "❌ Test data management example FAILED";
        }
        
        return result;
    } else {
        qDebug() << "⏭️  Test data management example SKIPPED (disabled by configuration)";
        return 0;
    }
}

#include "example_test_data_management.moc"