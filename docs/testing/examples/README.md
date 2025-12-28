# Test Examples

This directory contains comprehensive working examples for each category of tests in the CloneClean testing suite. These examples demonstrate best practices, common patterns, and proper usage of the testing framework.

## Example Categories

### Unit Test Examples
- [Basic Unit Test](unit/basic-unit-test.cpp) - Simple component testing
- [Mock-based Test](unit/mock-based-test.cpp) - Testing with mocks and stubs
- [Exception Testing](unit/exception-testing.cpp) - Testing error conditions
- [Parameterized Test](unit/parameterized-test.cpp) - Data-driven testing

### Integration Test Examples
- [Component Integration](integration/component-integration.cpp) - Testing component interactions
- [Database Integration](integration/database-integration.cpp) - Database testing patterns
- [File System Integration](integration/filesystem-integration.cpp) - File operation testing
- [Signal-Slot Integration](integration/signal-slot-integration.cpp) - Qt signal/slot testing

### UI Test Examples
- [Widget Interaction](ui/widget-interaction.cpp) - Basic UI automation
- [Dialog Testing](ui/dialog-testing.cpp) - Modal dialog testing
- [Visual Regression](ui/visual-regression.cpp) - Screenshot comparison
- [Accessibility Testing](ui/accessibility-testing.cpp) - A11y validation

### End-to-End Test Examples
- [Complete Workflow](e2e/complete-workflow.cpp) - Full user journey testing
- [Error Recovery](e2e/error-recovery.cpp) - Error handling workflows
- [Cross-Platform](e2e/cross-platform.cpp) - Platform-specific testing
- [Performance Workflow](e2e/performance-workflow.cpp) - Performance in workflows

### Performance Test Examples
- [Benchmark Testing](performance/benchmark-testing.cpp) - Performance measurement
- [Memory Testing](performance/memory-testing.cpp) - Memory leak detection
- [Load Testing](performance/load-testing.cpp) - High-volume testing
- [Stress Testing](performance/stress-testing.cpp) - System limit testing

### Specialized Examples
- [Custom Test Framework](specialized/custom-framework.cpp) - Extending the framework
- [Test Data Generation](specialized/test-data-generation.cpp) - Dynamic test data
- [CI/CD Integration](specialized/ci-cd-integration.cpp) - Continuous integration
- [Parallel Testing](specialized/parallel-testing.cpp) - Concurrent test execution

## Usage Instructions

### Running Examples
Each example can be compiled and run independently:

```bash
# Compile a specific example
g++ -I../../include -I../../../tests/framework \
    unit/basic-unit-test.cpp -o basic-unit-test \
    -lQt6Test -lQt6Core

# Run the example
./basic-unit-test
```

### Using Examples as Templates
1. Copy the relevant example file
2. Rename classes and methods appropriately
3. Modify test logic for your specific use case
4. Update includes and dependencies
5. Add to your test suite

### Example Structure
Each example follows this structure:
- **Header comments**: Purpose and key concepts
- **Includes**: Required headers and dependencies
- **Test class**: Main test class with setup/teardown
- **Test methods**: Individual test cases
- **Helper methods**: Utility functions
- **Main function**: Test execution entry point

## Best Practices Demonstrated

### Code Organization
- Clear separation of setup, execution, and verification
- Proper use of Qt Test Framework macros
- Consistent naming conventions
- Appropriate use of helper methods

### Test Data Management
- Creation and cleanup of test data
- Use of temporary directories
- Realistic but minimal test datasets
- Proper resource management

### Error Handling
- Comprehensive error condition testing
- Proper exception handling
- Meaningful error messages
- Graceful failure handling

### Performance Considerations
- Efficient test execution
- Minimal resource usage
- Appropriate timeouts
- Memory leak prevention

## Contributing Examples

When adding new examples:
1. Follow the established structure and naming conventions
2. Include comprehensive comments explaining the concepts
3. Demonstrate a specific testing pattern or technique
4. Ensure the example compiles and runs successfully
5. Add appropriate documentation

### Example Template
```cpp
/**
 * @file example-name.cpp
 * @brief Brief description of what this example demonstrates
 * 
 * This example shows how to:
 * - Key concept 1
 * - Key concept 2
 * - Key concept 3
 * 
 * Key learning points:
 * - Important point 1
 * - Important point 2
 */

#include <QtTest>
// Additional includes...

class ExampleTest : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();    // Setup before all tests
    void init();           // Setup before each test
    void cleanup();        // Cleanup after each test
    void cleanupTestCase(); // Cleanup after all tests
    
    void testExample();    // Example test method
};

// Implementation...

QTEST_MAIN(ExampleTest)
#include "example-name.moc"
```

## Index by Concept

### Testing Patterns
- **Arrange-Act-Assert**: [Basic Unit Test](unit/basic-unit-test.cpp)
- **Given-When-Then**: [BDD Style Test](specialized/bdd-style-test.cpp)
- **Test Fixtures**: [Database Integration](integration/database-integration.cpp)
- **Test Doubles**: [Mock-based Test](unit/mock-based-test.cpp)

### Qt-Specific Testing
- **Signal/Slot Testing**: [Signal-Slot Integration](integration/signal-slot-integration.cpp)
- **Widget Testing**: [Widget Interaction](ui/widget-interaction.cpp)
- **Event Testing**: [Event Handling](ui/event-handling.cpp)
- **Threading**: [Thread Testing](specialized/thread-testing.cpp)

### Advanced Techniques
- **Parameterized Tests**: [Parameterized Test](unit/parameterized-test.cpp)
- **Custom Matchers**: [Custom Assertions](specialized/custom-assertions.cpp)
- **Test Generators**: [Generated Tests](specialized/generated-tests.cpp)
- **Property-Based Testing**: [Property Testing](specialized/property-testing.cpp)

## Support

For questions about the examples:
1. Check the comments within each example file
2. Review the related documentation in the parent directories
3. Look for similar patterns in other examples
4. Create an issue if you find bugs or have suggestions