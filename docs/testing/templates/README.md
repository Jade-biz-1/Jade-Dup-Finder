# Test Templates

This directory contains boilerplate templates for quickly creating new tests in the DupFinder testing suite. Each template provides a complete starting point with proper structure, includes, and placeholder methods.

## Available Templates

### Unit Test Templates
- [Basic Unit Test](unit-test-template.cpp) - Standard unit test structure
- [Mock-based Test](mock-test-template.cpp) - Template using mocks and stubs
- [Parameterized Test](parameterized-test-template.cpp) - Data-driven test template
- [Exception Test](exception-test-template.cpp) - Error condition testing

### Integration Test Templates
- [Component Integration](integration-test-template.cpp) - Component interaction testing
- [Database Integration](database-integration-template.cpp) - Database testing template
- [File System Integration](filesystem-integration-template.cpp) - File operation testing

### UI Test Templates
- [Widget Test](ui-test-template.cpp) - UI automation testing
- [Dialog Test](dialog-test-template.cpp) - Modal dialog testing
- [Visual Regression](visual-test-template.cpp) - Screenshot comparison testing

### Performance Test Templates
- [Benchmark Test](benchmark-test-template.cpp) - Performance measurement
- [Load Test](load-test-template.cpp) - High-volume testing
- [Memory Test](memory-test-template.cpp) - Memory usage testing

### Specialized Templates
- [Custom Framework](custom-framework-template.cpp) - Extending the test framework
- [CI/CD Integration](ci-integration-template.cpp) - Continuous integration testing

## Using Templates

### Quick Start
1. Copy the appropriate template file
2. Rename the file and class names
3. Replace placeholder content with your test logic
4. Update includes and dependencies
5. Add to your test suite

### Template Structure
Each template includes:
- **File header**: Purpose and usage instructions
- **Includes**: Required headers and dependencies
- **Test class**: Properly structured test class
- **Setup/teardown**: Initialization and cleanup methods
- **Test methods**: Placeholder test methods with proper naming
- **Helper methods**: Common utility functions
- **Compilation instructions**: How to build and run

### Customization Guidelines
- Replace `COMPONENT_NAME` with your actual component name
- Update `#include` statements for your specific headers
- Modify test data and setup logic for your use case
- Add or remove test methods as needed
- Update compilation instructions for your build system

## Template Conventions

### Naming Patterns
- **File names**: `component-test-template.cpp`
- **Class names**: `ComponentNameTest`
- **Test methods**: `testMethod_WhenCondition_ExpectedBehavior`
- **Helper methods**: `createTestData`, `setupEnvironment`, etc.

### Code Structure
```cpp
/**
 * Template header with usage instructions
 */

#include <QtTest>
// Additional includes...

class ComponentNameTest : public QObject {
    Q_OBJECT

private slots:
    void initTestCase();    // One-time setup
    void init();           // Per-test setup
    void cleanup();        // Per-test cleanup
    void cleanupTestCase(); // One-time cleanup
    
    // Test methods
    void testBasicFunctionality();
    void testErrorConditions();
    void testEdgeCases();

private:
    // Helper methods
    void setupTestData();
    void verifyResults();
    
    // Test data members
    ComponentType* m_component;
};

// Implementation with TODO comments...

QTEST_MAIN(ComponentNameTest)
#include "component-test-template.moc"
```

### Placeholder Patterns
Templates use these placeholder patterns:
- `COMPONENT_NAME` - Replace with actual component name
- `TODO: Implement` - Replace with actual implementation
- `// Your test logic here` - Add specific test code
- `QVERIFY(false); // TODO` - Replace with real assertions

## Code Generation Tools

### Template Generator Script
Use the provided script to generate tests from templates:

```bash
# Generate a unit test for FileScanner
./generate_test.sh unit FileScanner

# Generate an integration test for DatabaseManager
./generate_test.sh integration DatabaseManager

# Generate a UI test for MainWindow
./generate_test.sh ui MainWindow
```

### IDE Integration
Most IDEs can be configured to use these templates:

#### Qt Creator
1. Tools → Options → C++ → File Naming
2. Add template files to the template directory
3. Use File → New File → C++ Class → Test Class

#### Visual Studio Code
1. Install the "File Templates" extension
2. Configure templates in `.vscode/templates/`
3. Use Ctrl+Shift+P → "File Templates: New File from Template"

#### CLion
1. File → Settings → Editor → File and Code Templates
2. Add new template based on our templates
3. Use File → New → C++ Class → Test Class

## Best Practices

### Template Maintenance
- Keep templates up-to-date with framework changes
- Include comprehensive comments and documentation
- Provide realistic placeholder examples
- Test templates periodically to ensure they compile

### Customization Tips
- Start with the closest matching template
- Don't remove setup/teardown methods unless unnecessary
- Keep the original structure and naming conventions
- Add template-specific helper methods as needed

### Common Modifications
1. **Change test data**: Update test data creation methods
2. **Add dependencies**: Include additional headers and libraries
3. **Modify assertions**: Use appropriate QVERIFY/QCOMPARE macros
4. **Update timeouts**: Adjust waiting times for your use case
5. **Add cleanup**: Ensure proper resource cleanup

## Template Validation

### Compilation Check
All templates should compile without errors:
```bash
# Test template compilation
for template in *.cpp; do
    echo "Testing $template..."
    g++ -I/path/to/qt/include -I/path/to/framework \
        -c "$template" -o "${template%.cpp}.o"
done
```

### Static Analysis
Run static analysis on templates:
```bash
clang-tidy *.cpp -- -I/path/to/qt/include
```

## Contributing Templates

### Adding New Templates
1. Follow the established structure and naming conventions
2. Include comprehensive placeholder comments
3. Provide compilation and usage instructions
4. Test the template with a real use case
5. Update this README with the new template

### Template Requirements
- Must compile without errors (with placeholders)
- Should demonstrate best practices
- Must include proper documentation
- Should be minimal but complete
- Must follow project coding standards

### Review Checklist
- [ ] Template compiles successfully
- [ ] Includes all necessary boilerplate
- [ ] Follows naming conventions
- [ ] Has clear placeholder comments
- [ ] Includes usage instructions
- [ ] Demonstrates best practices
- [ ] Is properly documented

## Support

For template-related questions:
1. Check the template comments and documentation
2. Review the corresponding examples in `../examples/`
3. Consult the main testing documentation
4. Create an issue for template bugs or improvements

## Index

### By Test Type
- **Unit Tests**: [Basic](unit-test-template.cpp), [Mock](mock-test-template.cpp), [Parameterized](parameterized-test-template.cpp)
- **Integration Tests**: [Component](integration-test-template.cpp), [Database](database-integration-template.cpp)
- **UI Tests**: [Widget](ui-test-template.cpp), [Dialog](dialog-test-template.cpp), [Visual](visual-test-template.cpp)
- **Performance Tests**: [Benchmark](benchmark-test-template.cpp), [Load](load-test-template.cpp)

### By Complexity
- **Beginner**: Basic Unit Test, Widget Test
- **Intermediate**: Integration Test, Performance Test
- **Advanced**: Custom Framework, Visual Regression Test