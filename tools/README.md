# Theme Compliance Checker

A **static analysis tool** for validating theme compliance in the CloneClean codebase.

## Why Static Analysis?

Theme compliance checking should **NOT** run during application runtime because:

1. ❌ **Wastes resources** - Uses CPU/memory during normal operation
2. ❌ **Generates noise** - Creates log spam that users don't need to see  
3. ❌ **Wrong separation** - Code quality checks belong in development, not production
4. ✅ **Static is better** - Run once during development/CI, not every app launch

## Usage

### Build the Tool

From the build directory:

```bash
cd build
cmake ..
make theme_compliance_checker
```

### Run Manually

```bash
# Check the entire src/ directory
./tools/theme_compliance_checker ../src

# Check a specific directory
./tools/theme_compliance_checker ../src/gui

# Output to a file
./tools/theme_compliance_checker ../src -o report.txt
```

### Run via CMake Target

```bash
# From build directory
make check-theme-compliance
```

### Integration Options

#### 1. Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
echo "Running theme compliance check..."
./build/tools/theme_compliance_checker ./src --quiet
if [ $? -ne 0 ]; then
    echo "❌ Theme compliance violations found!"
    echo "Run: make check-theme-compliance for details"
    exit 1
fi
```

#### 2. CI/CD Pipeline

Add to your `.github/workflows/ci.yml` or similar:

```yaml
- name: Check Theme Compliance
  run: |
    cd build
    make check-theme-compliance
```

#### 3. Manual Code Review

Run before committing major changes:

```bash
make check-theme-compliance > theme_report.txt
# Review theme_report.txt
```

## What It Checks

- ❌ **Hardcoded colors** - `#ffffff`, `rgb()`, `rgba()`
- ❌ **Hardcoded fonts** - `font-family: "Arial"`  
- ❌ **Inline styles** - Direct `setStyleSheet()` calls
- ❌ **Hardcoded sizes** - `24px`, `100px`, etc.

## Severity Levels

- **Critical**: Must fix (hardcoded colors breaking themes)
- **Warning**: Should fix (inline styles, fonts)
- **Info**: Nice to fix (hardcoded sizes)

## Example Output

```
Theme Compliance Checker
========================

Scanning directory: /home/user/cloneclean/src

Scan Complete
=============
Files scanned: 156/156
Violations found: 342

Severity breakdown:
  Critical: 12
  Warning:  145
  Info:     185

Sample violations (first 20):
==============================

[CRITICAL] /home/user/cloneclean/src/gui/main_window.cpp:523
  Type: hardcoded-color
  Line: button->setStyleSheet("background: #ff0000;");
  Fix:  Use ThemeManager::getCurrentThemeData().colors instead

...
```

## Best Practices

1. **Run during development** - Not in production
2. **Integrate with CI** - Catch violations early
3. **Fix critical first** - Colors that break themes
4. **Gradual improvement** - Address warnings over time
5. **Never in runtime** - Keep production code clean

## Comparison

### ❌ Bad: Runtime Checking
```cpp
// In MainWindow constructor
QTimer::singleShot(2000, []() {
    ThemeManager::instance()->performThemeComplianceTest(); // BAD!
});
```

### ✅ Good: Static Analysis
```bash
# In development/CI only
make check-theme-compliance
```

## Future Improvements

- [ ] Add autofix suggestions
- [ ] Generate HTML reports
- [ ] IDE integration (VS Code, Qt Creator)
- [ ] Diff-only checking (check only changed files)
- [ ] Configurable severity levels
- [ ] Custom rule definitions

## Why This Matters

**Separation of Concerns:**
- **Development tools** → Run during development/CI
- **Application code** → Run for users

Theme compliance is a **development tool**, not a **runtime feature**.
