# Quick Action Buttons Debug - In Progress

**Date:** October 14, 2025  
**Issue:** Quick action preset buttons (Quick Scan, Downloads, etc.) do nothing on click  
**Status:** 🔍 DEBUGGING - Added debug output

---

## 🐛 Problem

The 6 quick action buttons on the main window don't work:
- 🚀 Start Quick Scan
- 📂 Downloads Cleanup  
- 📸 Photo Cleanup
- 📄 Documents
- 🖥️ Full System Scan
- ⭐ Custom Preset

**Expected:** Click button → Scan dialog opens with preset loaded  
**Actual:** Click button → Nothing happens

---

## 🔍 Investigation

### Code Flow (Should Work)

1. **Button Click** → `QuickActionsWidget::onQuickScanClicked()`
2. **Emit Signal** → `emit presetSelected("quick")`
3. **MainWindow Receives** → `MainWindow::onPresetSelected(const QString& preset)`
4. **Load Preset** → `m_scanSetupDialog->loadPreset(preset)`
5. **Show Dialog** → `m_scanSetupDialog->show()`

### Code Verification ✅

**QuickActionsWidget Class:**
- ✅ Has `Q_OBJECT` macro
- ✅ Signal declared: `void presetSelected(const QString& preset)`
- ✅ Slots declared and implemented
- ✅ Buttons created and connected

**MainWindow:**
- ✅ Connection made: `connect(m_quickActions, &QuickActionsWidget::presetSelected, this, &MainWindow::onPresetSelected)`
- ✅ Handler implemented: `void MainWindow::onPresetSelected(const QString& preset)`
- ✅ `loadPreset()` implemented in ScanSetupDialog

**Everything looks correct in the code!**

---

## 🔧 Debug Output Added

### 1. Button Click Detection

**File:** `src/gui/main_window.cpp`

```cpp
void QuickActionsWidget::onQuickScanClicked() { 
    qDebug() << "Quick Scan button clicked!";  // ← Added
    emit presetSelected("quick"); 
}
// ... similar for all 6 buttons
```

**Purpose:** Verify button clicks are being detected

### 2. Signal Connection Verification

**File:** `src/gui/main_window.cpp`

```cpp
if (m_quickActions) {
    qDebug() << "Connecting QuickActionsWidget signals...";  // ← Added
    bool connected = connect(m_quickActions, &QuickActionsWidget::presetSelected, 
                            this, &MainWindow::onPresetSelected);
    qDebug() << "QuickActionsWidget connection result:" << connected;  // ← Added
} else {
    qDebug() << "WARNING: m_quickActions is NULL!";  // ← Added
}
```

**Purpose:** Verify widget exists and connection succeeds

### 3. Handler Invocation

**File:** `src/gui/main_window.cpp`

```cpp
void MainWindow::onPresetSelected(const QString& preset)
{
    qDebug() << "MainWindow::onPresetSelected called with preset:" << preset;  // ← Added
    LOG_INFO(LogCategories::UI, QString("User selected preset: %1").arg(preset));
    // ...
}
```

**Purpose:** Verify handler is being called

---

## 🧪 Testing Instructions

### 1. Run Application with Debug Output

```bash
# Run from terminal to see debug output
./build/cloneclean
```

### 2. Click a Quick Action Button

Click any of the 6 buttons (e.g., "🚀 Start Quick Scan")

### 3. Check Console Output

**Expected Output (if working):**
```
Connecting QuickActionsWidget signals...
QuickActionsWidget connection result: true
Quick Scan button clicked!
MainWindow::onPresetSelected called with preset: quick
```

**Possible Failure Scenarios:**

#### Scenario A: No output at all
```
(nothing)
```
**Diagnosis:** Button click not detected  
**Cause:** Button not created or not connected  
**Fix:** Check `createButtons()` method

#### Scenario B: Button clicked but no signal
```
Quick Scan button clicked!
(nothing else)
```
**Diagnosis:** Signal not emitted or not connected  
**Cause:** Connection failed or MOC issue  
**Fix:** Check Q_OBJECT macro, rebuild MOC files

#### Scenario C: Signal emitted but handler not called
```
Quick Scan button clicked!
(no "MainWindow::onPresetSelected" message)
```
**Diagnosis:** Connection failed  
**Cause:** Signature mismatch or widget NULL  
**Fix:** Check connection result, verify widget exists

#### Scenario D: Handler called but dialog doesn't show
```
Quick Scan button clicked!
MainWindow::onPresetSelected called with preset: quick
(dialog doesn't appear)
```
**Diagnosis:** Dialog creation or show() issue  
**Cause:** Dialog not created or show() fails  
**Fix:** Check dialog creation, window management

---

## 🔍 Possible Root Causes

### 1. MOC Files Not Generated
**Symptom:** Signals don't work at all  
**Solution:** Clean rebuild
```bash
rm -rf build
mkdir build
cd build
cmake ..
cmake --build .
```

### 2. Widget Not Created
**Symptom:** "m_quickActions is NULL" message  
**Solution:** Check `createContentWidgets()` is called

### 3. Connection Timing Issue
**Symptom:** Connection result is false  
**Solution:** Ensure widget created before connection

### 4. Signal/Slot Signature Mismatch
**Symptom:** Connection succeeds but handler not called  
**Solution:** Verify exact signature match

### 5. Event Loop Issue
**Symptom:** Everything logs correctly but UI doesn't update  
**Solution:** Check event processing, modal dialogs

---

## 📋 Next Steps

### Step 1: Run and Observe
1. Run the application
2. Click a quick action button
3. Note which debug messages appear
4. Report findings

### Step 2: Diagnose Based on Output
- Match output to scenarios above
- Identify which step fails
- Apply appropriate fix

### Step 3: Additional Debug (if needed)
If no output appears, add more debug:

```cpp
// In QuickActionsWidget constructor
QuickActionsWidget::QuickActionsWidget(QWidget* parent)
    : QGroupBox(tr("Quick Actions"), parent)
{
    qDebug() << "QuickActionsWidget constructor called";
    // ...
}

// In createButtons()
void QuickActionsWidget::createButtons()
{
    qDebug() << "Creating quick action buttons...";
    // ...
    qDebug() << "Buttons created, connecting signals...";
    connect(m_quickScanButton, &QPushButton::clicked, 
            this, &QuickActionsWidget::onQuickScanClicked);
    qDebug() << "Quick Scan button connected";
    // ...
}
```

---

## 🎯 Expected Resolution

Once we see the debug output, we'll know exactly where the flow breaks and can apply the appropriate fix.

**Most Likely Causes (in order):**
1. MOC files need regeneration (clean rebuild)
2. Button connections not made (check createButtons())
3. Widget created after connections (timing issue)
4. Dialog show() issue (window management)

---

## 📝 Status

**Current State:** Debug output added, application rebuilt  
**Next Action:** Run application and observe console output  
**Waiting For:** User to test and report debug messages

---

**Debug Session Started:** October 14, 2025  
**Build Status:** ✅ Successful  
**Ready for Testing:** ✅ Yes
