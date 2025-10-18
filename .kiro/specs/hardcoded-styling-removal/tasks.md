# Implementation Plan

- [x] 1. Enhance ThemeManager with component-specific styling capabilities
  - Add new styling methods for progress bars, status indicators, and custom widgets
  - Implement minimum size management system with configurable constraints
  - Create style registry for centralized theme-aware styling
  - Add hardcoded style detection and override mechanisms
  - _Requirements: 1.2, 2.1, 2.2, 6.1_

- [x] 2. Fix scan_progress_dialog.cpp hardcoded styling
  - [x] 2.1 Replace hardcoded progress bar colors with theme-aware styling
    - Remove all hex color values (#4CAF50, #45a049, #2196F3, etc.)
    - Implement theme-aware progress bar styling using ThemeManager methods
    - Add performance-based color coding that adapts to current theme
    - _Requirements: 3.1, 3.3, 3.4_
  
  - [x] 2.2 Update progress bar styling system
    - Replace inline setStyleSheet calls with ThemeManager-provided styles
    - Implement dynamic style updates for performance indicators
    - Add proper minimum size constraints for progress bars
    - _Requirements: 2.4, 3.2, 6.2_

- [x] 3. Fix smart_selection_dialog.cpp hardcoded styling
  - [x] 3.1 Replace hardcoded label colors and styling
    - Remove hardcoded color values (#666, #2c3e50)
    - Update description and preview labels to use palette colors
    - Implement theme-aware italic and bold text styling
    - _Requirements: 5.1, 5.2, 5.4_
  
  - [x] 3.2 Add minimum size constraints for dialog controls
    - Set minimum sizes for sliders, labels, and buttons
    - Ensure proper spacing and padding using theme system
    - _Requirements: 6.1, 6.3, 6.5_

- [x] 4. Fix scan_scope_preview_widget.cpp hardcoded styling
  - [x] 4.1 Replace hardcoded status colors with theme-aware alternatives
    - Remove hardcoded color values (#d32f2f, #2e7d32, #f57c00)
    - Implement theme-aware status indication using palette colors
    - Update error message styling to use theme system
    - _Requirements: 3.4, 5.1, 5.2_
  
  - [x] 4.2 Update widget styling to use ThemeManager
    - Replace inline setStyleSheet calls with centralized styling
    - Implement proper border and background styling using palette
    - Add minimum size constraints for tree widget and labels
    - _Requirements: 2.4, 2.5, 6.1_

- [x] 5. Fix thumbnail_delegate.cpp custom drawing
  - [x] 5.1 Replace hardcoded drawing colors with palette colors
    - Remove hardcoded QColor values (200, 200, 200), (240, 240, 240), etc.
    - Update custom painting to use theme-aware colors from palette
    - Implement proper selection and hover state colors
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [x] 5.2 Enhance custom rendering for theme compatibility
    - Add theme change notification handling for custom delegates
    - Implement proper contrast ratios for thumbnail borders and backgrounds
    - _Requirements: 4.4, 4.5_

- [x] 6. Fix preset_manager_dialog.cpp styling system
  - [x] 6.1 Replace inline stylesheets with ThemeManager integration
    - Remove hardcoded font styling and replace with theme system
    - Update button and list widget styling to use centralized approach
    - Implement proper minimum sizes for all dialog controls
    - _Requirements: 2.4, 5.5, 6.1_

- [x] 7. Fix results_window.cpp comprehensive styling
  - [x] 7.1 Update all hardcoded styling to use theme system
    - Replace palette-based inline styles with ThemeManager methods
    - Implement proper minimum sizes for splitters, tables, and buttons
    - Add theme-aware styling for result display components
    - _Requirements: 2.1, 2.4, 6.1_

- [x] 8. Fix file_operation_progress_dialog.cpp styling
  - [x] 8.1 Replace hardcoded font styling with theme-aware alternatives
    - Remove monospace font hardcoding and use theme system
    - Implement proper minimum sizes for progress dialog controls
    - _Requirements: 5.4, 6.1_

- [x] 9. Implement comprehensive style validation system
  - [x] 9.1 Create automated hardcoded style detection
    - Implement runtime scanning for remaining hardcoded styles
    - Add validation methods to ensure theme compliance
    - Create reporting system for style inconsistencies
    - _Requirements: 2.3, 4.5_
  
  - [x] 9.2 Add comprehensive theme testing
    - Write unit tests for theme consistency across all components
    - Create automated tests for minimum size enforcement
    - Implement visual regression testing for theme switching
    - _Requirements: 1.1, 3.2, 6.2_

- [x] 10. Finalize theme system integration
  - [x] 10.1 Ensure all GUI components are properly registered with ThemeManager
    - Update component constructors to register with theme system
    - Implement proper cleanup and deregistration
    - Add comprehensive theme update notifications
    - _Requirements: 1.1, 2.3_
  
  - [x] 10.2 Validate complete theme compliance
    - Perform comprehensive testing of all theme switching scenarios
    - Verify minimum size constraints work across all components
    - Ensure no hardcoded styling remains in any GUI component
    - _Requirements: 1.1, 1.4, 6.4_