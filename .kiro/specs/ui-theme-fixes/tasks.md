# Implementation Plan

- [x] 1. Create Enhanced ThemeManager System
  - Create ThemeManager singleton class with comprehensive theme management
  - Implement theme detection for system default theme
  - Create CSS stylesheet generation for light and dark themes
  - Add theme change propagation system to notify all widgets
  - _Requirements: 1.1, 1.2, 1.3, 4.1, 5.3_

- [x] 2. Fix Results Dialog Component Visibility
  - Audit ResultsWindow for missing file selection checkboxes
  - Implement proper checkbox creation in DuplicateGroupWidget
  - Apply theme-aware styling to all results dialog components
  - Fix checkbox visibility and contrast in both light and dark themes
  - Test file selection functionality and visual feedback
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [ ] 3. Fix Scan Configuration Dialog Layout
  - Increase minimum dialog size to ensure all content is visible
  - Fix tab content layout and component positioning
  - Improve spacing and alignment of all controls
  - Ensure proper text wrapping and scrolling where needed
  - Test dialog on different screen sizes and resolutions
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 4. Implement Comprehensive Dark Mode Styling
  - Create complete dark mode CSS stylesheet for all components
  - Apply dark theme colors to buttons, text fields, checkboxes, and other controls
  - Implement proper contrast ratios for text and background colors
  - Add hover effects and visual feedback for interactive elements
  - Test all dialogs and windows in dark mode for visibility issues
  - _Requirements: 1.4, 4.2, 4.4, 4.5, 4.6_

- [x] 5. Implement Theme Persistence and Settings Integration
  - Connect SettingsDialog theme selection to ThemeManager
  - Implement immediate theme application when settings change
  - Add theme preference persistence using QSettings
  - Implement system theme detection and automatic following
  - Add graceful fallback handling for theme application failures
  - _Requirements: 1.6, 5.1, 5.2, 5.4, 5.5_

- [x] 6. Apply Themes to All Dialogs and Windows
  - Update MainWindow to use ThemeManager
  - Apply themes to all existing dialogs (Settings, Scan History, etc.)
  - Ensure theme changes propagate to all open windows
  - Test theme switching with multiple dialogs open simultaneously
  - Verify consistent styling across the entire application
  - _Requirements: 4.1, 4.3, 5.3_

- [x] 7. Test and Polish Theme System
  - Comprehensive testing of light and dark themes across all components
  - Verify accessibility compliance with proper contrast ratios
  - Test theme persistence across application restarts
  - Performance testing for theme switching operations
  - Fix any remaining visual inconsistencies or bugs
  - _Requirements: All requirements verification_