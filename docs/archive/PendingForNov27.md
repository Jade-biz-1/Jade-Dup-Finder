# Pending Tasks for November 27, 2025

**Date Created:** November 26, 2025, 11:59 PM
**Last Updated:** November 26, 2025
**Status:** Active

## Overview

This document outlines pending tasks identified during the end-of-day review on November 26, 2025. These items were discovered while reviewing `IMPLEMENTATION_TASKS.md` and organizing the documentation in the `docs/` folder.

## Priority Tasks

### 1. Complete CloneClean Branding
**Priority:** P2 (Medium)
**Estimated Effort:** 30 minutes

Remaining branding items to finalize the transition from DupFinder to CloneClean:

- [ ] **Update GitHub URLs in About Dialog**
  - Location: `src/gui/about_dialog.cpp`
  - Current: Placeholder `github.com/dupfinder/dupfinder`
  - Action: Update to actual CloneClean repository URL once repository is renamed/created

- [ ] **Database Filename Migration** (Optional)
  - Current: Settings stored as `dupfinder.db`
  - Consideration: Migrate to `cloneclean.db` for consistency
  - Impact: Need migration logic for existing users
  - Decision: Low priority - existing users may prefer continuity

- [ ] **Code Comment Updates**
  - Found: `include/performance_benchmark.h:10` still references "DupFinder"
  - Action: Update any remaining code comments with old branding

### 2. Cross-Platform Testing
**Priority:** P1 (High)
**Estimated Effort:** 2-3 hours

CloneClean has been developed and tested primarily on Linux. Before declaring Phase 3 complete, we need:

- [ ] **Windows Platform Testing**
  - Test NSIS installer generation and installation
  - Verify all UI elements render correctly on Windows
  - Test file operations and hash generation on NTFS
  - Verify theme support and settings persistence

- [ ] **macOS Platform Testing**
  - Test DMG installer generation and installation
  - Verify all UI elements render correctly on macOS
  - Test file operations on APFS
  - Verify theme support and settings persistence
  - Test integration with macOS file system dialogs

### 3. Documentation Updates
**Priority:** P2 (Medium)
**Estimated Effort:** 15 minutes

- [ ] **README.md Phase Status Update**
  - Current: Shows Phase 1 & 2 complete, Phase 3 in progress
  - Update: Explicitly state Phase 2 at 100%, Phase 3 at 85%
  - Consider: Add completed features list for Phase 3

- [ ] **Archive Evaluation** (Optional)
  - Review: `IMPLEMENTATION_TASKS.md`, `REMAINING_TASKS.md`, `PERFORMANCE_OPTIMIZATIONS.md`
  - Decision: Determine if these should remain active or be archived after Phase 3 completion

## Completed Today (November 26)

### Critical Bug Fixes
- ✅ Fixed "Force Quit/Wait" dialog when reopening Results window
  - Root cause: `applyTheme()` iterating 6607 items on every window show
  - Solution: Added `m_isTreePopulated` flag to skip unnecessary rebuilds

### CloneClean Branding (95% Complete)
- ✅ Updated all window titles from "DupFinder" to "CloneClean"
- ✅ Updated QSettings organization name
- ✅ Updated Help dialog content and title
- ✅ Updated HTML export footer
- ✅ Cleaned build folder of old DupFinder artifacts
- ✅ Updated IMPLEMENTATION_TASKS.md with CloneClean branding

### Documentation Improvements
- ✅ Added Screenshots section to README.md (4 images)
- ✅ Moved 10 completed/historical documents to `docs/archive/`
- ✅ Added T32 task to IMPLEMENTATION_TASKS.md documenting recent work
- ✅ Updated phase status: Phase 2 100%, Phase 3 85%

### Git Operations
- ✅ Created comprehensive commit on `Enhancements-Linux` branch
- ✅ Successfully pushed all changes

## Next Session Goals

1. **Immediate:** Complete remaining CloneClean branding items (GitHub URLs, code comments)
2. **Important:** Conduct Windows and macOS testing to identify any platform-specific issues
3. **Final:** Update README.md with current phase status and feature completion

## Notes

- Build system is working well with profile-based configuration
- All Linux functionality is stable and tested
- Phase 3 is 85% complete - primarily pending cross-platform testing
- Once cross-platform testing is complete, Phase 3 can be marked as 100% and we can begin Phase 4 planning

## Dependencies

- Access to Windows and macOS systems for testing
- Decision on CloneClean GitHub repository location
- Confirmation on database filename migration approach

---

**Status Legend:**
- P1 (High): Should be completed in next session
- P2 (Medium): Should be completed within a few days
- P3 (Low): Can be deferred or is optional
