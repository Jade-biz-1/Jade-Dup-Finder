# CloneClean Brand Guide

**Version:** 1.0
**Last Updated:** 2025-11-26

## Brand Identity

### Name
**CloneClean**

### Tagline
**"One File. One Place."**

### Positioning
CloneClean is a professional, intelligent duplicate file finder and cleaner that helps users reclaim disk space and organize their digital files efficiently. The brand emphasizes simplicity, safety, and speed.

---

## Logo & Icon Design

### Primary Icon Concept: Duplicate Documents with Checkmark

**Description:**
Two document silhouettes positioned side-by-side:
- **Left document:** Solid, with a checkmark overlay (represents the file to keep)
- **Right document:** Fading/dissolving effect (represents the duplicate being removed)
- Clean, minimalist design that's instantly understandable
- Works at all sizes from 16px to 1024px

**Icon Characteristics:**
- **Style:** Flat design with subtle depth
- **Shape:** Rounded square container (iOS-style) or transparent background
- **Primary Elements:**
  - Two overlapping or adjacent document shapes
  - Checkmark on the preserved document
  - Fade/dissolve effect on the duplicate
  - Optional: Small "×2" or clone indicator

**Icon Sizes Required:**

**macOS:**
- 16×16, 32×32, 64×64, 128×128, 256×256, 512×512, 1024×1024 (all @1x and @2x)

**Linux:**
- 16×16, 22×22, 24×24, 32×32, 48×48, 64×64, 128×128, 256×256, 512×512

**Windows:**
- 16×16, 24×24, 32×32, 48×48, 256×256

---

## Color Palette

### Primary Colors

**Primary Blue** - `#2E86DE`
- RGB: 46, 134, 222
- Use: Primary actions, links, selected items, branding
- Meaning: Trust, professionalism, technology

**Success Green** - `#26DE81`
- RGB: 38, 222, 129
- Use: Success states, completion indicators, kept files
- Meaning: Positive action, health, completion

**Accent Orange** - `#FF6B35`
- RGB: 255, 107, 53
- Use: Action buttons, highlights, important CTAs
- Meaning: Energy, attention, warmth

### Supporting Colors

**Background Light** - `#F8F9FA`
- RGB: 248, 249, 250
- Use: Light mode background, panels

**Background Dark** - `#1E1E1E`
- RGB: 30, 30, 30
- Use: Dark mode background

**Text Dark** - `#2C3E50`
- RGB: 44, 62, 80
- Use: Primary text in light mode

**Text Light** - `#ECF0F1`
- RGB: 236, 240, 241
- Use: Primary text in dark mode

**Warning Red** - `#E74C3C`
- RGB: 231, 76, 60
- Use: Deletion warnings, critical actions, errors

**Border Gray** - `#CED4DA`
- RGB: 206, 212, 218
- Use: Borders, dividers, subtle UI elements

---

## Typography

### Primary Font (UI Elements)
**Inter** (preferred) or **SF Pro** (macOS native)
- Use for: Buttons, menus, labels, body text, input fields
- Weights: Regular (400), Medium (500), SemiBold (600)
- Characteristics: Modern, clean, excellent readability at all sizes

### Secondary Font (Headers & Titles)
**Poppins SemiBold** (600)
- Use for: Window titles, section headers, dialog titles, feature names
- Characteristics: Friendly yet professional, strong presence

### Monospace Font (Technical Data)
**JetBrains Mono** (preferred) or system monospace
- Use for: File paths, file sizes, hash values, technical details, logs
- Weight: Regular (400)
- Characteristics: Clear distinction between characters, modern

### Font Sizes

**Desktop Application:**
- Window Title: 16pt (Poppins SemiBold)
- Section Header: 14pt (Poppins SemiBold)
- Body Text: 13pt (Inter Regular)
- Button Text: 13pt (Inter Medium)
- Small Text/Labels: 11pt (Inter Regular)
- Technical Data: 12pt (JetBrains Mono Regular)

**Minimum Sizes:**
- Never go below 10pt for any readable text
- Small labels can be 11pt minimum

---

## Icon Style Guide

### Design Principles
1. **Clarity:** Icons should be immediately recognizable
2. **Consistency:** All icons use the same stroke width and corner radius
3. **Simplicity:** Minimal details, focus on essential shapes
4. **Scalability:** Design at largest size (512px), test at smallest (16px)

### Technical Specifications
- **Stroke Width:** 2-3px for outlines at 512px size (scales proportionally)
- **Corner Radius:** 4-8px for rounded corners at 512px
- **Padding:** 10-15% margin from icon bounds
- **Grid:** Align to pixel grid at all sizes

### Icon Colors
- Primary icons: Use Primary Blue (#2E86DE)
- Success/positive icons: Use Success Green (#26DE81)
- Warning/destructive icons: Use Warning Red (#E74C3C)
- Neutral/inactive icons: Use 50% opacity of Text color

---

## Brand Voice & Messaging

### Tone of Voice
**Professional yet approachable** - Helpful without being condescing, confident without being arrogant

### Key Characteristics
- **Clear:** Simple language, no jargon
- **Helpful:** Focuses on user benefits
- **Confident:** Trustworthy and reliable
- **Friendly:** Approachable and human

### Key Messages

#### Speed
"Scan thousands of files in seconds"
"Lightning-fast duplicate detection"
"Instant results, no waiting"

#### Safety
"Preview before you delete - never lose important files"
"Safe cleanup with built-in protection"
"Keep what matters, remove what doesn't"

#### Intelligence
"Smart algorithms find duplicates you'd miss"
"Intelligent file comparison beyond just names"
"Advanced detection: content, metadata, and more"

#### Simplicity
"Three clicks to a cleaner system"
"Clean interface, powerful results"
"Effortless organization"

---

## Application Branding Elements

### Window Titles
```
CloneClean
CloneClean - Scanning...
CloneClean - [Folder Name]
CloneClean - Results
```

### About Dialog
```
CloneClean
Version 1.0.0

One File. One Place.

Your intelligent duplicate file finder and cleaner.
© 2025 CloneClean. All rights reserved.
```

### Menu Bar / Application Menu
```
CloneClean (macOS only - application name)
  About CloneClean
  Preferences...
  Quit CloneClean
```

### Copyright Notice
```
© 2025 CloneClean. All rights reserved.
```

### Version Format
```
CloneClean v1.0.0
Version 1.0.0
1.0.0 (in logs/technical contexts)
```

---

## UI Component Styling

### Buttons

**Primary Action Button:**
- Background: Primary Blue (#2E86DE)
- Text: White
- Border Radius: 6px
- Padding: 8px 16px
- Font: Inter Medium 13pt
- Hover: Darken 10%

**Secondary Action Button:**
- Background: Transparent
- Text: Primary Blue (#2E86DE)
- Border: 1px solid Primary Blue
- Border Radius: 6px
- Padding: 8px 16px
- Font: Inter Medium 13pt
- Hover: Light blue background (#EBF5FB)

**Destructive Action Button:**
- Background: Warning Red (#E74C3C)
- Text: White
- Border Radius: 6px
- Padding: 8px 16px
- Font: Inter Medium 13pt
- Hover: Darken 10%

### Dialogs & Windows

**Dialog Background:**
- Light Mode: #FFFFFF
- Dark Mode: #2C2C2C

**Border Radius:** 8px for dialog windows

**Shadow:** Subtle drop shadow for depth
- Light Mode: 0 4px 12px rgba(0,0,0,0.15)
- Dark Mode: 0 4px 12px rgba(0,0,0,0.5)

---

## Marketing & Documentation

### Imagery Style

**Screenshots:**
- Clean, uncluttered
- Show real-world use cases
- Highlight key features with subtle annotations
- Use actual data, not Lorem Ipsum

**Photography:**
- Modern, clean workspaces
- Natural lighting
- Focus on organization and clarity
- Diverse users and scenarios

**Graphics:**
- Flat design with subtle gradients
- Consistent with icon style
- Use brand color palette
- Clean, minimal illustrations

### Use Cases to Highlight

1. **Photographer organizing photo library**
   - "Find duplicate photos across multiple import sessions"
   - Visual: Photo library with duplicates highlighted

2. **Developer cleaning build artifacts**
   - "Reclaim gigabytes from node_modules and build folders"
   - Visual: Code editor with duplicate dependencies

3. **Home user managing downloads folder**
   - "Never download the same file twice"
   - Visual: Organized downloads folder

4. **IT professional maintaining servers**
   - "Keep servers lean and efficient"
   - Visual: Server dashboard showing freed space

---

## File Naming Conventions

### Application Binaries
```
cloneclean (Linux/macOS executable)
CloneClean.exe (Windows)
CloneClean.app (macOS application bundle)
```

### Configuration Files
```
cloneclean.conf
cloneclean-settings.json
```

### Documentation
```
README.md
CLONECLEAN_BRAND_GUIDE.md
CLONECLEAN_USER_GUIDE.md
```

### Code
```
Namespace/Package: CloneClean
Class names: CloneClean::ClassName
```

---

## Don'ts - Brand Misuse

**Don't:**
- ❌ Use "Clone Clean" (two words)
- ❌ Use "clone clean" (all lowercase) in user-facing text
- ❌ Abbreviate to "CC" in branding
- ❌ Change the color palette
- ❌ Use different fonts for UI
- ❌ Add drop shadows to the logo
- ❌ Rotate or distort the logo
- ❌ Place logo on busy backgrounds
- ❌ Use outdated "duplicate file finder" generic imagery

**Do:**
- ✅ Always write as "CloneClean" (one word, CamelCase)
- ✅ Use "cloneclean" (lowercase) only in technical contexts (URLs, filenames)
- ✅ Maintain clear space around logo
- ✅ Use approved color combinations
- ✅ Keep icon simple and recognizable
- ✅ Follow typography guidelines
- ✅ Use modern, clean imagery

---

## Implementation Checklist

### Code & Configuration
- [ ] Update CMakeLists.txt project name
- [ ] Update application name in main.cpp
- [ ] Update window titles throughout UI
- [ ] Update organization name in QSettings
- [ ] Update About dialog
- [ ] Update version strings
- [ ] Update log messages
- [ ] Update error messages

### Documentation
- [ ] Update README.md
- [ ] Update all docs/ files
- [ ] Update code comments with new name
- [ ] Update build scripts
- [ ] Update LICENSE file

### Assets
- [ ] Create new icon set (all required sizes)
- [ ] Update application icon
- [ ] Create launcher icons
- [ ] Create marketing images
- [ ] Create screenshot templates

### Distribution
- [ ] Update package names
- [ ] Update installer branding
- [ ] Update desktop entry files (Linux)
- [ ] Update Info.plist (macOS)
- [ ] Update file associations

---

## Contact & Support

### Support Channels
**GitHub:** github.com/[username]/cloneclean
**Email:** support@cloneclean.com (when available)
**Website:** cloneclean.com (when available)

### Social Media Handles
**Twitter:** @CloneClean
**GitHub:** @cloneclean

---

## Version History

**1.0 - 2025-11-26**
- Initial brand guide creation
- Defined core brand elements
- Established visual identity
- Set typography and color standards
