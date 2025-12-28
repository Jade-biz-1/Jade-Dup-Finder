# Distribution Package Setup

**Date:** November 2025  
**Status:** ✅ Implemented  
**Purpose:** Enable distribution of pre-built binaries with checksum verification

---

## Overview

CloneClean now includes automated checksum generation and comprehensive distribution documentation to enable secure distribution of pre-built binaries through the repository.

## Changes Implemented

### 1. Automated Checksum Generation

**File:** `scripts/build.py`

**Added Functions:**
- `generate_checksums(file_path)` - Generates SHA256 and MD5 checksums
- `write_checksum_files(file_path, checksums)` - Writes `.sha256` and `.md5` files

**Integration:**
- Checksums are automatically generated when artifacts are copied to `dist/`
- Each package gets two checksum files:
  - `<package>.sha256` - SHA256 checksum (recommended)
  - `<package>.md5` - MD5 checksum (legacy compatibility)

**Format:**
```
<hash> *<filename>
```

This format is compatible with standard verification tools (`sha256sum`, `md5sum`, etc.)

### 2. Distribution Documentation

**File:** `dist/README.md`

**Contents:**
- Package availability by platform (Windows/Linux/macOS)
- Checksum verification instructions for all platforms
- Installation instructions for each package format:
  - Windows: `.exe` NSIS installer
  - Linux: `.deb`, `.rpm`, `.tgz`
  - macOS: `.dmg`
- Running instructions
- System requirements
- Troubleshooting guide
- Support information

### 3. Root README Updates

**File:** `README.md`

**Added Section:** "📥 Download Pre-Built Binaries"
- Links to `dist/` folder
- Quick download instructions
- Reference to detailed `dist/README.md`

### 4. Git Configuration

**File:** `.gitattributes` (NEW)
- Proper handling of binary files
- Line ending normalization for text files
- Checksum files marked as text
- Prepared for Git LFS (commented out)

**File:** `.gitignore` (UPDATED)
- `dist/` folder is now tracked (not ignored)
- Binary packages in `dist/` are explicitly allowed
- Build artifacts elsewhere remain ignored
- Checksum files are tracked

---

## Usage

### Building with Checksums

Checksums are generated automatically during the build process:

```bash
# Standard build - checksums generated automatically
python scripts/build.py --target linux-ninja-cpu --build-type Release

# Output includes:
# - Package: dist/Linux/Release/cloneclean-1.0.0-linux-x86_64-cpu.deb
# - Checksum: dist/Linux/Release/cloneclean-1.0.0-linux-x86_64-cpu.deb.sha256
# - Checksum: dist/Linux/Release/cloneclean-1.0.0-linux-x86_64-cpu.deb.md5
```

### Verifying Checksums

**Linux/macOS:**
```bash
cd dist/Linux/Release
sha256sum -c cloneclean-1.0.0-linux-x86_64-cpu.deb.sha256
```

**Windows (PowerShell):**
```powershell
cd dist\Win64\Release
$expected = Get-Content cloneclean-1.0.0-win64-msvc-cpu.exe.sha256 | Select-Object -First 1 | ForEach-Object { $_.Split()[0] }
$actual = (Get-FileHash cloneclean-1.0.0-win64-msvc-cpu.exe -Algorithm SHA256).Hash
$expected -eq $actual
```

---

## Distribution Workflow

### For Maintainers

1. **Build Release Packages:**
   ```bash
   python scripts/build.py --target <platform> --build-type Release
   ```

2. **Verify Checksums:**
   ```bash
   cd dist/<Platform>/Release
   sha256sum -c *.sha256
   ```

3. **Commit to Repository:**
   ```bash
   git add dist/
   git commit -m "Release: v1.0.0 packages with checksums"
   git push
   ```

4. **Create GitHub Release:**
   - Tag the release: `git tag v1.0.0`
   - Push tag: `git push origin v1.0.0`
   - Create release on GitHub
   - Attach packages from `dist/` folder

### For Users

1. **Clone Repository or Download Release:**
   ```bash
   git clone https://github.com/yourusername/cloneclean.git
   cd cloneclean/dist
   ```

2. **Verify Checksum:**
   ```bash
   sha256sum -c <package>.sha256
   ```

3. **Install Package:**
   - Follow instructions in `dist/README.md`

---

## File Structure

```
dist/
├── README.md                          # Distribution guide
├── Win64/
│   ├── Debug/
│   │   ├── cloneclean-1.0.0-win64-msvc-cpu.exe
│   │   ├── cloneclean-1.0.0-win64-msvc-cpu.exe.sha256
│   │   └── cloneclean-1.0.0-win64-msvc-cpu.exe.md5
│   └── Release/
│       ├── cloneclean-1.0.0-win64-msvc-cpu.exe
│       ├── cloneclean-1.0.0-win64-msvc-cpu.exe.sha256
│       └── cloneclean-1.0.0-win64-msvc-cpu.exe.md5
├── Linux/
│   ├── Debug/
│   │   ├── cloneclean-1.0.0-linux-x86_64-cpu.deb
│   │   ├── cloneclean-1.0.0-linux-x86_64-cpu.deb.sha256
│   │   ├── cloneclean-1.0.0-linux-x86_64-cpu.deb.md5
│   │   ├── cloneclean-1.0.0-linux-x86_64-cpu.rpm
│   │   ├── cloneclean-1.0.0-linux-x86_64-cpu.rpm.sha256
│   │   ├── cloneclean-1.0.0-linux-x86_64-cpu.rpm.md5
│   │   ├── cloneclean-1.0.0-linux-x86_64-cpu.tgz
│   │   ├── cloneclean-1.0.0-linux-x86_64-cpu.tgz.sha256
│   │   └── cloneclean-1.0.0-linux-x86_64-cpu.tgz.md5
│   └── Release/
│       └── (same structure as Debug)
└── MacOS/
    ├── X64/
    │   ├── Debug/
    │   │   ├── cloneclean-1.0.0-macos-x86_64.dmg
    │   │   ├── cloneclean-1.0.0-macos-x86_64.dmg.sha256
    │   │   └── cloneclean-1.0.0-macos-x86_64.dmg.md5
    │   └── Release/
    │       └── (same structure as Debug)
    └── ARM/
        ├── Debug/
        │   ├── cloneclean-1.0.0-macos-arm64.dmg
        │   ├── cloneclean-1.0.0-macos-arm64.dmg.sha256
        │   └── cloneclean-1.0.0-macos-arm64.dmg.md5
        └── Release/
            └── (same structure as Debug)
```

---

## Security Considerations

### Why Two Hash Algorithms?

- **SHA256:** Cryptographically secure, recommended for verification
- **MD5:** Legacy compatibility, some tools still use it

### Checksum File Format

Standard format compatible with GNU coreutils:
```
<hash> *<filename>
```

The asterisk (`*`) indicates binary mode, which is standard for package verification.

### Verification Best Practices

1. **Always verify checksums** before installation
2. **Use SHA256** when possible (more secure than MD5)
3. **Download from trusted sources** (official repository or releases)
4. **Check file sizes** match expected values
5. **Verify digital signatures** if available (future enhancement)

---

## Future Enhancements

### Planned Improvements

1. **GPG Signatures:**
   - Add GPG signing of packages
   - Provide public key for verification
   - Document GPG verification process

2. **Automated Release Process:**
   - GitHub Actions workflow for releases
   - Automatic checksum generation
   - Automatic GitHub release creation

3. **Download Statistics:**
   - Track package downloads
   - Popular platform analytics
   - Version adoption metrics

4. **Mirror Support:**
   - Multiple download locations
   - CDN integration
   - Faster downloads worldwide

5. **Delta Updates:**
   - Incremental update packages
   - Reduce download size for updates
   - Faster update process

---

## Testing

### Manual Testing Checklist

- [ ] Build package for each platform
- [ ] Verify checksums are generated
- [ ] Verify checksum files have correct format
- [ ] Test checksum verification on each platform
- [ ] Test installation from packages
- [ ] Verify application runs after installation
- [ ] Test uninstallation process

### Automated Testing

Future: Add CI/CD pipeline to:
- Build packages automatically
- Generate checksums
- Verify checksums
- Test installation in clean environments

---

## Troubleshooting

### Checksum Mismatch

**Symptom:** Checksum verification fails

**Possible Causes:**
1. File was corrupted during download
2. File was modified after checksum generation
3. Wrong checksum file used

**Solution:**
1. Re-download the package
2. Verify you're using the correct checksum file
3. Check file size matches expected value

### Missing Checksum Files

**Symptom:** `.sha256` or `.md5` files not found

**Possible Causes:**
1. Old build without checksum generation
2. Files not committed to repository
3. Incomplete download

**Solution:**
1. Rebuild with latest `build.py` script
2. Check git status: `git status dist/`
3. Re-download or re-clone repository

---

## References

- **Build System Overview:** `docs/BUILD_SYSTEM_OVERVIEW.md`
- **Distribution README:** `dist/README.md`
- **Root README:** `README.md`
- **SHA256 Standard:** FIPS 180-4
- **MD5 Standard:** RFC 1321 (legacy)

---

**Last Updated:** November 2025  
**Maintainer:** CloneClean Team
