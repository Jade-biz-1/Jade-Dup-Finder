# CloneClean - Distribution Packages

Welcome to CloneClean! This directory contains pre-built binaries for Windows, Linux, and macOS.
**Tagline:** *One File. One Place.*

## 📦 Available Packages

### Windows
- **Location:** `Win64/Release/` or `Win64/Debug/`
- **Format:** `.exe` (NSIS Installer)
- **Variants:**
  - `CloneClean-*-win64-msvc-cpu.exe` - CPU-only version (recommended for most users)
  - `CloneClean-*-win64-msvc-cuda.exe` - GPU-accelerated version (requires NVIDIA GPU)
  - `CloneClean-*-win64-mingw-cpu.exe` - MinGW build (alternative toolchain)

### Linux
- **Location:** `Linux/Release/` or `Linux/Debug/`
- **Formats:**
  - `.deb` - Debian/Ubuntu packages
  - `.rpm` - RedHat/Fedora/CentOS packages
  - `.tgz` - Universal tarball (portable)
- **Variants:**
  - `CloneClean-*-linux-x86_64-cpu.*` - CPU-only version
  - `CloneClean-*-linux-x86_64-gpu.*` - GPU-accelerated version (requires NVIDIA GPU)

### macOS
- **Location:** `MacOS/X64/Release/` (Intel) or `MacOS/ARM/Release/` (Apple Silicon)
- **Format:** `.dmg` (Disk Image)
- **Architectures:**
  - `CloneClean-*-macos-x86_64.dmg` - Intel Macs
  - `CloneClean-*-macos-arm64.dmg` - Apple Silicon (M1/M2/M3/M4)

---

## 🔐 Verifying Package Integrity

Each package includes SHA256 and MD5 checksum files for verification.

### Windows (PowerShell)

```powershell
# Verify SHA256
$expectedHash = Get-Content CloneClean-1.0.0-windows.exe.sha256 | Select-Object -First 1 | ForEach-Object { $_.Split()[0] }
$actualHash = (Get-FileHash CloneClean-1.0.0-windows.exe -Algorithm SHA256).Hash
if ($expectedHash -eq $actualHash) { Write-Host "✓ Checksum verified!" -ForegroundColor Green } else { Write-Host "✗ Checksum mismatch!" -ForegroundColor Red }
```

### Linux / macOS

```bash
# Verify SHA256 (recommended)
sha256sum -c CloneClean-1.0.0-linux.deb.sha256

# Verify MD5 (alternative)
md5sum -c CloneClean-1.0.0-linux.deb.md5
```

**Expected output:** `CloneClean-1.0.0-linux.deb: OK`

---

## 💿 Installation Instructions

### Windows

1. **Download** the appropriate `.exe` installer
2. **Verify** the checksum (see above)
3. **Run** the installer by double-clicking
4. **Follow** the installation wizard
5. **Launch** from Start Menu or Desktop shortcut

**Default Installation Path:** `C:\Program Files\CloneClean\`

**Uninstall:** Use "Add/Remove Programs" in Windows Settings

### Linux - Debian/Ubuntu (.deb)

```bash
# Download and verify
wget https://github.com/yourusername/cloneclean/releases/download/v1.0.0/CloneClean-1.0.0-linux.deb
sha256sum -c CloneClean-1.0.0-linux.deb.sha256

# Install
sudo dpkg -i CloneClean-1.0.0-linux.deb

# If dependencies are missing, run:
sudo apt-get install -f

# Launch
cloneclean
```

**Uninstall:**
```bash
sudo apt remove cloneclean
```

### Linux - RedHat/Fedora/CentOS (.rpm)

```bash
# Download and verify
wget https://github.com/yourusername/cloneclean/releases/download/v1.0.0/CloneClean-1.0.0-linux.rpm
sha256sum -c CloneClean-1.0.0-linux.rpm.sha256

# Install (Fedora/RHEL 8+)
sudo dnf install CloneClean-1.0.0-linux.rpm

# Install (CentOS 7/RHEL 7)
sudo yum install CloneClean-1.0.0-linux.rpm

# Launch
cloneclean
```

**Uninstall:**
```bash
sudo dnf remove cloneclean  # or: sudo yum remove cloneclean
```

### Linux - Portable (.tgz)

```bash
# Download and verify
wget https://github.com/yourusername/cloneclean/releases/download/v1.0.0/CloneClean-1.0.0-linux.tgz
sha256sum -c CloneClean-1.0.0-linux.tgz.sha256

# Extract
tar -xzf CloneClean-1.0.0-linux.tgz

# Run directly (no installation needed)
cd CloneClean-1.0.0-linux
./bin/cloneclean

# Optional: Create symlink for system-wide access
sudo ln -s $(pwd)/bin/cloneclean /usr/local/bin/cloneclean
```

### macOS

1. **Download** the appropriate `.dmg` file for your Mac
   - Intel Macs: `CloneClean-*-macos-x86_64.dmg`
   - Apple Silicon: `CloneClean-*-macos-arm64.dmg`

2. **Verify** the checksum:
   ```bash
   shasum -a 256 -c CloneClean-1.0.0-macos-arm64.dmg.sha256
   ```

3. **Open** the `.dmg` file by double-clicking

4. **Drag** the CloneClean icon to the Applications folder

5. **Launch** from Applications or Spotlight

**First Launch:** macOS may show a security warning. Go to System Preferences → Security & Privacy → General and click "Open Anyway"

**Uninstall:** Drag CloneClean from Applications to Trash

---

## 🚀 Running CloneClean

### First Launch

1. **Select Scan Location:** Choose folders to scan for duplicates
2. **Configure Options:** Set minimum file size, detection algorithm, etc.
3. **Start Scan:** Click "Start Scan" to begin
4. **Review Results:** View duplicate groups and recommendations
5. **Take Action:** Delete, move, or keep files as needed

### Command Line Usage (Advanced)

```bash
# Linux/macOS
cloneclean --help
cloneclean --scan /path/to/folder

# Windows
"C:\Program Files\CloneClean\bin\cloneclean.exe" --help
```

---

## 📋 System Requirements

### Minimum Requirements

**Windows:**
- Windows 10 or later (64-bit)
- 2 GB RAM
- 100 MB disk space
- Qt 6.4+ runtime (included in installer)

**Linux:**
- Ubuntu 20.04+ / Debian 11+ / Fedora 35+ / CentOS 8+
- 2 GB RAM
- 100 MB disk space
- Qt 6.4+ (installed automatically with package)

**macOS:**
- macOS 11 (Big Sur) or later
- 2 GB RAM
- 100 MB disk space
- Intel or Apple Silicon processor

### GPU-Accelerated Version (Optional)

**Additional Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit 11.x or later
- NVIDIA drivers (latest recommended)

**Performance Benefit:** 4-10x faster hash calculation for large files

---

## 🆘 Troubleshooting

### Windows

**Issue:** "Windows protected your PC" warning  
**Solution:** Click "More info" → "Run anyway"

**Issue:** Missing DLL errors  
**Solution:** Reinstall using the official installer (includes all dependencies)

### Linux

**Issue:** `dupfinder: command not found`  
**Solution:** Ensure `/usr/local/bin` is in your PATH, or use full path

**Issue:** Permission denied  
**Solution:** Make executable: `chmod +x /usr/local/bin/dupfinder`

**Issue:** Missing Qt libraries  
**Solution:** Install Qt6 base packages:
```bash
# Ubuntu/Debian
sudo apt install qt6-base-dev libqt6widgets6

# Fedora
sudo dnf install qt6-qtbase qt6-qtbase-gui
```

### macOS

**Issue:** "CloneClean is damaged and can't be opened"
**Solution:** Remove quarantine attribute:
```bash
xattr -cr /Applications/CloneClean.app
```

**Issue:** Application won't launch  
**Solution:** Check System Preferences → Security & Privacy → General

---

## 📚 Documentation

- **User Guide:** See `docs/` folder in the repository
- **Build Instructions:** See `docs/BUILD_SYSTEM_OVERVIEW.md`
- **Brand Guide:** See `docs/CLONECLEAN_BRAND_GUIDE.md`
- **Contributing:** See `CONTRIBUTING.md` in the repository root
- **Issues:** Report bugs at https://github.com/yourusername/cloneclean/issues

---

## 📄 License

CloneClean is released under the MIT License. See `LICENSE` file for details.

---

## 🤝 Support

- **GitHub Issues:** https://github.com/yourusername/cloneclean/issues
- **Discussions:** https://github.com/yourusername/cloneclean/discussions
- **Email:** support@cloneclean.example.com

---

**Version:** 1.0.0  
**Last Updated:** November 2025  
**Build System:** Automated with checksums for security
