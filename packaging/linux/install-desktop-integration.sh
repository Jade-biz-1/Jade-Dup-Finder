#!/bin/bash
#
# CloneClean Desktop Integration Installer
# Installs .desktop file and file manager integrations for Linux
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="cloneclean"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_warn "Running as root. Will install system-wide."
    INSTALL_MODE="system"
    DESKTOP_DIR="/usr/share/applications"
    ICON_DIR="/usr/share/icons/hicolor"
    NAUTILUS_EXT_DIR="/usr/share/nautilus-python/extensions"
else
    print_info "Running as user. Will install for current user only."
    INSTALL_MODE="user"
    DESKTOP_DIR="$HOME/.local/share/applications"
    ICON_DIR="$HOME/.local/share/icons/hicolor"
    NAUTILUS_EXT_DIR="$HOME/.local/share/nautilus-python/extensions"
fi

# Create directories if they don't exist
print_info "Creating directories..."
mkdir -p "$DESKTOP_DIR"
mkdir -p "$ICON_DIR/scalable/apps"
mkdir -p "$ICON_DIR/256x256/apps"
mkdir -p "$NAUTILUS_EXT_DIR"

# Install .desktop file
print_info "Installing .desktop file..."
if [ -f "$SCRIPT_DIR/cloneclean.desktop" ]; then
    cp "$SCRIPT_DIR/cloneclean.desktop" "$DESKTOP_DIR/"
    chmod 644 "$DESKTOP_DIR/cloneclean.desktop"
    print_info "Installed: $DESKTOP_DIR/cloneclean.desktop"
else
    print_error ".desktop file not found!"
    exit 1
fi

# Install icon (if available)
print_info "Installing application icon..."
if [ -f "$SCRIPT_DIR/../../resources/icons/cloneclean.svg" ]; then
    cp "$SCRIPT_DIR/../../resources/icons/cloneclean.svg" "$ICON_DIR/scalable/apps/"
    print_info "Installed SVG icon"
elif [ -f "$SCRIPT_DIR/../../resources/icons/cloneclean.png" ]; then
    cp "$SCRIPT_DIR/../../resources/icons/cloneclean.png" "$ICON_DIR/256x256/apps/"
    print_info "Installed PNG icon"
else
    print_warn "No application icon found. Using default."
fi

# Install Nautilus extension
if command -v nautilus &> /dev/null; then
    print_info "Installing Nautilus extension..."
    if [ -f "$SCRIPT_DIR/nautilus-cloneclean.py" ]; then
        cp "$SCRIPT_DIR/nautilus-cloneclean.py" "$NAUTILUS_EXT_DIR/"
        chmod 644 "$NAUTILUS_EXT_DIR/nautilus-cloneclean.py"
        print_info "Installed: $NAUTILUS_EXT_DIR/nautilus-cloneclean.py"
        print_info "Restart Nautilus to activate: nautilus -q"
    else
        print_warn "Nautilus extension file not found"
    fi
else
    print_warn "Nautilus not found. Skipping file manager integration."
fi

# Update desktop database
print_info "Updating desktop database..."
if [ "$INSTALL_MODE" = "system" ]; then
    if command -v update-desktop-database &> /dev/null; then
        update-desktop-database /usr/share/applications
    fi
    if command -v gtk-update-icon-cache &> /dev/null; then
        gtk-update-icon-cache -f -t /usr/share/icons/hicolor
    fi
else
    if command -v update-desktop-database &> /dev/null; then
        update-desktop-database "$DESKTOP_DIR"
    fi
    if command -v gtk-update-icon-cache &> /dev/null; then
        gtk-update-icon-cache -f -t "$ICON_DIR"
    fi
fi

print_info ""
print_info "============================================"
print_info "Desktop integration installed successfully!"
print_info "============================================"
print_info ""
print_info "You can now:"
print_info "  1. Launch CloneClean from your application menu"
print_info "  2. Right-click folders in Nautilus to find duplicates"
print_info "  3. Use 'cloneclean' command from terminal"
print_info ""

if [ "$INSTALL_MODE" = "user" ]; then
    print_info "Note: Integration installed for current user only."
    print_info "To install system-wide, run: sudo $0"
fi

print_info ""
print_info "To uninstall, run: $SCRIPT_DIR/uninstall-desktop-integration.sh"
