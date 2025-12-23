#!/usr/bin/env python3
"""
Nautilus extension for CloneClean
Adds "Find Duplicates" context menu item to folders
"""

import os
import subprocess
from gi.repository import Nautilus, GObject

class CloneCleanExtension(GObject.GObject, Nautilus.MenuProvider):
    def __init__(self):
        super().__init__()

    def _run_cloneclean(self, paths):
        """Launch CloneClean with selected paths"""
        try:
            args = ['cloneclean', '--scan'] + paths
            subprocess.Popen(args)
        except Exception as e:
            print(f"Error launching CloneClean: {e}")

    def _run_quick_scan(self, paths):
        """Launch CloneClean quick scan"""
        try:
            args = ['cloneclean', '--quick-scan'] + paths
            subprocess.Popen(args)
        except Exception as e:
            print(f"Error launching CloneClean: {e}")
    
    def menu_activate_cb(self, menu, paths):
        """Callback for menu activation"""
        self._run_cloneclean(paths)

    def quick_scan_activate_cb(self, menu, paths):
        """Callback for quick scan menu activation"""
        self._run_quick_scan(paths)

    def get_file_items(self, *args):
        """Return menu items for selected files/folders"""
        # Handle both old and new Nautilus API
        if len(args) == 1:
            # Old API: get_file_items(files)
            files = args[0]
        else:
            # New API: get_file_items(window, files)
            files = args[1]

        if not files:
            return []

        # Only show for directories
        if not all(file.is_directory() for file in files):
            return []

        # Get file paths
        paths = [file.get_location().get_path() for file in files]

        # Create menu items
        items = []

        # Main menu item
        item = Nautilus.MenuItem(
            name='CloneCleanExtension::FindDuplicates',
            label='Find Duplicates with CloneClean',
            tip='Search for duplicate files in selected folder(s)'
        )
        item.connect('activate', self.menu_activate_cb, paths)
        items.append(item)

        # Quick scan menu item
        quick_item = Nautilus.MenuItem(
            name='CloneCleanExtension::QuickScan',
            label='Quick Scan for Duplicates',
            tip='Quick scan for duplicates (common locations only)'
        )
        quick_item.connect('activate', self.quick_scan_activate_cb, paths)
        items.append(quick_item)

        return items

    def get_background_items(self, *args):
        """Return menu items for folder background"""
        # Handle both old and new Nautilus API
        if len(args) == 1:
            # Old API: get_background_items(current_folder)
            current_folder = args[0]
        else:
            # New API: get_background_items(window, current_folder)
            current_folder = args[1]

        if not current_folder:
            return []

        path = current_folder.get_location().get_path()

        # Create menu item for current folder
        item = Nautilus.MenuItem(
            name='CloneCleanExtension::FindDuplicatesHere',
            label='Find Duplicates Here',
            tip='Search for duplicates in this folder'
        )
        item.connect('activate', self.menu_activate_cb, [path])

        return [item]
