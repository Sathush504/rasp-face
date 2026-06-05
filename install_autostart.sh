#!/bin/bash
# install_autostart.sh — Autostart setup script for Smart Door Access System.

set -e

echo "====================================================="
echo "  Smart Door Access System - Autostart Setup Script  "
echo "====================================================="

# Get the absolute directory of the project
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
AUTOSTART_DIR="$HOME/.config/autostart"

# 1. Create standard autostart folder if it does not exist
mkdir -p "$AUTOSTART_DIR"

# 2. Write the .desktop file
cat <<EOF > "$AUTOSTART_DIR/face_unlock.desktop"
[Desktop Entry]
Type=Application
Name=Smart Door Access System
Comment=Launches Face Recognition Smart Lock GUI on graphical startup
Exec=$PROJECT_DIR/.venv/bin/python $PROJECT_DIR/face_unlock_gui.py
Path=$PROJECT_DIR
Terminal=false
Categories=Utility;
X-GNOME-Autostart-enabled=true
EOF

# Make sure desktop entry has execute permission
chmod +x "$AUTOSTART_DIR/face_unlock.desktop"

echo "✓ Created autostart entry at: $AUTOSTART_DIR/face_unlock.desktop"
echo "✓ Executable path set to: $PROJECT_DIR/.venv/bin/python"
echo "✓ Project directory set to: $PROJECT_DIR"
echo "-----------------------------------------------------"
echo "Success! The GUI app will launch automatically when the graphical desktop loads."
echo "====================================================="
