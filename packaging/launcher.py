"""PyInstaller entry point of the installed app, SwimmerTracker.exe."""
import sys

from swimmertracker.ui.app import main

if __name__ == "__main__":
    sys.exit(main())
