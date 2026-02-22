"""
Example: Using Textual Log Viewer in JDXI Editor

This example shows how to integrate the decologr Textual log viewer
into the JDXI editor or any other application.
"""

from pathlib import Path
from decologr.viewer import create_log_viewer_widget, LogViewerWidget

# Example 1: Create a viewer widget for embedding
def create_viewer_for_jdxi(log_file_path: str):
    """
    Create a log viewer widget that can be embedded in JDXI editor.
    
    Args:
        log_file_path: Path to the log file to digital
    """
    log_file = Path(log_file_path).expanduser()
    widget = create_log_viewer_widget(log_file=log_file)
    return widget


# Example 2: Use the widget in a Textual app
def example_jdxi_integration():
    """
    Example of how JDXI editor might integrate the log viewer.
    """
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Container
        
        class JDXILogViewer(App):
            """Example JDXI editor integration."""
            
            def compose(self) -> ComposeResult:
                # Create the log viewer widget
                log_file = Path.home() / ".decologr" / "logs" / "myproject-*.log"
                viewer = create_log_viewer_widget(log_file=log_file)
                yield Container(viewer)
        
        # Run the app
        app = JDXILogViewer()
        app.run()
        
    except ImportError:
        print("Textual is required. Install with: pip install decologr[textual]")


# Example 3: Standalone viewer
if __name__ == "__main__":
    from decologr.viewer import run_log_viewer
    
    # Find most recent log file
    log_dir = Path.home() / ".decologr" / "logs"
    if log_dir.exists():
        log_files = sorted(
            log_dir.glob("*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if log_files:
            print(f"Viewing: {log_files[0]}")
            run_log_viewer(log_files[0])
        else:
            print("No log files found")
    else:
        print(f"Log directory not found: {log_dir}")
