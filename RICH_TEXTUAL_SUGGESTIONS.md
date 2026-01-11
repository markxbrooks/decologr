# Rich and Textual Integration Suggestions for decologr

## Overview

[Rich](https://github.com/Textualize/rich) and [Textual](https://github.com/Textualize/textual) are powerful Python libraries that could significantly enhance decologr's visual output and user experience. Rich provides beautiful terminal formatting, while Textual enables interactive terminal user interfaces (TUIs).

## Rich Integration Suggestions

### 1. **Enhanced Console Output Formatting**

**Current State**: Basic text formatting with emojis
**Rich Enhancement**: Color-coded log levels, syntax highlighting, and structured panels

**Benefits**:
- Color-coded log levels (INFO=blue, WARNING=yellow, ERROR=red, CRITICAL=bold red)
- Syntax highlighting for JSON, Python code snippets, and file paths
- Panel borders and backgrounds for better visual separation
- Markdown rendering support in log messages

**Example Implementation**:
```python
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax

# Color-coded log levels
LEVEL_COLORS = {
    logging.DEBUG: "dim white",
    logging.INFO: "blue",
    logging.WARNING: "yellow",
    logging.ERROR: "red",
    logging.CRITICAL: "bold red on white"
}
```

### 2. **Improved JSON Logging**

**Current State**: Compact JSON on single line
**Rich Enhancement**: Pretty-printed JSON with syntax highlighting and collapsible structure

**Benefits**:
- Syntax-highlighted JSON output
- Pretty-printed formatting (optional)
- Collapsible nested structures in TUI mode
- Better readability for complex data structures

**Example**:
```python
from rich.json import JSON

def json(self, data, pretty=True, silent=False):
    if pretty:
        console.print(JSON.from_data(data))
    else:
        # Compact mode for file logging
        compact_json = json.dumps(data, separators=(",", ":"))
        console.print(compact_json)
```

### 3. **Structured Parameter Display**

**Current State**: Text-based parameter logging with truncation
**Rich Enhancement**: Tables, trees, and formatted displays for complex data

**Benefits**:
- Tables for dictionary/list data
- Tree structures for nested data
- Progress bars for long-running operations
- Better visualization of NumPy arrays, DataFrames, etc.

**Example**:
```python
from rich.table import Table
from rich.tree import Tree

def parameter(self, message, parameter, ...):
    if isinstance(parameter, dict):
        table = Table(title=message)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        for k, v in parameter.items():
            table.add_row(str(k), str(v))
        console.print(table)
```

### 4. **Enhanced Traceback Formatting**

**Current State**: Basic exception logging
**Rich Enhancement**: Beautiful, syntax-highlighted tracebacks

**Benefits**:
- Syntax-highlighted code in tracebacks
- Line numbers and file paths clearly marked
- Collapsible stack frames
- Better error context visualization

**Example**:
```python
from rich.traceback import Traceback

def error(self, message, exception=None, ...):
    if exception:
        console.print(f"[red]{message}[/red]")
        console.print(Traceback.from_exception(type(exception), exception, exception.__traceback__))
```

### 5. **Progress Indicators**

**Current State**: No progress tracking
**Rich Enhancement**: Progress bars, spinners, and status indicators

**Benefits**:
- Progress bars for long operations
- Spinners for indeterminate progress
- Status indicators for multi-step processes
- Time estimates and ETA

**Example**:
```python
from rich.progress import Progress, SpinnerColumn, TextColumn

def log_progress(self, task_description, total=None):
    # Returns a progress context manager
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
```

### 6. **Header Messages Enhancement**

**Current State**: Simple separator lines
**Rich Enhancement**: Rich panels with borders, colors, and icons

**Benefits**:
- Visually distinct header panels
- Customizable borders and styles
- Icon support (beyond emojis)
- Better visual hierarchy

**Example**:
```python
from rich.panel import Panel

def header_message(self, message, level=logging.INFO):
    style = LEVEL_COLORS.get(level, "blue")
    panel = Panel(
        message,
        title="[bold]Log Header[/bold]",
        border_style=style,
        expand=False
    )
    console.print(panel)
```

### 7. **Markdown Support**

**Current State**: Plain text only
**Rich Enhancement**: Markdown rendering in log messages

**Benefits**:
- Rich text formatting (bold, italic, links)
- Code blocks with syntax highlighting
- Lists and tables in log messages
- Better documentation in logs

## Textual Integration Suggestions

### 1. **Interactive Log Viewer TUI**

**Feature**: Real-time log monitoring dashboard

**Benefits**:
- Live log streaming with auto-scroll
- Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Search functionality
- Color-coded log entries
- Scrollable history

**Use Case**:
```bash
decologr view --file ~/.decologr/logs/myproject-*.log
# Opens interactive TUI log viewer
```

### 2. **Log File Browser**

**Feature**: Browse and navigate log files

**Benefits**:
- List all log files in `.decologr/logs/`
- Sort by date, size, project name
- Preview log files
- Open in full viewer
- Delete old logs

**Use Case**:
```bash
decologr browse
# Opens TUI file browser for log directory
```

### 3. **Log Level Configuration UI**

**Feature**: Interactive log level configuration

**Benefits**:
- Visual log level selector
- Real-time preview of filtered logs
- Save preferences
- Per-project settings

**Use Case**:
```bash
decologr config
# Opens TUI configuration interface
```

### 4. **Real-time Log Monitor**

**Feature**: Monitor logs from running applications

**Benefits**:
- Watch multiple log files simultaneously
- Filter and highlight specific patterns
- Alert on error patterns
- Export filtered logs

**Use Case**:
```bash
decologr monitor --project myproject
# Opens TUI monitoring dashboard
```

### 5. **Log Statistics Dashboard**

**Feature**: Visual statistics and analytics

**Benefits**:
- Log level distribution charts
- Error frequency over time
- Most common log messages
- Performance metrics visualization

**Use Case**:
```bash
decologr stats --file ~/.decologr/logs/myproject-*.log
# Shows statistics dashboard
```

## Implementation Strategy

### Phase 1: Rich Integration (Low Risk, High Value)
1. Add Rich as optional dependency
2. Enhance console handler with Rich Console
3. Implement color-coded log levels
4. Improve JSON and parameter formatting
5. Add traceback formatting

### Phase 2: Rich Advanced Features (Medium Risk, Medium Value)
1. Progress bars and spinners
2. Markdown support
3. Enhanced header messages
4. Table and tree displays

### Phase 3: Textual TUI Tools (Higher Risk, High Value)
1. Log viewer TUI
2. Log file browser
3. Configuration UI
4. Real-time monitor

## Configuration Options

### Rich Configuration
```python
# In setup_logging or new configure_rich_logging()
rich_config = {
    "use_rich": True,  # Enable Rich formatting
    "color_system": "auto",  # auto, standard, 256, truecolor
    "force_terminal": False,  # Force terminal mode
    "no_color": False,  # Disable colors
    "markup": True,  # Enable markup in messages
    "highlight": True,  # Enable syntax highlighting
}
```

### Textual Configuration
```python
# For TUI tools
textual_config = {
    "theme": "default",  # Color theme
    "auto_scroll": True,  # Auto-scroll in viewer
    "refresh_rate": 0.1,  # Update frequency
    "max_lines": 10000,  # Max lines in buffer
}
```

## Backward Compatibility

- Rich/Textual should be **optional dependencies**
- Default behavior should remain unchanged if Rich is not installed
- Feature flags to enable/disable Rich formatting
- File logging should remain plain text (or optionally formatted)

## Example Usage After Integration

```python
from decologr import Decologr, setup_logging

# Setup with Rich enabled
logger = setup_logging(use_rich=True, project_name="myproject")

# Rich-formatted output
Decologr.info("Starting process...")  # Blue, formatted
Decologr.json({"key": "value"})  # Syntax-highlighted JSON
Decologr.parameter("Config", {"setting": "value"})  # Table format
Decologr.error("Failed", exception=ValueError("test"))  # Red with traceback

# TUI tools
# decologr view --file logs/app.log
# decologr monitor --project myproject
```

## Benefits Summary

### Rich Benefits
- ✅ Better visual appeal and readability
- ✅ Syntax highlighting for code/JSON
- ✅ Color-coded log levels
- ✅ Progress indicators
- ✅ Enhanced error display
- ✅ Minimal API changes required

### Textual Benefits
- ✅ Interactive log viewing
- ✅ Real-time monitoring
- ✅ Better log file management
- ✅ Configuration UI
- ✅ Statistics and analytics
- ✅ Professional tooling feel

## Dependencies

```toml
[project.optional-dependencies]
rich = ["rich>=13.0.0"]
textual = ["textual>=0.40.0"]
all = ["rich>=13.0.0", "textual>=0.40.0"]
```

## Migration Path

1. **Stage 1**: Add Rich as optional dependency, enhance console output
2. **Stage 2**: Add Rich formatting to all log methods
3. **Stage 3**: Add CLI commands for Textual TUI tools
4. **Stage 4**: Add configuration system for Rich/Textual options

## Conclusion

Rich and Textual would transform decologr from a functional logging utility into a visually appealing and interactive logging solution. Rich provides immediate visual improvements with minimal changes, while Textual enables powerful interactive tools for log management and monitoring.
