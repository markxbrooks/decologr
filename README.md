# decologr - Decorative Logger

A logging utility with emoji decorations and structured message formatting.

## Features

- **Emoji Decorations**: Automatic emoji decorations based on log level and message content
- **Structured Logging**: Support for JSON, parameters, and formatted messages
- **Project-Aware**: Configurable project name for log file naming
- **Backward Compatible**: Static method interface for easy migration

## Installation

```bash
pip install decologr
```

Or install from source:

```bash
cd decologr
pip install -e .
```

## Quick Start

```python
from decologr import Logger, set_project_name, setup_logging

# Set project name (optional, defaults to "decologr")
set_project_name("myproject")

# Setup logging (optional)
setup_logging(verbose=True, project_name="myproject")

# Use Logger
Logger.info("Hello, world!")
Logger.error("Something went wrong", exception=ValueError("test"))
Logger.json({"key": "value"})
```

## Usage

### Basic Logging

```python
from decologr import Logger

Logger.info("Information message")
Logger.warning("Warning message")
Logger.error("Error message")
Logger.debug("Debug message")
```

### Logging with Exceptions

```python
try:
    result = 1 / 0
except ZeroDivisionError as e:
    Logger.error("Division failed", exception=e)
```

### JSON Logging

```python
data = {"user": "alice", "action": "login"}
Logger.json(data)
```

### Parameter Logging

```python
Logger.parameter("User ID", user_id)
Logger.parameter("Settings", {"theme": "dark", "lang": "en"})
```

### Header Messages

```python
Logger.header_message("Starting Processing")
```

## Configuration

### Setting Project Name

```python
from decologr import set_project_name, get_project_name

set_project_name("myproject")
print(get_project_name())  # "myproject"
```

### Setup Logging

```python
from decologr import setup_logging

# Setup with default project name
logger = setup_logging()

# Setup with custom project name
logger = setup_logging(project_name="myproject", verbose=True)
```

## API Reference

### Logger Class

All methods are static methods:

- `Logger.info(message, *args, level=logging.INFO, stacklevel=3, silent=False)`
- `Logger.debug(message, *args, level=logging.DEBUG, stacklevel=3, silent=False)`
- `Logger.warning(message, *args, exception=None, level=logging.WARNING, stacklevel=4, silent=False)`
- `Logger.error(message, *args, exception=None, level=logging.ERROR, stacklevel=4, silent=False)`
- `Logger.json(data, silent=False)`
- `Logger.parameter(message, parameter, float_precision=2, max_length=300, level=logging.INFO, stacklevel=4, silent=False)`
- `Logger.header_message(message, level=logging.INFO, silent=False, stacklevel=3)`
- `Logger.debug_info(successes, failures, stacklevel=3)`

### Functions

- `setup_logging(verbose=False, project_name="decologr")` - Setup logging configuration
- `cleanup_logging(logger)` - Clean up logging handlers
- `set_project_name(project_name)` - Set the project name for logging
- `get_project_name()` - Get the current project name

## License

MIT License

## Author

Part of the mxflask project.

