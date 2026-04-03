"""SophT project."""

import logging

# avoid "No handler found" warning when the user does not configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
