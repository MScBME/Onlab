from PySide6.QtCore import Property


def ro(type_, attr: str, notify):
    """Read-only Qt property backed by a plain Python attribute."""
    return Property(type_, lambda self: getattr(self, attr), notify=notify)
