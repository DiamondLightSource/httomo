from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("httomo")
except PackageNotFoundError:
    # package not installed
    pass
