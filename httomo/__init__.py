from importlib.metadata import version, PackageNotFoundError

__all__ = ["__version__"]

try:
    __version__ = version("httomo")
except PackageNotFoundError:
    # package not installed
    pass


