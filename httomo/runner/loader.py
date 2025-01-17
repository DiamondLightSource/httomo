from typing import Protocol, Tuple

from httomo.runner.dataset_store_interfaces import DataSetSource
from httomo.preview import PreviewConfig

from httomo_backends.methods_database.query import Pattern


class LoaderInterface(Protocol):
    """Interface to a loader object"""

    # Patterns the loader supports
    pattern: Pattern = Pattern.all
    # purely informational, for use by the logger
    method_name: str
    package_name: str = "httomo"

    def make_data_source(self, padding: Tuple[int, int]) -> DataSetSource:
        """Create a dataset source that can produce padded blocks of data from the file.

        This will be called after the patterns and sections have been determined,
        just before the execution of the first section starts."""
        ...  # pragma: no cover

    @property
    def detector_x(self) -> int:
        """detector x-dimension of the loaded data"""
        ...  # pragma: no cover

    @property
    def detector_y(self) -> int:
        """detector y-dimension of the loaded data"""
        ...  # pragma: no cover

    @property
    def angles_total(self) -> int:
        """angles dimension of the loaded data"""
        ...  # pragma: no cover

    @property
    def preview(self) -> PreviewConfig:
        """get preview of the loaded data"""
        ...  # pragma: no cover

    @preview.setter
    def preview(self, preview: PreviewConfig):
        """Update preview of the loaded data"""
        ...  # pragma: no cover
