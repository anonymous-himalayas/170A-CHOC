"""
Module for the DataPartitioner class.

Handles loading and partitioning of the UDS dataset into mutually
exclusive subsets driven entirely by a user-supplied partition config.
Each partition is defined by an attribute name, an output filename,
and a filter callable — making the class agnostic to the number,
naming, and logic of its partitions.

Default partitions:
    - choc:         CHOC data only (no SVI, no COI)
    - choc_svi:     CHOC + SVI data (no COI)
    - choc_svi_coi: CHOC + SVI + COI data (all features present)
"""
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.DEBUG) -> None:
    """Configure the root logger's format and verbosity level.

    Should be called once at the application entry point. Subsequent
    calls have no effect due to basicConfig's idempotent behavior.

    Args:
        level: The logging level threshold. Messages below this level
            are suppressed. Defaults to logging.DEBUG.

    Example:
        >>> setup_logging(level=logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Defined at module level to avoid the mutable-default-argument pitfall
DEFAULT_PARTITION_CONFIG: dict[str, dict] = {
    "choc": {
        "filename": "choc.csv",
        "filter": lambda df: df[df["RPL_THEME1"].isna()].dropna(axis=1),
    },
    "choc_svi": {
        "filename": "choc_svi.csv",
        "filter": lambda df: df[
            df["r_HE_nat"].isna() & ~df["RPL_THEME1"].isna()
        ].dropna(axis=1),
    },
    "choc_svi_coi": {
        "filename": "choc_svi_coi.csv",
        "filter": lambda df: df[
            ~df["r_HE_nat"].isna() & ~df["RPL_THEME1"].isna()
        ].dropna(axis=1),
    },
}


class DataPartitioner:
    """Loads or creates partitioned subsets of the UDS dataset.

    On first run, reads a master CSV file and splits it into mutually
    exclusive subsets according to the filter callables defined in
    partition_config, then persists them as individual CSV files. On
    subsequent runs, the pre-partitioned files are loaded directly from
    disk, skipping the partitioning step entirely.

    The number, names, and logic of partitions are fully controlled by
    partition_config, making the class agnostic to any specific schema.

    Attributes:
        data_dir (Path): Directory containing all data files.
        main_df_path (Path): Full path to the source CSV file.
        partition_config (dict): Config mapping each attribute name to
            its filename and filter callable.
        file_map (dict[str, Path]): Maps each attribute name to its
            resolved output Path.
        <attr> (pd.DataFrame): One DataFrame attribute is registered
            per key in partition_config. e.g. self.choc, self.choc_svi.

    Example:
        >>> dp = DataPartitioner(data_dir="data")
        >>> dp.load_datasets()
        >>> dp.choc.head()

        >>> # Custom partitions:
        >>> config = {
        ...     "group_a": {
        ...         "filename": "group_a.csv",
        ...         "filter": lambda df: df[df["col"].isna()].dropna(axis=1),
        ...     },
        ...     "group_b": {
        ...         "filename": "group_b.csv",
        ...         "filter": lambda df: df[~df["col"].isna()].dropna(axis=1),
        ...     },
        ... }
        >>> dp = DataPartitioner(data_dir="data", partition_config=config)
        >>> dp.load_datasets()
        >>> dp.group_a.head()
    """

    def __init__(
        self,
        data_dir: str = "../data",
        main_csv_filename: str = "uds_data.csv",
        partition_config: dict[str, dict] | None = None,
    ) -> None:
        """Initialize DataPartitioner with directory and partition configuration.

        Args:
            data_dir: Path to the directory containing data files.
                Defaults to "../data".
            main_csv_filename: Filename of the master source CSV to
                partition from. Defaults to "uds_data.csv".
            partition_config: A dict mapping attribute names to a nested
                dict with two required keys:
                    - "filename" (str): Output CSV filename for this partition.
                    - "filter" (Callable[[pd.DataFrame], pd.DataFrame]):
                        A function that accepts the raw DataFrame and returns
                        the filtered partition.
                Defaults to None, which applies DEFAULT_PARTITION_CONFIG.
        """
        self.data_dir = Path(data_dir)
        self.main_df_path = self.data_dir / main_csv_filename
        self.partition_config = partition_config or DEFAULT_PARTITION_CONFIG

        # Map each attr name to its resolved output Path
        self.file_map: dict[str, Path] = {
            attr: self.data_dir / cfg["filename"]
            for attr, cfg in self.partition_config.items()
        }

        # Dynamically register each partition as an empty DataFrame
        for attr in self.partition_config:
            setattr(self, attr, pd.DataFrame())

        logger.debug(
            "DataPartitioner initialized | data_dir=%s | partitions=%s",
            self.data_dir,
            list(self.partition_config.keys()),
        )

    def load_datasets(self) -> None:
        """Load partition DataFrames from disk or build them from the source CSV.

        Checks whether all partition files defined in file_map already exist.
        If they do, reads them directly into their corresponding instance
        attributes. If any are missing, delegates to create_files() to rebuild
        all partitions from the master CSV, then persists them via save_files().
        """
        if all(path.exists() for path in self.file_map.values()):
            logger.info("Existing partition files found — loading from disk.")
            for attr, path in self.file_map.items():
                setattr(self, attr, pd.read_csv(path))
                logger.debug("Loaded '%s' | path=%s", attr, path)
        else:
            logger.warning(
                "Partition files not found under '%s'. Rebuilding from source.",
                self.data_dir,
            )
            self.create_files()
            logger.info("Writing partitioned dataframes to disk.")
            self.save_files()

    def create_files(self) -> None:
        """Partition the master CSV into DataFrames using configured filter callables.

        Reads the source CSV from main_df_path, then applies each partition's
        filter callable to produce the corresponding DataFrame, storing it as
        an instance attribute under the configured name.

        Raises:
            FileNotFoundError: If main_df_path does not exist.
            KeyError: If a filter callable references a column absent from
                the source CSV.
        """
        logger.info("Reading source dataframe from '%s'", self.main_df_path)
        raw_df = pd.read_csv(self.main_df_path)
        logger.debug("Source dataframe loaded | shape=%s", raw_df.shape)

        total = len(self.partition_config)
        for i, (attr, cfg) in enumerate(self.partition_config.items(), start=1):
            filtered_df = cfg["filter"](raw_df)
            setattr(self, attr, filtered_df)
            logger.debug(
                "Partition [%d/%d] ('%s') created | shape=%s",
                i, total, attr, filtered_df.shape,
            )

    def save_files(self) -> None:
        """Persist all partition DataFrames to their configured CSV paths.

        Iterates over file_map, writing each partition's DataFrame to its
        corresponding output path. Existing files will be overwritten.
        Row indices are not written to the output files.

        Raises:
            OSError: If data_dir does not exist or is not writable.
        """
        for attr, path in self.file_map.items():
            df = getattr(self, attr)
            df.to_csv(path, index=False)
            logger.info("Saved '%s' | shape=%-15s -> %s", attr, str(df.shape), path)


if __name__ == "__main__":
    setup_logging(level=logging.DEBUG)  # Switch to logging.INFO in production
    dp = DataPartitioner(data_dir="data")
    dp.load_datasets()
