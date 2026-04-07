"""
Module for the DataPartitioner class.

Handles loading and partitioning of the UDS dataset into three
subsets based on the availability of SVI (Social Vulnerability Index)
and COI (Child Opportunity Index) feature columns:

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


class DataPartitioner:
    """Loads or creates partitioned subsets of the UDS dataset.

    On first run, reads a master CSV file and splits it into three
    mutually exclusive subsets based on the presence of SVI and COI
    columns, then persists them as individual CSV files. On subsequent
    runs, the pre-partitioned files are loaded directly from disk,
    skipping the partitioning step entirely.

    Attributes:
        data_dir (Path): Directory containing all data files.
        main_df_path (Path): Full path to the source CSV file.
        filenames (tuple[str, ...]): Filenames for the three output partitions.
        files (list[Path]): Full paths for the three output partitions.
        choc (pd.DataFrame): Partition containing CHOC-only records.
        choc_svi (pd.DataFrame): Partition containing CHOC + SVI records.
        choc_svi_coi (pd.DataFrame): Partition containing CHOC + SVI + COI records.

    Example:
        >>> dp = DataPartitioner(data_dir="data")
        >>> dp.load_datasets()
        >>> dp.choc.head()
    """

    def __init__(
        self,
        data_dir: str = "../data",
        main_csv_filename: str = "uds_data.csv",
        new_filenames: tuple = ("choc.csv", "choc_svi.csv", "choc_svi_coi.csv"),
    ) -> None:
        """Initialize DataPartitioner with directory and filename configuration.

        Args:
            data_dir: Path to the directory containing data files.
                Defaults to "../data".
            main_csv_filename: Filename of the master source CSV to
                partition from. Defaults to "uds_data.csv".
            new_filenames: A 3-tuple of output CSV filenames corresponding
                to the choc, choc_svi, and choc_svi_coi partitions respectively.
                Defaults to ("choc.csv", "choc_svi.csv", "choc_svi_coi.csv").
        """
        self.data_dir = Path(data_dir)
        self.main_df_path = self.data_dir / main_csv_filename
        self.filenames = new_filenames
        self.files = [self.data_dir / file for file in self.filenames]
        self.choc = pd.DataFrame()
        self.choc_svi = pd.DataFrame()
        self.choc_svi_coi = pd.DataFrame()
        logger.debug("DataPartitioner initialized | data_dir=%s", self.data_dir)

    def load_datasets(self) -> None:
        """Load partition DataFrames from disk or build them from the source CSV.

        Checks whether all three partition files already exist in data_dir.
        If they do, reads them directly into the instance attributes. If any
        are missing, delegates to create_files() to rebuild all partitions
        from the master CSV, then saves them via save_files().
        """
        if all(Path(file).exists() for file in self.files):
            logger.info("Existing partition files found — loading from disk.")
            self.choc, self.choc_svi, self.choc_svi_coi = (
                pd.read_csv(file) for file in self.files
            )
        else:
            logger.warning(
                "Partition files not found under '%s'. Rebuilding from source.",
                self.data_dir,
            )
            self.create_files()
            logger.info("Writing partitioned dataframes to disk.")
            self.save_files()

    def create_files(self) -> None:
        """Partition the master CSV into three mutually exclusive DataFrames.

        Reads the source CSV from main_df_path and splits rows into three
        subsets based on the presence of SVI (RPL_THEME1) and COI (r_HE_nat)
        columns. Columns that are entirely NaN within each subset are dropped.

        Partition logic:
            - choc:         RPL_THEME1 is NaN   (CHOC data only)
            - choc_svi:     RPL_THEME1 is NOT NaN AND r_HE_nat is NaN (CHOC + SVI)
            - choc_svi_coi: RPL_THEME1 is NOT NaN AND r_HE_nat is NOT NaN (CHOC + SVI + COI)

        Raises:
            FileNotFoundError: If main_df_path does not exist.
            KeyError: If RPL_THEME1 or r_HE_nat columns are absent from the source CSV.
        """
        logger.info("Reading source dataframe from '%s'", self.main_df_path)
        raw_df = pd.read_csv(self.main_df_path)
        logger.debug("Source dataframe loaded | shape=%s", raw_df.shape)

        self.choc = raw_df[raw_df["RPL_THEME1"].isna()].dropna(axis=1)
        logger.debug("Partition [1/3] (choc) created | shape=%s", self.choc.shape)

        self.choc_svi = raw_df[
            raw_df["r_HE_nat"].isna() & ~raw_df["RPL_THEME1"].isna()
        ].dropna(axis=1)
        logger.debug("Partition [2/3] (choc_svi) created | shape=%s", self.choc_svi.shape)

        self.choc_svi_coi = raw_df[
            ~raw_df["r_HE_nat"].isna() & ~raw_df["RPL_THEME1"].isna()
        ].dropna(axis=1)
        logger.debug(
            "Partition [3/3] (choc_svi_coi) created | shape=%s", self.choc_svi_coi.shape
        )

    def save_files(self) -> None:
        """Persist all three partition DataFrames to CSV files in data_dir.

        Writes self.choc, self.choc_svi, and self.choc_svi_coi to their
        corresponding paths defined in self.files. Existing files at those
        paths will be overwritten. Row indices are not written to the files.

        Raises:
            OSError: If data_dir does not exist or is not writable.
        """
        file_map = dict(zip(self.files, [self.choc, self.choc_svi, self.choc_svi_coi]))
        for path, df in file_map.items():
            df.to_csv(path, index=False)
            logger.info("Saved | shape=%-15s -> %s", str(df.shape), path)


if __name__ == "__main__":
    setup_logging(level=logging.DEBUG)  # Switch to logging.INFO in production
    dp = DataPartitioner(data_dir="data")
    dp.load_datasets()
