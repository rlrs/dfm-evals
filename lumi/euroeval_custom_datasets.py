"""Custom EuroEval dataset configs for local runs.

This file is loaded via EuroEval's `--custom-datasets-file` option.
"""

from euroeval.data_models import DatasetConfig
from euroeval.languages import DANISH
from euroeval.tasks import LA


DALA_CONFIG = DatasetConfig(
    name="dala",
    pretty_name="DaLA",
    source="giannor/dala",
    task=LA,
    languages=[DANISH],
    unofficial=True,
)

