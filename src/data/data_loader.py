import pandas as pd
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger('retail_forecast')

class DataLoader:
    def __init__(self, config: Dict):
        self.config = config
        # TODO: Implement data loading logic
