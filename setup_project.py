import os
from pathlib import Path
import shutil

def create_directory_structure():
    """Create all necessary directories"""
    directories = [
        'config',
        'data/raw',
        'data/processed',
        'models/trained_models',
        'logs',
        'src/data',
        'src/forecasting',
        'src/utils',
        'app',
        'notebooks',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create __init__.py files in Python package directories
        if directory.startswith('src'):
            init_file = Path(directory) / '__init__.py'
            init_file.touch()

def create_file_placeholders():
    """Create all necessary Python files with basic imports and structure"""
    files = {
        # Config
        'config/config.yaml': '''
paths:
  data_dir: "data/raw"
  processed_dir: "data/processed"
  models_dir: "models/trained_models"
  predictions_dir: "predictions"
  logs_dir: "logs"

data:
  date_col: "date"
  target_col: "quantity"
  id_cols: ["store_id", "item_id"]

forecasting:
  historical_window: 90
  forecast_horizon: 30
  train_end_date: "2024-01-01"
  val_end_date: "2024-02-01"
''',
        
        # Source files
        'src/data/data_loader.py': '''
import pandas as pd
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger('retail_forecast')

class DataLoader:
    def __init__(self, config: Dict):
        self.config = config
        # TODO: Implement data loading logic
''',
        
        'src/data/data_validator.py': '''
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger('retail_forecast')

class DataValidator:
    def __init__(self, config: Dict):
        self.config = config
        # TODO: Implement data validation logic
''',
        
        'src/forecasting/time_series_processor.py': '''
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger('retail_forecast')

class TimeSeriesProcessor:
    def __init__(self, config: Dict):
        self.config = config
        # TODO: Implement time series processing logic
''',
        
        'src/forecasting/feature_creator.py': '''
import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger('retail_forecast')

class FeatureCreator:
    def __init__(self, config: Dict):
        self.config = config
        # TODO: Implement feature engineering logic
''',
        
        'src/forecasting/forecasting_model.py': '''
import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger('retail_forecast')

class BaseForecastingModel:
    def __init__(self, config: Dict):
        self.config = config
        # TODO: Implement model logic
''',
        
        'src/utils/logger.py': '''
import logging
from pathlib import Path
from datetime import datetime

def setup_logger(logs_dir: str) -> logging.Logger:
    # TODO: Implement logger setup
    pass
''',
        
        'src/utils/config.py': '''
import yaml
from pathlib import Path
from typing import Dict

def load_config(config_path: str = "config/config.yaml") -> Dict:
    # TODO: Implement config loading
    pass
''',
        
        'src/utils/evaluation.py': '''
import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger('retail_forecast')

class ForecastEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        # TODO: Implement evaluation logic
''',
        
        # App
        'app/app.py': '''
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.append('..')

class DemandForecastApp:
    def __init__(self):
        # TODO: Implement app logic
        pass

    def run(self):
        st.title('Retail Demand Forecasting')
        # TODO: Implement app interface

if __name__ == "__main__":
    app = DemandForecastApp()
    app.run()
''',
        
        # Main script
        'main.py': '''
from pathlib import Path
from src.utils.config import load_config
from src.utils.logger import setup_logger
import logging

def main():
    # Load configuration
    config = load_config()
    logger = setup_logger(config['paths']['logs_dir'])
    logger.info("Starting demand forecasting pipeline")
    
    try:
        # TODO: Implement main pipeline
        pass
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
''',
        
        # Requirements
        'requirements.txt': '''
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
lightgbm==4.0.0
xgboost==2.0.0
catboost==1.2.1
optuna==3.3.0
streamlit==1.24.0
plotly==5.15.0
joblib==1.3.2
pyyaml==6.0.1
python-dateutil==2.8.2
pyarrow==13.0.0
''',
        
        # Docker files
        'Dockerfile': '''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
''',
        
        'docker-compose.yml': '''
version: '3'

services:
  retail-forecast:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app
'''
    }
    
    for file_path, content in files.items():
        file = Path(file_path)
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, 'w') as f:
            f.write(content.lstrip())

def main():
    """Initialize project structure"""
    print("Creating project structure...")
    create_directory_structure()
    
    print("Creating file placeholders...")
    create_file_placeholders()
    
    print("\nProject structure created successfully!")
    print("\nNext steps:")
    print("1. Create virtual environment:")
    print("   python -m venv venv")
    print("2. Activate virtual environment:")
    print("   source venv/bin/activate  # Linux/Mac")
    print("   venv\\Scripts\\activate     # Windows")
    print("3. Install requirements:")
    print("   pip install -r requirements.txt")
    print("4. Place your data files in data/raw/")
    print("5. Run the pipeline:")
    print("   python main.py")
    print("6. Start the web app:")
    print("   streamlit run app/app.py")

if __name__ == "__main__":
    main()