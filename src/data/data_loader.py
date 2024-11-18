import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger('retail_forecast')

class DataLoader:
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['paths']['data_dir'])
        self.processed_dir = Path(config['paths']['processed_dir'])
        self.required_files = {
            'sales': 'sales.csv',
            'online': 'online.csv',
            'markdowns': 'markdowns.csv',
            'price_history': 'price_history.csv',
            'discounts': 'discounts_history.csv',
            'matrix': 'actual_matrix.csv',
            'catalog': 'catalog.csv',
            'stores': 'stores.csv',
            'test': 'test.csv'
        }
        
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load all raw data files"""
        data_dict = {}
        missing_files = []
        
        for name, filename in self.required_files.items():
            file_path = self.data_dir / filename
            try:
                df = pd.read_csv(file_path)
                df = self._initial_preprocessing(df, name)
                data_dict[name] = df
                logger.info(f"Loaded {name}: {df.shape}")
            except FileNotFoundError:
                missing_files.append(filename)
                logger.error(f"File not found: {filename}")
                
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
            
        return data_dict
    
    def _initial_preprocessing(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Initial preprocessing of loaded data"""
        df = df.copy()
        
        # Convert date columns
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        # Convert ID columns to int
        id_cols = ['store_id', 'item_id']
        for col in id_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
                
        # Handle specific dataset preprocessing
        if name in ['sales', 'online']:
            df['quantity'] = df['quantity'].clip(lower=0)
            df['price_base'] = df['price_base'].clip(lower=0)
            
        return df
    
    def save_processed_data(self, data: Dict[str, pd.DataFrame]):
        """Save processed datasets"""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in data.items():
            output_path = self.processed_dir / f"{name}.parquet"
            df.to_parquet(output_path)
            logger.info(f"Saved processed {name} to {output_path}")
            
    def load_processed_data(self, name: str) -> Optional[pd.DataFrame]:
        """Load a processed dataset"""
        file_path = self.processed_dir / f"{name}.parquet"
        
        if not file_path.exists():
            logger.warning(f"Processed file not found: {file_path}")
            return None
            
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded processed {name}: {df.shape}")
        return df
        
    def get_date_range(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Timestamp]:
        """Get date range of the data"""
        sales_data = pd.concat([data_dict['sales'], data_dict['online']])
        return {
            'start_date': sales_data['date'].min(),
            'end_date': sales_data['date'].max()
        }