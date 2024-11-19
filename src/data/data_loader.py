import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging
import yaml
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
            'test': 'test2.csv'  # Updated test file name
        }
        
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load all raw data files"""
        data_dict = {}
        missing_files = []
        
        for name, filename in self.required_files.items():
            file_path = self.data_dir / filename
            try:
                # Log the file being loaded
                logger.info(f"Loading {filename}...")
                df = pd.read_csv(file_path)
                # Log the initial shape and data types
                logger.info(f"Initial {name} shape: {df.shape}")
                logger.info(f"Initial {name} dtypes:\n{df.dtypes}")
                
                df = self._initial_preprocessing(df, name)
                data_dict[name] = df
                logger.info(f"Successfully processed {name}: {df.shape}")
            except FileNotFoundError:
                missing_files.append(filename)
                logger.error(f"File not found: {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                raise
                
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
            
        return data_dict
    
    def _initial_preprocessing(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Initial preprocessing of loaded data"""
        df = df.copy()
        
        try:
            # Convert date columns
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"Converted date column for {name}")
            
            # Handle ID columns based on the dataset
            if name in ['sales', 'online', 'stores', 'catalog']:
                # For these datasets, IDs should be numeric
                id_cols = ['store_id', 'item_id']
                for col in id_cols:
                    if col in df.columns:
                        # First check if we need to handle hex or other formats
                        if df[col].dtype == 'object':
                            # Try to convert hex to int if present
                            try:
                                df[col] = df[col].apply(lambda x: int(str(x), 16) 
                                                      if isinstance(x, str) and 'x' in str(x).lower() 
                                                      else int(x))
                            except ValueError:
                                # If conversion fails, create a mapping
                                unique_values = df[col].unique()
                                id_mapping = {val: idx for idx, val in enumerate(unique_values)}
                                df[col] = df[col].map(id_mapping)
                        else:
                            df[col] = df[col].astype(int)
                        logger.info(f"Converted {col} for {name}")
            
            # Handle specific dataset preprocessing
            if name in ['sales', 'online']:
                # Handle quantity and price
                df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).clip(lower=0)
                df['price_base'] = pd.to_numeric(df['price_base'], errors='coerce').fillna(0).clip(lower=0)
                logger.info(f"Processed quantity and price columns for {name}")
            
            # Drop unnamed columns if they exist
            unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
                logger.info(f"Dropped unnamed columns: {unnamed_cols}")
            
            # Log the final data types
            logger.info(f"Final {name} dtypes:\n{df.dtypes}")
            
        except Exception as e:
            logger.error(f"Error in preprocessing {name}: {str(e)}")
            logger.error(f"Problematic columns:\n{df.dtypes}")
            raise
            
        return df
    
    def save_processed_data(self, data: Dict[str, pd.DataFrame]):
        """Save processed datasets"""
        try:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            
            for name, df in data.items():
                output_path = self.processed_dir / f"{name}.parquet"
                try:
                    df.to_parquet(output_path, index=False)
                    logger.info(f"Saved processed {name} to {output_path}")
                except Exception as e:
                    # Try saving as CSV if parquet fails
                    csv_path = self.processed_dir / f"{name}.csv"
                    df.to_csv(csv_path, index=False)
                    logger.warning(f"Failed to save as parquet, saved {name} as CSV instead: {csv_path}")
                    logger.error(f"Parquet save error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def load_processed_data(self, name: str) -> Optional[pd.DataFrame]:
        """Load a processed dataset"""
        parquet_path = self.processed_dir / f"{name}.parquet"
        csv_path = self.processed_dir / f"{name}.csv"
        
        try:
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                logger.info(f"Loaded processed {name} from parquet: {df.shape}")
                return df
            elif csv_path.exists():
                df = pd.read_csv(csv_path)
                logger.info(f"Loaded processed {name} from csv: {df.shape}")
                return df
            else:
                logger.warning(f"Processed file not found for {name}")
                return None
        except Exception as e:
            logger.error(f"Error loading processed data {name}: {str(e)}")
            return None
        
    def get_date_range(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Timestamp]:
        """Get date range of the data"""
        sales_data = pd.concat([data_dict['sales'], data_dict['online']])
        return {
            'start_date': sales_data['date'].min(),
            'end_date': sales_data['date'].max()
        }

def load_config():
    """Load configuration from YAML file"""
    try:
        with open('config/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("Config file not found: config/config.yaml")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        
        # Initialize data loader
        logger.info("Initializing DataLoader...")
        loader = DataLoader(config)
        
        # Load raw data
        logger.info("Loading raw data files...")
        data_dict = loader.load_raw_data()
        logger.info(f"Successfully loaded {len(data_dict)} datasets")
        
        # Get date range info
        date_range = loader.get_date_range(data_dict)
        logger.info(f"Data spans from {date_range['start_date']} to {date_range['end_date']}")
        
        # Save processed data
        logger.info("Saving processed data...")
        loader.save_processed_data(data_dict)
        
        logger.info("Data loading and processing completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Error details: {str(e.__class__.__name__)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)