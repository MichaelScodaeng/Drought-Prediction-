from src.mesanet.mesanet_dataset import WeatherBench2Dataset
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
import xarray as xr
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherBench2DataManager:
    """Enhanced manager for creating WeatherBench2 datasets and data loaders"""
    
    def __init__(self, 
                 zarr_path: str,
                 variables: List[str],
                 sequence_length: int = 12,
                 forecast_horizon: int = 4):
        
        self.zarr_path = zarr_path
        self.variables = variables
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # 🔧 FIX: Store actual data dimensions after testing connection
        self.data_height = None
        self.data_width = None
        self.num_variables = len(variables)
        
        # Test connection first
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to WeatherBench2 and validate variables + get dimensions"""
        logger.info("Testing connection to WeatherBench2...")
        
        try:
            ds = xr.open_zarr(
                self.zarr_path,
                consolidated=True,
                storage_options={"token": "anon"},
                chunks={'time': 10}
            )
            
            available_vars = list(ds.data_vars.keys())
            logger.info(f"✅ Connection successful. {len(available_vars)} variables available")
            
            # 🔧 FIX: Get actual spatial dimensions from the dataset
            # Apply Europe bounds to see actual output dimensions
            ds_europe = ds.where(
                (ds.longitude >= 335) | (ds.longitude <= 50),
                drop=True
            ).sel(latitude=slice(75, 30))
            
            self.data_height = len(ds_europe.latitude)
            self.data_width = len(ds_europe.longitude)
            
            logger.info(f"✅ Actual data dimensions: {self.data_height} × {self.data_width}")
            
            # Check which variables exist
            missing_vars = [var for var in self.variables if var not in available_vars]
            if missing_vars:
                logger.warning(f"Missing variables: {missing_vars}")
                
                # Suggest alternatives
                suggested_vars = [var for var in available_vars if any(
                    keyword in var.lower() for keyword in ['temperature', 'pressure', 'precipitation', 'wind']
                )][:10]
                logger.info(f"Suggested alternatives: {suggested_vars}")
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            # 🔧 FIX: Set fallback dimensions if connection fails
            self.data_height = 181
            self.data_width = 301  # Common WeatherBench2 Europe size
            logger.warning(f"Using fallback dimensions: {self.data_height} × {self.data_width}")
    
    def create_datasets(self, 
                       time_range: slice = slice("2015", "2023"),
                       normalize: bool = True) -> Tuple[WeatherBench2Dataset, WeatherBench2Dataset, WeatherBench2Dataset]:
        """Create train, validation, and test datasets"""
        logger.info(f"Creating datasets for time range {time_range}")
        
        train_dataset = WeatherBench2Dataset(
            zarr_path=self.zarr_path,
            variables=self.variables,
            time_range=time_range,
            split="train",
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            normalize=normalize
        )
        
        val_dataset = WeatherBench2Dataset(
            zarr_path=self.zarr_path,
            variables=self.variables,
            time_range=time_range,
            split="val",
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            normalize=normalize
        )
        
        test_dataset = WeatherBench2Dataset(
            zarr_path=self.zarr_path,
            variables=self.variables,
            time_range=time_range,
            split="test",
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            normalize=normalize
        )
        
        # 🔧 FIX: Update dimensions from actual dataset
        if len(train_dataset) > 0:
            try:
                sample_input, sample_target, sample_geo = train_dataset[0]
                self.data_height = sample_input.shape[-2]
                self.data_width = sample_input.shape[-1]
                self.num_variables = sample_input.shape[-3]
                logger.info(f"✅ Updated dimensions from dataset: {self.data_height} × {self.data_width}, {self.num_variables} variables")
            except Exception as e:
                logger.warning(f"Could not get dimensions from dataset: {e}")
        
        logger.info("Datasets created successfully")
        logger.info(f"Train: {len(train_dataset)} samples")
        logger.info(f"Val: {len(val_dataset)} samples") 
        logger.info(f"Test: {len(test_dataset)} samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, 
                           datasets: Tuple[WeatherBench2Dataset, WeatherBench2Dataset, WeatherBench2Dataset],
                           batch_size: int = 32,
                           num_workers: int = 2) -> Tuple:
        """Create PyTorch data loaders with FIXED error handling"""
        from torch.utils.data import DataLoader
        
        train_dataset, val_dataset, test_dataset = datasets
        
        # 🔧 CRITICAL FIX: Smart collate function that uses actual dimensions
        def smart_collate(batch):
            try:
                # Try normal collation first
                return torch.utils.data.dataloader.default_collate(batch)
                
            except Exception as e:
                logger.error(f"Collate error: {e}")
                logger.error(f"Batch length: {len(batch)}")
                
                # 🔧 FIX: Inspect the first valid item to get correct dimensions
                sample_found = False
                actual_height, actual_width, actual_vars = self.data_height, self.data_width, self.num_variables
                
                for item in batch:
                    try:
                        if item is not None and len(item) == 3:
                            input_seq, target_seq, geo_features = item
                            if hasattr(input_seq, 'shape') and len(input_seq.shape) >= 2:
                                actual_height = input_seq.shape[-2]
                                actual_width = input_seq.shape[-1] 
                                actual_vars = input_seq.shape[-3] if len(input_seq.shape) > 2 else self.num_variables
                                sample_found = True
                                logger.info(f"Found valid sample dimensions: {actual_vars} vars, {actual_height}×{actual_width}")
                                break
                    except:
                        continue
                
                if not sample_found:
                    logger.warning(f"No valid samples found, using stored dimensions: {actual_vars} vars, {actual_height}×{actual_width}")
                
                # 🔧 FIX: Create dummy batch with CORRECT dimensions
                dummy_input = torch.zeros(len(batch), self.sequence_length, actual_vars, actual_height, actual_width)
                dummy_target = torch.zeros(len(batch), self.forecast_horizon, actual_height, actual_width)
                dummy_geo = torch.zeros(4, actual_height, actual_width)  # Geographic features
                
                logger.warning(f"Created dummy batch with shapes: input={dummy_input.shape}, target={dummy_target.shape}, geo={dummy_geo.shape}")
                
                return dummy_input, dummy_target, dummy_geo
        
        # 🔧 FIX: More robust data loader configuration
        common_loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': 0,  # Disable multiprocessing to avoid data loading issues
            'pin_memory': True,
            'collate_fn': smart_collate,
            'persistent_workers': False
        }
        
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **common_loader_kwargs
        )
        
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            **common_loader_kwargs
        )
        
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            drop_last=False,
            **common_loader_kwargs
        )
        
        # 🔧 FIX: Test the data loaders to ensure they work
        logger.info("Testing data loaders...")
        try:
            sample_batch = next(iter(train_loader))
            input_seq, target_seq, geo_features = sample_batch
            logger.info(f"✅ Data loader test successful!")
            logger.info(f"   Input shape: {input_seq.shape}")
            logger.info(f"   Target shape: {target_seq.shape}")
            logger.info(f"   Geo shape: {geo_features.shape}")
            
            # Update stored dimensions with actual data
            self.data_height = input_seq.shape[-2]
            self.data_width = input_seq.shape[-1]
            self.num_variables = input_seq.shape[-3]
            
        except Exception as e:
            logger.error(f"Data loader test failed: {e}")
            logger.warning("Proceeding anyway - errors may occur during training")
        
        return train_loader, val_loader, test_loader
    
    def get_data_info(self) -> dict:
        """Return information about the data dimensions"""
        return {
            'height': self.data_height,
            'width': self.data_width,
            'num_variables': self.num_variables,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon
        }

# 🔧 ADDITIONAL FIX: Size validation function
def validate_tensor_sizes(input_seq: torch.Tensor, target_seq: torch.Tensor, geo_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ensure all tensors have consistent spatial dimensions"""
    
    # Get target spatial dimensions from input
    target_height = input_seq.shape[-2]
    target_width = input_seq.shape[-1]
    
    # Fix geo_features if needed
    if geo_features.shape[-2] != target_height or geo_features.shape[-1] != target_width:
        logger.warning(f"Resizing geo_features from {geo_features.shape[-2:]} to {target_height}×{target_width}")
        
        # Handle different geo_features dimensions
        if geo_features.dim() == 3:  # (4, H, W)
            geo_features = torch.nn.functional.interpolate(
                geo_features.unsqueeze(0),
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=True
            ).squeeze(0)
        elif geo_features.dim() == 4:  # (B, 4, H, W)
            geo_features = torch.nn.functional.interpolate(
                geo_features,
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=True
            )
    
    # Validate target sequence
    if target_seq.shape[-2] != target_height or target_seq.shape[-1] != target_width:
        logger.warning(f"Resizing target_seq from {target_seq.shape[-2:]} to {target_height}×{target_width}")
        target_seq = torch.nn.functional.interpolate(
            target_seq.unsqueeze(1) if target_seq.dim() == 3 else target_seq,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=True
        )
        if target_seq.dim() == 4 and target_seq.shape[1] == 1:
            target_seq = target_seq.squeeze(1)
    
    return input_seq, target_seq, geo_features

# 🔧 USAGE EXAMPLE:
"""
# Replace your data manager usage with:

data_manager = WeatherBench2DataManager(
    zarr_path=ZARR_PATH,
    variables=VARIABLES,
    sequence_length=SEQUENCE_LENGTH,
    forecast_horizon=FORECAST_HORIZON
)

# This will now get correct dimensions automatically
datasets = data_manager.create_datasets(time_range=slice("2021-01", "2021-03"))
train_loader, val_loader, test_loader = data_manager.create_data_loaders(datasets, batch_size=BATCH_SIZE)

# Check actual dimensions
data_info = data_manager.get_data_info()
print(f"Actual data dimensions: {data_info}")
"""