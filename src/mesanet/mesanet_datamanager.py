from src.mesanet.mesanet_dataset import WeatherBench2Dataset
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
import xarray as xr


# =============================================================================
# IMPROVED DATA MANAGER
# =============================================================================

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
        
        # Test connection first
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to WeatherBench2 and validate variables"""
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
            raise
    
    def create_datasets(self, 
                       time_range: slice = slice("2015", "2023"),
                       normalize: bool = True) -> Tuple[WeatherBench2Dataset, WeatherBench2Dataset, WeatherBench2Dataset]:
        """
        Create train, validation, and test datasets
        
        Returns:
            train_dataset, val_dataset, test_dataset
        """
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
        
        logger.info("Datasets created successfully")
        logger.info(f"Train: {len(train_dataset)} samples")
        logger.info(f"Val: {len(val_dataset)} samples") 
        logger.info(f"Test: {len(test_dataset)} samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, 
                           datasets: Tuple[WeatherBench2Dataset, WeatherBench2Dataset, WeatherBench2Dataset],
                           batch_size: int = 32,
                           num_workers: int = 2) -> Tuple:
        """
        Create PyTorch data loaders with error handling
        """
        from torch.utils.data import DataLoader
        
        train_dataset, val_dataset, test_dataset = datasets
        
        # Custom collate function to handle errors
        def safe_collate(batch):
            try:
                return torch.utils.data.dataloader.default_collate(batch)
            except Exception as e:
                logger.error(f"Collate error: {e}")
                # Return a dummy batch
                dummy_input = torch.zeros(len(batch), 12, 4, 181, 301)
                dummy_target = torch.zeros(len(batch), 4, 181, 301)
                dummy_geo = torch.zeros(len(batch), 4, 181, 301)
                return dummy_input, dummy_target, dummy_geo
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=safe_collate
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=safe_collate
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=safe_collate
        )
        
        return train_loader, val_loader, test_loader