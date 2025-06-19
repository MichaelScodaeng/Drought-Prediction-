from src.mesanet.mesanet_dataset import WeatherBench2Dataset
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
import xarray as xr


class WeatherBench2DataManager:
    """Manager for creating WeatherBench2 datasets and data loaders"""
    
    def __init__(self, 
                 zarr_path: str,
                 variables: List[str],
                 sequence_length: int = 12,
                 forecast_horizon: int = 4):
        
        self.zarr_path = zarr_path
        self.variables = variables
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
    
    def create_datasets(self, 
                       time_range: slice = slice("2015", "2023"),
                       normalize: bool = True) -> Tuple[WeatherBench2Dataset, WeatherBench2Dataset, WeatherBench2Dataset]:
        """
        Create train, validation, and test datasets
        
        Returns:
            train_dataset, val_dataset, test_dataset
        """
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
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, 
                           datasets: Tuple[WeatherBench2Dataset, WeatherBench2Dataset, WeatherBench2Dataset],
                           batch_size: int = 32,
                           num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch data loaders
        
        Args:
            datasets: (train_dataset, val_dataset, test_dataset)
            batch_size: Batch size for training
            num_workers: Number of worker processes for data loading
            
        Returns:
            train_loader, val_loader, test_loader
        """
        train_dataset, val_dataset, test_dataset = datasets
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader