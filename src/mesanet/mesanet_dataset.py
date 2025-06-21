from typing import List, Tuple, Dict
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherBench2Dataset(Dataset):
    """PyTorch Dataset for WeatherBench2 ERA5 data - FIXED VERSION"""
    
    def __init__(self, 
                 zarr_path: str,
                 variables: List[str],
                 time_range: slice = slice("1959", "2023"),
                 split: str = "train",
                 sequence_length: int = 12,
                 forecast_horizon: int = 4,
                 normalize: bool = True):
        """
        Initialize WeatherBench2 Dataset
        
        Args:
            zarr_path: Path to WeatherBench2 zarr dataset
            variables: List of variable names to load
            time_range: Time range to load (e.g., slice("2020", "2023"))
            split: Data split ('train', 'val', 'test')
            sequence_length: Input sequence length (12 = 3 days of 6h data)
            forecast_horizon: Forecast horizon (4 = 1 day ahead)
            normalize: Whether to normalize the data
        """
        self.zarr_path = zarr_path
        self.variables = variables
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.normalize = normalize
        self.split = split
        
        logger.info(f"Initializing WeatherBench2Dataset for {split} split")
        
        # Load dataset with lazy evaluation
        self.ds = self._load_dataset(time_range)
        
        # Validate variables exist
        self.available_variables = self._validate_variables()
        
        # Create geographic features once
        self.geo_features = self._create_geo_features()
        
        # Create valid time indices for this split
        self.time_indices = self._create_time_indices(split)
        
        # Compute normalization statistics if needed
        if self.normalize:
            self.norm_stats = self._compute_normalization_stats()
        else:
            self.norm_stats = {}
        
        logger.info(f"Dataset initialized: {len(self)} samples")
        
    def _load_dataset(self, time_range: slice) -> xr.Dataset:
        """Load and preprocess WeatherBench2 dataset"""
        logger.info("Loading WeatherBench2 dataset...")
        
        ds = xr.open_zarr(
            self.zarr_path,
            consolidated=True,
            storage_options={"token": "anon", "asynchronous": False}
        )
        
        # Select time range first
        ds = ds.sel(time=time_range)
        
        # Apply Europe bounds (handle longitude wrapping)
        ds = ds.where(
            (ds.longitude >= 335) | (ds.longitude <= 50),
            drop=True
        ).sel(latitude=slice(75, 30))
        
        logger.info(f"Dataset loaded: {dict(ds.dims)}")
        logger.info(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
        logger.info(f"Available variables: {len(list(ds.data_vars.keys()))}")
        
        return ds
    
    def _validate_variables(self) -> List[str]:
        """Validate which variables actually exist in the dataset"""
        available_vars = list(self.ds.data_vars.keys())
        validated_vars = []
        missing_vars = []
        
        for var in self.variables:
            if var in available_vars:
                validated_vars.append(var)
                logger.info(f"✅ Variable found: {var}")
            else:
                missing_vars.append(var)
                logger.warning(f"❌ Variable missing: {var}")
        
        if missing_vars:
            logger.warning(f"Missing variables: {missing_vars}")
            logger.info(f"Will proceed with available variables: {validated_vars}")
        
        if not validated_vars:
            raise ValueError("No valid variables found in dataset!")
        
        # Update variables to only include available ones
        self.variables = validated_vars
        return validated_vars
    
    def _create_time_indices(self, split: str) -> np.ndarray:
        """Create valid time indices for sequence creation based on split"""
        total_time_steps = len(self.ds.time)
        
        # Create valid indices (need enough history and future for sequences)
        valid_indices = np.arange(
            self.sequence_length,
            total_time_steps - self.forecast_horizon
        )
        
        # Split data temporally
        if split == "train":
            split_idx = int(0.7 * len(valid_indices))
            indices = valid_indices[:split_idx]
        elif split == "val":
            start_idx = int(0.7 * len(valid_indices))
            end_idx = int(0.85 * len(valid_indices))
            indices = valid_indices[start_idx:end_idx]
        elif split == "test":
            start_idx = int(0.85 * len(valid_indices))
            indices = valid_indices[start_idx:]
        else:
            indices = valid_indices
        
        logger.info(f"{split.capitalize()} split: {len(indices)} samples")
        return indices
    
    def _create_geo_features(self) -> torch.Tensor:
        """Create geographic features tensor"""
        logger.info("Creating geographic features...")
        
        # Get spatial dimensions
        lats = self.ds.latitude.values
        lons = self.ds.longitude.values
        height, width = len(lats), len(lons)
        
        # Create coordinate grids
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        
        # Initialize geographic features: [lat, lon, elevation, land_sea_mask]
        geo_features = np.zeros((4, height, width), dtype=np.float32)
        
        # Latitude and longitude (normalized to [-1, 1])
        geo_features[0] = (lat_grid - lat_grid.mean()) / (lat_grid.std() + 1e-8)
        geo_features[1] = (lon_grid - lon_grid.mean()) / (lon_grid.std() + 1e-8)
        
        # Try to get elevation and land-sea mask if available
        if 'geopotential_at_surface' in self.ds.data_vars:
            try:
                elevation = self.ds['geopotential_at_surface'].values / 9.81  # Convert to meters
                geo_features[2] = (elevation - elevation.mean()) / (elevation.std() + 1e-8)
                logger.info("✅ Loaded elevation from geopotential_at_surface")
            except Exception as e:
                logger.warning(f"Failed to load elevation: {e}")
        else:
            logger.warning("geopotential_at_surface not found - using zeros for elevation")
        
        if 'land_sea_mask' in self.ds.data_vars:
            try:
                land_sea = self.ds['land_sea_mask'].values
                geo_features[3] = land_sea
                logger.info("✅ Loaded land_sea_mask")
            except Exception as e:
                logger.warning(f"Failed to load land_sea_mask: {e}")
        else:
            logger.warning("land_sea_mask not found - using zeros")
        
        return torch.tensor(geo_features, dtype=torch.float32)
    
    def _compute_normalization_stats(self) -> Dict[str, Tuple[float, float]]:
        """Compute mean and std for each variable using a subset of data"""
        logger.info("Computing normalization statistics...")
        
        norm_stats = {}
        
        # Use every 50th time step to compute stats (for efficiency)
        sample_indices = np.arange(0, len(self.ds.time), 50)
        
        for var in self.available_variables:
            try:
                # Load a subset of data
                sample_data = self.ds[var].isel(time=sample_indices).load()
                
                # Compute statistics
                mean_val = float(sample_data.mean().values)
                std_val = float(sample_data.std().values)
                
                # Avoid division by zero
                if std_val < 1e-8:
                    std_val = 1.0
                    logger.warning(f"Very small std for {var}, using 1.0")
                
                norm_stats[var] = (mean_val, std_val)
                logger.info(f"Stats for {var}: mean={mean_val:.4f}, std={std_val:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to compute stats for {var}: {e}")
                # Use default values
                norm_stats[var] = (0.0, 1.0)
        
        return norm_stats
    
    def __len__(self) -> int:
        """Return number of valid sequences"""
        return len(self.time_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training sample
        
        Args:
            idx: Index of the sample
            
        Returns:
            input_seq: (seq_len, vars, lat, lon) - Input meteorological sequence
            target_seq: (forecast_horizon, lat, lon) - Target precipitation
            geo_features: (4, lat, lon) - Geographic features
        """
        try:
            print(f"[DEBUG] Loading sample index: {idx}")
            # Get the actual time index
            time_idx = self.time_indices[idx]
            
            # Create input sequence
            input_slice = self.ds.isel(
                time=slice(time_idx - self.sequence_length, time_idx)
            ).load()
            
            # Create target sequence (precipitation only)
            if 'total_precipitation_6hr' in self.available_variables:
                target_slice = self.ds['total_precipitation_6hr'].isel(
                    time=slice(time_idx, time_idx + self.forecast_horizon)
                ).load()
            else:
                # Fallback to first available variable
                target_var = self.available_variables[0]
                target_slice = self.ds[target_var].isel(
                    time=slice(time_idx, time_idx + self.forecast_horizon)
                ).load()
                logger.warning(f"Using {target_var} as target instead of precipitation")
            
            # Convert to tensors
            input_tensor = self._xarray_to_tensor(input_slice, normalize=True)
            target_tensor = self._xarray_to_tensor(target_slice, normalize=False, single_var=True)
            
            return input_tensor, target_tensor, self.geo_features
            
        except Exception as e:
            logger.error(f"Error getting item {idx}: {e}")
            # Return dummy data to avoid crashes
            dummy_input = torch.zeros(self.sequence_length, len(self.available_variables), 
                                    self.geo_features.shape[1], self.geo_features.shape[2])
            dummy_target = torch.zeros(self.forecast_horizon, 
                                     self.geo_features.shape[1], self.geo_features.shape[2])
            return dummy_input, dummy_target, self.geo_features
    
    def _xarray_to_tensor(self, 
                         data, 
                         normalize: bool = True, 
                         single_var: bool = False) -> torch.Tensor:
        """
        Convert xarray data to PyTorch tensor - FIXED VERSION
        
        Args:
            data: xarray Dataset or DataArray
            normalize: Whether to apply normalization
            single_var: Whether this is a single variable (target) or multi-variable (input)
            
        Returns:
            torch.Tensor: Converted tensor
        """
        try:
            if single_var:
                # Single variable (e.g., precipitation target)
                if isinstance(data, xr.DataArray):
                    tensor_data = torch.tensor(data.values, dtype=torch.float32)
                else:
                    # If it's a Dataset, get the first variable
                    var_name = list(data.data_vars.keys())[0]
                    tensor_data = torch.tensor(data[var_name].values, dtype=torch.float32)
                
                # Handle NaN values
                tensor_data = torch.nan_to_num(tensor_data, nan=0.0, posinf=0.0, neginf=0.0)
                return tensor_data
            
            else:
                # Multiple variables - stack along variable dimension
                var_arrays = []
                
                for var in self.available_variables:
                    if var in data.data_vars:
                        var_data = data[var].values
                        
                        # Handle different dimensionalities
                        if var_data.ndim == 2:  # (lat, lon) - static field
                            # Repeat for all time steps
                            var_data = np.repeat(var_data[None, ...], 
                                               data.dims['time'], axis=0)
                            logger.debug(f"Expanded static field {var} to shape {var_data.shape}")
                            
                        elif var_data.ndim == 3:  # (time, lat, lon) - surface field
                            pass  # Already correct shape
                            
                        elif var_data.ndim == 4:  # (time, level, lat, lon) - multi-level
                            # Average over pressure levels for simplicity
                            var_data = np.mean(var_data, axis=1)
                            logger.debug(f"Averaged {var} over pressure levels: {var_data.shape}")
                            
                        else:
                            logger.warning(f"Unexpected dimensionality for {var}: {var_data.ndim}")
                            continue
                        
                        # Normalize if requested
                        if normalize and self.normalize and var in self.norm_stats:
                            mean_val, std_val = self.norm_stats[var]
                            var_data = (var_data - mean_val) / std_val
                        
                        # Handle NaN values
                        var_data = np.nan_to_num(var_data, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        var_arrays.append(var_data)
                        logger.debug(f"Processed {var}: final shape {var_data.shape}")
                    
                    else:
                        logger.warning(f"Variable {var} not found in data")
                
                if not var_arrays:
                    raise ValueError("No variables could be processed!")
                
                # Stack variables: (time, vars, lat, lon)
                try:
                    stacked_array = np.stack(var_arrays, axis=1)
                    tensor_data = torch.tensor(stacked_array, dtype=torch.float32)
                    
                    logger.debug(f"Final tensor shape: {tensor_data.shape}")
                    return tensor_data
                    
                except ValueError as e:
                    logger.error(f"Failed to stack variables: {e}")
                    logger.error(f"Variable shapes: {[arr.shape for arr in var_arrays]}")
                    raise
            
        except Exception as e:
            logger.error(f"Error in _xarray_to_tensor: {e}")
            logger.error(f"Data type: {type(data)}")
            if hasattr(data, 'dims'):
                logger.error(f"Data dims: {data.dims}")
            raise

    def get_variable_info(self) -> Dict:
        """Get information about available variables and their properties"""
        info = {
            'available_variables': self.available_variables,
            'requested_variables': self.variables,
            'normalization_stats': self.norm_stats if hasattr(self, 'norm_stats') else {},
            'dataset_shape': dict(self.ds.dims),
            'geographic_features_shape': self.geo_features.shape,
            'num_samples': len(self)
        }
        return info


