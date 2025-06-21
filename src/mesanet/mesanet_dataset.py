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
    """PyTorch Dataset for WeatherBench2 ERA5 data - FIXED VERSION with size validation"""
    
    def __init__(self, 
                 zarr_path: str,
                 variables: List[str],
                 time_range: slice = slice("1959", "2023"),
                 split: str = "train",
                 sequence_length: int = 12,
                 forecast_horizon: int = 4,
                 normalize: bool = True):
        """Initialize WeatherBench2 Dataset with comprehensive error handling"""
        
        self.zarr_path = zarr_path
        self.variables = variables
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.normalize = normalize
        self.split = split
        
        # 🔧 FIX: Store target dimensions
        self.target_height = None
        self.target_width = None
        
        logger.info(f"Initializing WeatherBench2Dataset for {split} split")
        
        # Load dataset with lazy evaluation
        self.ds = self._load_dataset(time_range)
        
        # 🔧 FIX: Get and store actual spatial dimensions
        self._determine_spatial_dimensions()
        
        # Validate variables exist
        self.available_variables = self._validate_variables()
        
        # Create geographic features with CORRECT dimensions
        self.geo_features = self._create_geo_features()
        
        # Create valid time indices for this split
        self.time_indices = self._create_time_indices(split)
        
        # Compute normalization statistics if needed
        if self.normalize:
            self.norm_stats = self._compute_normalization_stats()
        else:
            self.norm_stats = {}
        
        logger.info(f"Dataset initialized: {len(self)} samples")
        logger.info(f"Target spatial dimensions: {self.target_height} × {self.target_width}")
        
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
        # Apply Europe bounds with consistent selection
        europe_mask = (ds.longitude >= 335) | (ds.longitude <= 50)
        ds = ds.where(europe_mask, drop=True).sel(latitude=slice(75, 30))

        # Ensure consistent longitude selection
        target_lons = ds.longitude.values
        
        logger.info(f"Dataset loaded: {dict(ds.dims)}")
        logger.info(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
        logger.info(f"Available variables: {len(list(ds.data_vars.keys()))}")
        
        return ds
    
    def _determine_spatial_dimensions(self):
        """🔧 FIX: Determine the actual spatial dimensions from the dataset"""
        try:
            # Get dimensions from the dataset
            self.target_height = len(self.ds.latitude)
            self.target_width = len(self.ds.longitude)
            
            logger.info(f"✅ Determined spatial dimensions: {self.target_height} × {self.target_width}")
            
            # Double-check with actual data variable
            for var_name in list(self.ds.data_vars.keys())[:3]:  # Check first few variables
                try:
                    var = self.ds[var_name]
                    if 'latitude' in var.dims and 'longitude' in var.dims:
                        var_height = var.sizes['latitude']
                        var_width = var.sizes['longitude']
                        
                        if var_height != self.target_height or var_width != self.target_width:
                            logger.warning(f"Variable {var_name} has different dimensions: {var_height}×{var_width}")
                        else:
                            logger.debug(f"✅ Variable {var_name} matches target dimensions")
                        break
                except Exception as e:
                    logger.debug(f"Could not check dimensions for {var_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to determine spatial dimensions: {e}")
            # Fallback to common WeatherBench2 Europe dimensions
            self.target_height = 181
            self.target_width = 301
            logger.warning(f"Using fallback dimensions: {self.target_height} × {self.target_width}")
    
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
        """🔧 FIXED: Create geographic features with CORRECT dimensions"""
        logger.info("Creating geographic features...")
        
        # 🔧 FIX: Use target dimensions, not dataset coordinate dimensions
        height, width = self.target_height, self.target_width
        
        # Get coordinate values
        lats = self.ds.latitude.values
        lons = self.ds.longitude.values
        
        # Ensure we have the right number of coordinates
        if len(lats) != height:
            logger.warning(f"Latitude mismatch: expected {height}, got {len(lats)}")
            # Interpolate to target size
            lats = np.linspace(lats[0], lats[-1], height)
            
        if len(lons) != width:
            logger.warning(f"Longitude mismatch: expected {width}, got {len(lons)}")
            # Interpolate to target size  
            lons = np.linspace(lons[0], lons[-1], width)
        
        # Create coordinate grids with CORRECT dimensions
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        
        # Initialize geographic features: [lat, lon, elevation, land_sea_mask]
        geo_features = np.zeros((4, height, width), dtype=np.float32)
        
        # Latitude and longitude (normalized to [-1, 1])
        geo_features[0] = (lat_grid - lat_grid.mean()) / (lat_grid.std() + 1e-8)
        geo_features[1] = (lon_grid - lon_grid.mean()) / (lon_grid.std() + 1e-8)
        
        # 🔧 FIX: Safely handle elevation and land-sea mask with size validation
        if 'geopotential_at_surface' in self.ds.data_vars:
            try:
                elevation_data = self.ds['geopotential_at_surface']
                
                # Get first time step if it's time-dependent
                if 'time' in elevation_data.dims:
                    elevation_data = elevation_data.isel(time=0)
                
                elevation = elevation_data.values / 9.81  # Convert to meters
                
                # 🔧 FIX: Validate and resize if needed
                if elevation.shape != (height, width):
                    logger.warning(f"Elevation shape mismatch: {elevation.shape} vs ({height}, {width})")
                    elevation = self._resize_field(elevation, height, width)
                
                geo_features[2] = (elevation - elevation.mean()) / (elevation.std() + 1e-8)
                logger.info("✅ Loaded elevation from geopotential_at_surface")
                
            except Exception as e:
                logger.warning(f"Failed to load elevation: {e}")
        else:
            logger.warning("geopotential_at_surface not found - using zeros for elevation")
        
        if 'land_sea_mask' in self.ds.data_vars:
            try:
                land_sea_data = self.ds['land_sea_mask']
                
                # Get first time step if it's time-dependent
                if 'time' in land_sea_data.dims:
                    land_sea_data = land_sea_data.isel(time=0)
                
                land_sea = land_sea_data.values
                
                # 🔧 FIX: Validate and resize if needed
                if land_sea.shape != (height, width):
                    logger.warning(f"Land-sea mask shape mismatch: {land_sea.shape} vs ({height}, {width})")
                    land_sea = self._resize_field(land_sea, height, width)
                
                geo_features[3] = land_sea
                logger.info("✅ Loaded land_sea_mask")
                
            except Exception as e:
                logger.warning(f"Failed to load land_sea_mask: {e}")
        else:
            logger.warning("land_sea_mask not found - using zeros")
        
        logger.info(f"✅ Geographic features created: {geo_features.shape}")
        return torch.tensor(geo_features, dtype=torch.float32)
    
    def _resize_field(self, field: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """🔧 FIX: Resize a 2D field to target dimensions using interpolation"""
        try:
            import torch.nn.functional as F
            
            # Convert to tensor and add batch/channel dimensions
            field_tensor = torch.tensor(field, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Resize using bilinear interpolation
            resized_tensor = F.interpolate(
                field_tensor, 
                size=(target_height, target_width), 
                mode='bilinear', 
                align_corners=True 
            )
            
            # Remove batch/channel dimensions and convert back to numpy
            resized_field = resized_tensor.squeeze(0).squeeze(0).numpy()
            
            logger.info(f"Resized field from {field.shape} to {resized_field.shape}")
            return resized_field
            
        except Exception as e:
            logger.error(f"Failed to resize field: {e}")
            # Fallback: create zeros with correct shape
            return np.zeros((target_height, target_width), dtype=np.float32)
    
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
        """Get a single training sample with size validation"""
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
            
            # 🔧 FIX: Validate tensor dimensions
            input_tensor, target_tensor = self._validate_tensor_dimensions(input_tensor, target_tensor)
            
            return input_tensor, target_tensor, self.geo_features
            
        except Exception as e:
            logger.error(f"Error getting item {idx}: {e}")
            # Return dummy data with CORRECT dimensions
            dummy_input = torch.zeros(self.sequence_length, len(self.available_variables), 
                                    self.target_height, self.target_width)
            dummy_target = torch.zeros(self.forecast_horizon, 
                                     self.target_height, self.target_width)
            return dummy_input, dummy_target, self.geo_features
    
    def _validate_tensor_dimensions(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """🔧 FIX: Ensure tensors have correct spatial dimensions"""
        
        # Check input tensor dimensions
        if input_tensor.shape[-2:] != (self.target_height, self.target_width):
            logger.warning(f"Input tensor size mismatch: {input_tensor.shape[-2:]} vs ({self.target_height}, {self.target_width})")
            # Resize using interpolation
            input_tensor = self._resize_tensor(input_tensor, self.target_height, self.target_width)
        
        # Check target tensor dimensions
        if target_tensor.shape[-2:] != (self.target_height, self.target_width):
            logger.warning(f"Target tensor size mismatch: {target_tensor.shape[-2:]} vs ({self.target_height}, {self.target_width})")
            # Resize using interpolation
            target_tensor = self._resize_tensor(target_tensor, self.target_height, self.target_width)
        
        return input_tensor, target_tensor
    
    def _resize_tensor(self, tensor: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """🔧 FIX: Resize tensor to target spatial dimensions"""
        try:
            import torch.nn.functional as F
            
            original_shape = tensor.shape
            
            # Handle different tensor dimensions
            if tensor.dim() == 3:  # (time, lat, lon) 
                resized = F.interpolate(
                    tensor.unsqueeze(1),  # Add channel dim: (time, 1, lat, lon)
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=True
                ).squeeze(1)  # Remove channel dim: (time, lat, lon)
                
            elif tensor.dim() == 4:  # (time, vars, lat, lon)
                resized = F.interpolate(
                    tensor,
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=True
                )
            else:
                logger.error(f"Unexpected tensor dimensions: {tensor.shape}")
                return tensor
            
            logger.info(f"Resized tensor from {original_shape} to {resized.shape}")
            return resized
            
        except Exception as e:
            logger.error(f"Failed to resize tensor: {e}")
            return tensor
    
    def _xarray_to_tensor(self, 
                         data, 
                         normalize: bool = True, 
                         single_var: bool = False) -> torch.Tensor:
        """Convert xarray data to PyTorch tensor with size validation"""
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
                        
                        # 🔧 FIX: Validate spatial dimensions for each variable
                        if var_data.shape[-2:] != (self.target_height, self.target_width):
                            logger.warning(f"Variable {var} shape mismatch: {var_data.shape[-2:]} vs ({self.target_height}, {self.target_width})")
                            # Resize this variable's data
                            for t in range(var_data.shape[0]):
                                var_data[t] = self._resize_field(var_data[t], self.target_height, self.target_width)
                        
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
            'target_spatial_dims': (self.target_height, self.target_width),
            'geographic_features_shape': self.geo_features.shape,
            'num_samples': len(self)
        }
        return info