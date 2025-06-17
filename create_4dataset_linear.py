import numpy as np
import os
import argparse
from typing import Tuple, Optional
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VolumeProcessor:

    def __init__(self, target_dimensions: Optional[Tuple[int, int, int]] = None):
        if target_dimensions is None:
            self.target_dims = (128, 512, 512)
            logger.info("Using default ML-friendly dimensions: (128, 512, 512)")
        else:
            self.target_dims = target_dimensions
            logger.info(f"Using custom target dimensions: {target_dimensions}")

        self.original_volume = None
        self.processed_volume = None

    def load_volume(self, filepath: str) -> np.ndarray:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Volume file not found: {filepath}")

        logger.info(f"Loading volume from: {filepath}")
        self.original_volume = np.load(filepath)
        logger.info(f"Loaded volume shape: {self.original_volume.shape}")

        return self.original_volume

    def determine_target_dimensions(self, volume: np.ndarray, use_default: bool = True) -> Tuple[int, int, int]:
        original_shape = volume.shape

        if use_default:
            target_dims = self.target_dims
            logger.info(f"Using predefined target dimensions: {original_shape} -> {target_dims}")
        else:
            target_dims = volume.shape
            logger.info(f"Keeping original dimensions: {target_dims}")

        return target_dims

    def resize_volume(self, volume: np.ndarray, target_dims: Optional[Tuple[int, int, int]] = None,
                      method: str = 'trilinear') -> np.ndarray:
        if target_dims is None:
            target_dims = self.target_dims

        if volume.shape == target_dims:
            logger.info("Volume already at target dimensions")
            return volume

        
        scale_factors = tuple(target / current for target, current in
                              zip(target_dims, volume.shape))

        logger.info(f"Resizing volume: {volume.shape} -> {target_dims}")
        logger.info(f"Scale factors: {scale_factors}")

        order_map = {'nearest': 0, 'linear': 1, 'trilinear': 1, 'cubic': 3}
        order = order_map.get(method, 1)

        resized = zoom(volume, scale_factors, order=order, mode='nearest')

        if resized.shape != target_dims:
            logger.warning(f"Slight dimension mismatch: {resized.shape} vs {target_dims}")
            resized = self._adjust_dimensions(resized, target_dims)

        return resized

    def _adjust_dimensions(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        adjusted = volume.copy()

        for dim in range(3):
            current_size = adjusted.shape[dim]
            target_size = target_shape[dim]

            if current_size > target_size:
                start = (current_size - target_size) // 2
                end = start + target_size
                adjusted = np.take(adjusted, range(start, end), axis=dim)
            elif current_size < target_size:
                pad_before = (target_size - current_size) // 2
                pad_after = target_size - current_size - pad_before
                pad_width = [(0, 0)] * 3
                pad_width[dim] = (pad_before, pad_after)
                adjusted = np.pad(adjusted, pad_width, mode='edge')

        return adjusted

    def normalize_volume(self, volume: np.ndarray, method: str = 'standard') -> np.ndarray:
        log_volume = np.log1p(volume + 1e-8)

        if method == 'standard':
            
            mean_val = log_volume.mean()
            std_val = log_volume.std()
            normalized = (log_volume - mean_val) / (std_val + 1e-8)
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)

        elif method == 'minmax':
            vmin, vmax = log_volume.min(), log_volume.max()
            normalized = (log_volume - vmin) / (vmax - vmin + 1e-8)

        elif method == 'robust':
            p5, p95 = np.percentile(log_volume, [5, 95])
            normalized = np.clip(log_volume, p5, p95)
            normalized = (normalized - p5) / (p95 - p5 + 1e-8)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        logger.info(f"Normalized volume stats: mean={normalized.mean():.3f}, "
                    f"std={normalized.std():.3f}, range=[{normalized.min():.3f}, {normalized.max():.3f}]")

        return normalized


class ResolutionSimulator:

    def __init__(self, pixel_spacing: float = 1.0,
                 slice_thickness: float = 5.0,
                 slice_overlap: float = 2.5):
        self.pixel_spacing = pixel_spacing
        self.slice_thickness = slice_thickness
        self.slice_overlap = slice_overlap

        self.overlap_samples = 9

        logger.info(f"Resolution simulator initialized: pixel_spacing={pixel_spacing}mm, "
                    f"slice_thickness={slice_thickness}mm, overlap={slice_overlap}mm")

    def create_coordinate_mapping(self, image_size: int) -> Tuple[np.ndarray, np.ndarray]:
        physical_length = self.pixel_spacing + (image_size - 1) * self.pixel_spacing

        original_coords = np.linspace(self.pixel_spacing / 2,
                                      physical_length - self.pixel_spacing / 2,
                                      image_size)

        effective_spacing = self.slice_thickness - self.slice_overlap
        num_target_samples = max(1, int((physical_length - self.slice_thickness) /
                                        effective_spacing) + 1)

        target_coords = np.linspace(self.slice_thickness / 2,
                                    physical_length - self.slice_thickness / 2,
                                    num_target_samples)

        return original_coords, target_coords

    def simulate_overlap_averaging(self, interpolator, target_coords: np.ndarray) -> np.ndarray:
        if self.slice_overlap <= 0:
            return interpolator(target_coords)

        averaged_values = interpolator(target_coords).copy()

        for sample_idx in range(1, self.overlap_samples // 2 + 1):
            offset = self.slice_thickness * sample_idx / self.overlap_samples

            left_samples = interpolator(target_coords - offset)
            right_samples = interpolator(target_coords + offset)

            averaged_values += left_samples + right_samples

        averaged_values /= self.overlap_samples

        return averaged_values

    def degrade_slice_vertical(self, image_slice: np.ndarray,
                               preserve_dimensions: bool = True) -> np.ndarray:
        height, width = image_slice.shape
        original_coords, target_coords = self.create_coordinate_mapping(height)

        if preserve_dimensions:
            degraded_slice = np.zeros_like(image_slice)
        else:
            degraded_slice = np.zeros((len(target_coords), width))

        intensity_range = (image_slice.min(), image_slice.max())

        for col_idx in range(width):
            column_data = image_slice[:, col_idx]

            interpolator = interp1d(original_coords, column_data,
                                    kind='linear', bounds_error=False,
                                    fill_value='extrapolate')

            degraded_column = self.simulate_overlap_averaging(interpolator, target_coords)

            if preserve_dimensions:
                reverse_interpolator = interp1d(target_coords, degraded_column,
                                                kind='linear', bounds_error=False,
                                                fill_value='extrapolate')
                degraded_slice[:, col_idx] = reverse_interpolator(original_coords)
            else:
                degraded_slice[:, col_idx] = degraded_column

        degraded_slice = np.clip(degraded_slice, intensity_range[0], intensity_range[1])

        return degraded_slice

    def degrade_slice_horizontal(self, image_slice: np.ndarray,
                                 preserve_dimensions: bool = True) -> np.ndarray:
        height, width = image_slice.shape
        original_coords, target_coords = self.create_coordinate_mapping(width)

        if preserve_dimensions:
            degraded_slice = np.zeros_like(image_slice)
        else:
            degraded_slice = np.zeros((height, len(target_coords)))

        intensity_range = (image_slice.min(), image_slice.max())

        for row_idx in range(height):
            row_data = image_slice[row_idx, :]

            interpolator = interp1d(original_coords, row_data,
                                    kind='linear', bounds_error=False,
                                    fill_value='extrapolate')

            degraded_row = self.simulate_overlap_averaging(interpolator, target_coords)

            if preserve_dimensions:
                reverse_interpolator = interp1d(target_coords, degraded_row,
                                                kind='linear', bounds_error=False,
                                                fill_value='extrapolate')
                degraded_slice[row_idx, :] = reverse_interpolator(original_coords)
            else:
                degraded_slice[row_idx, :] = degraded_row

        degraded_slice = np.clip(degraded_slice, intensity_range[0], intensity_range[1])

        return degraded_slice


class SR4ZCTDatasetGenerator:

    def __init__(self, output_directory: str = "sr4zct_exp_dataset",
                 custom_dimensions: Optional[Tuple[int, int, int]] = None):
        self.output_dir = output_directory
        self.processor = VolumeProcessor(target_dimensions=custom_dimensions)

        self.simulator = ResolutionSimulator(
            pixel_spacing=1.0,  
            slice_thickness=5.0,  
            slice_overlap=2.5  
        )

        os.makedirs(output_directory, exist_ok=True)
        logger.info(f"Dataset will be saved to: {output_directory}")
        logger.info(f"Target dimensions: {self.processor.target_dims}")

    def configure_simulation_parameters(self, pixel_spacing: float,
                                        slice_thickness: float,
                                        slice_overlap: float):
        self.simulator = ResolutionSimulator(pixel_spacing, slice_thickness, slice_overlap)
        logger.info("Simulation parameters updated")

    def generate_training_data(self, input_volume_path: str,
                               normalization_method: str = 'standard',
                               use_default_size: bool = True) -> dict:
        
        logger.info("=== Starting SR4ZCT Dataset Generation ===")

        original_volume = self.processor.load_volume(input_volume_path)

        if use_default_size:
            target_dims = self.processor.target_dims
            logger.info(f"Using ML-friendly dimensions: {target_dims}")
            processed_volume = self.processor.resize_volume(original_volume, target_dims)
            logger.info(f"Volume resized from {original_volume.shape} to {processed_volume.shape}")
        else:
            processed_volume = original_volume
            logger.info(f"Using original volume dimensions: {processed_volume.shape}")

        final_dimensions = processed_volume.shape
        logger.info(f"Final volume dimensions for dataset generation: {final_dimensions}")

        normalized_volume = self.processor.normalize_volume(processed_volume, normalization_method)

        reference_path = os.path.join(self.output_dir, 'recon_low.npy')
        np.save(reference_path, normalized_volume)
        logger.info(f"Saved reference volume: {reference_path}")

        logger.info("Generating vertically degraded images...")
        vertical_degraded = self._process_volume_slices(normalized_volume, 'vertical')
        vertical_path = os.path.join(self.output_dir, 'recon_low_vertical.npy')
        np.save(vertical_path, vertical_degraded)
        logger.info(f"Saved vertical degraded data: {vertical_path}")

        logger.info("Generating horizontally degraded images...")
        horizontal_degraded = self._process_volume_slices(normalized_volume, 'horizontal')
        horizontal_path = os.path.join(self.output_dir, 'recon_low_horizontal.npy')
        np.save(horizontal_path, horizontal_degraded)
        logger.info(f"Saved horizontal degraded data: {horizontal_path}")

        logger.info("Generating test volume...")
        test_volume = self._generate_test_volume(normalized_volume)
        test_path = os.path.join(self.output_dir, 'recon_low_test.npy')
        np.save(test_path, test_volume)
        logger.info(f"Saved test volume: {test_path}")

        dataset_info = self._generate_dataset_summary(final_dimensions)

        return {
            'reference': reference_path,
            'vertical_degraded': vertical_path,
            'horizontal_degraded': horizontal_path,
            'test_volume': test_path,
            'dimensions': final_dimensions,
            'dataset_info': dataset_info
        }

    def _process_volume_slices(self, volume: np.ndarray, direction: str) -> np.ndarray:
        degraded_volume = np.zeros_like(volume)

        degradation_func = (self.simulator.degrade_slice_vertical if direction == 'vertical'
                            else self.simulator.degrade_slice_horizontal)

        desc = f"{direction.capitalize()} degradation"
        for slice_idx in tqdm(range(volume.shape[0]), desc=desc):
            degraded_volume[slice_idx] = degradation_func(volume[slice_idx], preserve_dimensions=True)

        return degraded_volume

    def _generate_test_volume(self, reference_volume: np.ndarray) -> np.ndarray:
        test_simulator = ResolutionSimulator(
            pixel_spacing=self.simulator.slice_thickness,
            slice_thickness=self.simulator.pixel_spacing,
            slice_overlap=0.0
        )

        sample_slice = reference_volume[:, 0]  
        upsampled_slice = test_simulator.degrade_slice_vertical(sample_slice,
                                                                preserve_dimensions=False)

        test_shape = (upsampled_slice.shape[0],
                      reference_volume.shape[1],
                      reference_volume.shape[2])
        test_volume = np.zeros(test_shape, dtype=np.float32)

        logger.info(f"Test volume shape: {test_shape}")

        for col_idx in tqdm(range(reference_volume.shape[1]), desc="Generating test data"):
            column_slice = reference_volume[:, col_idx, :]
            test_volume[:, col_idx, :] = test_simulator.degrade_slice_vertical(
                column_slice, preserve_dimensions=False)

        return test_volume

    def _generate_dataset_summary(self, final_dimensions: Tuple[int, int, int]) -> dict:
        summary_path = os.path.join(self.output_dir, 'dataset_summary.txt')

        datasets = {}
        for filename in ['recon_low.npy', 'recon_low_vertical.npy',
                         'recon_low_horizontal.npy', 'recon_low_test.npy']:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                datasets[filename] = np.load(filepath)

        dataset_info = {
            'final_dimensions': final_dimensions,
            'simulation_parameters': {
                'pixel_spacing': self.simulator.pixel_spacing,
                'slice_thickness': self.simulator.slice_thickness,
                'slice_overlap': self.simulator.slice_overlap
            },
            'file_statistics': {}
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("SR4ZCT Dataset Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write("Final Volume Dimensions:\n")
            f.write(
                f"  Depth x Height x Width: {final_dimensions[0]} x {final_dimensions[1]} x {final_dimensions[2]}\n\n")

            f.write("Dataset Configuration:\n")
            f.write(f"  Pixel spacing: {self.simulator.pixel_spacing} mm\n")
            f.write(f"  Slice thickness: {self.simulator.slice_thickness} mm\n")
            f.write(f"  Slice overlap: {self.simulator.slice_overlap} mm\n\n")

            f.write("Dataset Statistics:\n")
            for name, data in datasets.items():
                stats = {
                    'shape': data.shape,
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max())
                }
                dataset_info['file_statistics'][name] = stats

                f.write(f"\n{name}:\n")
                f.write(f"  Shape: {data.shape}\n")
                f.write(f"  Mean: {data.mean():.6f}\n")
                f.write(f"  Std: {data.std():.6f}\n")
                f.write(f"  Min: {data.min():.6f}\n")
                f.write(f"  Max: {data.max():.6f}\n")

        import json
        info_path = os.path.join(self.output_dir, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)

        logger.info(f"Dataset summary saved: {summary_path}")
        logger.info(f"Dataset info (JSON) saved: {info_path}")
        logger.info(f"IMPORTANT: Final dataset dimensions are {final_dimensions}")

        return dataset_info


def main():

    parser = argparse.ArgumentParser(description='Generate SR4ZCT training dataset')
    parser.add_argument('input_volume', help='Path to input reconstructed volume (.npy)')
    parser.add_argument('-o', '--output', default='sr4zct_dataset',
                        help='Output directory for generated dataset')
    parser.add_argument('--pixel-spacing', type=float, default=1.0,
                        help='Pixel spacing in mm')
    parser.add_argument('--slice-thickness', type=float, default=5.0,
                        help='Target slice thickness in mm')
    parser.add_argument('--slice-overlap', type=float, default=2.5,
                        help='Slice overlap in mm')
    parser.add_argument('--normalization', choices=['standard', 'minmax', 'robust'],
                        default='standard', help='Normalization method')
    parser.add_argument('--custom-dimensions', nargs=3, type=int, metavar=('D', 'H', 'W'),
                        help='Custom target dimensions (depth height width), e.g., --custom-dimensions 100 256 256')
    parser.add_argument('--keep-original-size', action='store_true',
                        help='Keep original volume size instead of resizing to ML-friendly dimensions')

    args = parser.parse_args()

    custom_dims = None
    if args.custom_dimensions:
        custom_dims = tuple(args.custom_dimensions)
        logger.info(f"Using custom dimensions: {custom_dims}")

    generator = SR4ZCTDatasetGenerator(args.output, custom_dims)
    generator.configure_simulation_parameters(
        args.pixel_spacing, args.slice_thickness, args.slice_overlap
    )

    try:
        output_paths = generator.generate_training_data(
            args.input_volume, args.normalization, not args.keep_original_size
        )

        logger.info("\n" + "=" * 20)
        logger.info("Dataset generation completed successfully!")
        logger.info("Generated files:")
        for key, path in output_paths.items():
            if key not in ['dimensions', 'dataset_info']:
                logger.info(f"  {key}: {path}")
        logger.info(f"Final dataset dimensions: {output_paths['dimensions']}")
        logger.info("=" * 20)

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise


if __name__ == "__main__":
    input_path = "output_3d/recon3d_FBP_180.npy"

    if os.path.exists(input_path):
        logger.info(f"Using default input path: {input_path}")

        generator = SR4ZCTDatasetGenerator("sr4zct_exp_dataset")

        try:
            output_paths = generator.generate_training_data(input_path)

            logger.info("\n" + "=" * 20)
            logger.info("Dataset generation completed successfully!")
            logger.info("Generated files:")
            for key, path in output_paths.items():
                if key not in ['dimensions', 'dataset_info']:
                    logger.info(f"  {key}: {path}")
            logger.info(f"Final dataset dimensions: {output_paths['dimensions']}")
            logger.info("=" * 20)

        except Exception as e:
            logger.error(f"Dataset generation failed: {e}")
            raise
    else:
        logger.warning(f"Default input file not found: {input_path}")
        logger.info("Running with command line arguments...")
        main()