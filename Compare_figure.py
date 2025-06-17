import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class SR4ZCTFigureGenerator:
    
    def __init__(self, dataset_path="sr4zct_exp_dataset"):
        self.dataset_path = Path(dataset_path)
        self.load_reference_data()
        
    def load_reference_data(self):
        print("Loading reference data...")
        
        self.gt_volume = np.load(self.dataset_path / 'recon_low.npy')
        
        self.test_volume = np.load(self.dataset_path / 'recon_low_test.npy')
        
        print(f"GT volume shape: {self.gt_volume.shape}")
        print(f"Test volume shape: {self.test_volume.shape}")
    
    def extract_slice(self, volume, view_type, slice_idx):
        if view_type == 'coronal':
            return volume[:, slice_idx, :]
        elif view_type == 'sagittal':
            return volume[:, :, slice_idx]
        else:
            raise ValueError(f"Unknown view type: {view_type}")
    
    def resize_slice_to_match(self, slice_img, target_shape):
        from scipy.ndimage import zoom
        
        if slice_img.shape == target_shape:
            return slice_img
        
        zoom_factors = tuple(target_shape[i] / slice_img.shape[i] for i in range(2))
        return zoom(slice_img, zoom_factors, order=1)
    
    def calculate_metrics(self, pred_img, gt_img):
        if pred_img.shape != gt_img.shape:
            pred_img = self.resize_slice_to_match(pred_img, gt_img.shape)
        
        data_range = gt_img.max() - gt_img.min()
        if data_range == 0:
            return 0.0, 0.0
        
        try:
            psnr_val = psnr(gt_img, pred_img, data_range=data_range)
            ssim_val = ssim(gt_img, pred_img, data_range=data_range)
            return psnr_val, ssim_val
        except:
            return 0.0, 0.0
    
    def create_comparison_figure(self, view_type='coronal', slice_idx=None, 
                                output_dir='comparison_figures',
                                method_dirs=None, crop_region=None):
        if method_dirs is None:
            method_dirs = {
                'MSDNet': 'msdnet_training_results',
                'UNet': 'unet_training_results', 
                'ResNet': 'resnet_training_results'
            }
        
        os.makedirs(output_dir, exist_ok=True)
        
        if slice_idx is None:
            if view_type == 'coronal':
                slice_idx = self.gt_volume.shape[1] // 2
            else:
                slice_idx = self.gt_volume.shape[2] // 2
        
        print(f"Generating {view_type} comparison at slice {slice_idx}")
        
        gt_slice = self.extract_slice(self.gt_volume, view_type, slice_idx)
        
        original_slice = self.extract_slice(self.test_volume, view_type, slice_idx)
        original_slice = self.resize_slice_to_match(original_slice, gt_slice.shape)
        
        if crop_region is not None:
            y_start, y_end, x_start, x_end = crop_region
            gt_slice = gt_slice[y_start:y_end, x_start:x_end]
            original_slice = original_slice[y_start:y_end, x_start:x_end]
        
        methods_data = {
            'Original': {
                'slice': original_slice,
                'psnr': 0.0,
                'ssim': 0.0
            }
        }
        
        methods_data['Original']['psnr'], methods_data['Original']['ssim'] = \
            self.calculate_metrics(original_slice, gt_slice)
        
        for method_name, method_dir in method_dirs.items():
            try:
                if view_type == 'coronal':
                    output_path = Path(method_dir) / 'output_cor.npy'
                else:
                    output_path = Path(method_dir) / 'output_sag.npy'
                
                if output_path.exists():
                    method_volume = np.load(output_path)
                    method_slice = self.extract_slice(method_volume, view_type, slice_idx)
                    method_slice = self.resize_slice_to_match(method_slice, gt_slice.shape)
                    
                    if crop_region is not None:
                        method_slice = method_slice[y_start:y_end, x_start:x_end]
                    
                    psnr_val, ssim_val = self.calculate_metrics(method_slice, gt_slice)
                    
                    methods_data[method_name] = {
                        'slice': method_slice,
                        'psnr': psnr_val,
                        'ssim': ssim_val
                    }
                    
                    print(f"{method_name}: PSNR={psnr_val:.2f}dB, SSIM={ssim_val:.3f}")
                else:
                    print(f"Warning: {output_path} not found")
                    
            except Exception as e:
                print(f"Error loading {method_name}: {e}")
        
        self._generate_individual_images(methods_data, output_dir, view_type, slice_idx)
        
        print(f"Comparison figures saved to: {output_dir}")
    
    def _generate_individual_images(self, methods_data, output_dir, view_type, slice_idx):
        
        fig_width = 3.0
        fig_height = 3.5
        dpi = 300
        
        for method_name, data in methods_data.items():
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=dpi)
            
            slice_img = data['slice']
            vmin, vmax = slice_img.min(), slice_img.max()
            
            im = ax.imshow(slice_img, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            title_text = method_name
            ax.text(0.5, 1.05, title_text, transform=ax.transAxes,
                   fontsize=6, fontweight='bold', ha='center', va='bottom',
                   color='black')
            
            if method_name != 'Original':
                metrics_text = f"PSNR: {data['psnr']:.2f}\nSSIM: {data['ssim']:.3f}"
            else:
                metrics_text = f"PSNR: {data['psnr']:.2f}\nSSIM: {data['ssim']:.3f}"
            
            ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes,
                   fontsize=6, ha='right', va='top', color='white',
                   bbox=None,
                   weight='normal')
            
            plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
            
            output_path = Path(output_dir) / f"{method_name}_{view_type}_slice{slice_idx}.png"
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', pad_inches=0.1)
            plt.close()
            
            print(f"Saved: {output_path}")
    
    def create_combined_figure(self, view_type='coronal', slice_idx=None,
                             output_dir='comparison_figures',
                             method_dirs=None, crop_region=None):
        if method_dirs is None:
            method_dirs = {
                'MSDNet': 'msdnet_training_results',
                'UNet': 'unet_training_results',
                'ResNet': 'resnet_training_results'
            }
        
        os.makedirs(output_dir, exist_ok=True)
        
        if slice_idx is None:
            if view_type == 'coronal':
                slice_idx = self.gt_volume.shape[1] // 2
            else:
                slice_idx = self.gt_volume.shape[2] // 2
        
        gt_slice = self.extract_slice(self.gt_volume, view_type, slice_idx)
        original_slice = self.extract_slice(self.test_volume, view_type, slice_idx)
        original_slice = self.resize_slice_to_match(original_slice, gt_slice.shape)
        
        if crop_region is not None:
            y_start, y_end, x_start, x_end = crop_region
            gt_slice = gt_slice[y_start:y_end, x_start:x_end]
            original_slice = original_slice[y_start:y_end, x_start:x_end]
        
        methods_order = ['Original', 'MSDNet', 'UNet', 'ResNet']
        methods_data = {}
        
        psnr_orig, ssim_orig = self.calculate_metrics(original_slice, gt_slice)
        methods_data['Original'] = {
            'slice': original_slice,
            'psnr': psnr_orig,
            'ssim': ssim_orig
        }
        
        for method_name, method_dir in method_dirs.items():
            try:
                if view_type == 'coronal':
                    output_path = Path(method_dir) / 'output_cor.npy'
                else:
                    output_path = Path(method_dir) / 'output_sag.npy'
                
                if output_path.exists():
                    method_volume = np.load(output_path)
                    method_slice = self.extract_slice(method_volume, view_type, slice_idx)
                    method_slice = self.resize_slice_to_match(method_slice, gt_slice.shape)
                    
                    if crop_region is not None:
                        method_slice = method_slice[y_start:y_end, x_start:x_end]
                    
                    psnr_val, ssim_val = self.calculate_metrics(method_slice, gt_slice)
                    
                    methods_data[method_name] = {
                        'slice': method_slice,
                        'psnr': psnr_val,
                        'ssim': ssim_val
                    }
            except Exception as e:
                print(f"Error loading {method_name}: {e}")
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=300)
        axes = axes.flatten()
        
        for i, method_name in enumerate(methods_order):
            if method_name in methods_data:
                ax = axes[i]
                data = methods_data[method_name]
                slice_img = data['slice']
                
                vmin, vmax = slice_img.min(), slice_img.max()
                ax.imshow(slice_img, cmap='gray', vmin=vmin, vmax=vmax)
                
                ax.set_title(method_name, fontsize=6, fontweight='bold', pad=10)
                
                metrics_text = f"PSNR: {data['psnr']:.2f}\nSSIM: {data['ssim']:.3f}"
                ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes,
                       fontsize=6, ha='right', va='top', color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        
        combined_path = Path(output_dir) / f"combined_{view_type}_slice{slice_idx}.png"
        plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Combined figure saved: {combined_path}")


def main():
    
    generator = SR4ZCTFigureGenerator("sr4zct_exp_dataset")
    
    method_dirs = {
        'MSDNet': 'msdnet_training_results',
        'UNet': 'unet_training_results', 
        'ResNet': 'resnet_training_results'
    }
    
    slice_idx = None
    
    crop_region = None
    
    print("Generating SR4ZCT comparison figures...")
    
    print("\n=== Generating Coronal View ===")
    generator.create_comparison_figure(
        view_type='coronal',
        slice_idx=slice_idx,
        output_dir='figures_coronal',
        method_dirs=method_dirs,
        crop_region=crop_region
    )
    
    print("\n=== Generating Sagittal View ===")
    generator.create_comparison_figure(
        view_type='sagittal', 
        slice_idx=slice_idx,
        output_dir='figures_sagittal',
        method_dirs=method_dirs,
        crop_region=crop_region
    )
    
    print("\n=== Generating Combined Figures ===")
    generator.create_combined_figure(
        view_type='coronal',
        slice_idx=slice_idx,
        output_dir='figures_combined',
        method_dirs=method_dirs,
        crop_region=crop_region
    )
    
    print("\nAll comparison figures generated successfully.")


if __name__ == "__main__":
    main()