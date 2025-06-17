import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import zoom
from pathlib import Path

class PaperFormatEvaluator:
    
    def __init__(self, dataset_path="sr4zct_exp_dataset"):
        self.dataset_path = Path(dataset_path)
        self.load_data()
    
    def load_data(self):
        print("Loading data for evaluation...")
        
        self.gt_volume = np.load(self.dataset_path / 'recon_low.npy')
        
        self.test_volume = np.load(self.dataset_path / 'recon_low_test.npy')
        
        print(f"GT volume shape: {self.gt_volume.shape}")
        print(f"Test volume shape: {self.test_volume.shape}")
        print("Data loaded successfully.\n")
    
    def resize_to_match_2d(self, img, target_shape):

        if img.shape == target_shape:
            return img
        zoom_factors = tuple(target_shape[i] / img.shape[i] for i in range(2))
        return zoom(img, zoom_factors, order=1)
    
    def calculate_metrics_for_view(self, enhanced_volume, view_type, gt_volume):
        
        psnr_values = []
        ssim_values = []
        
        if view_type == 'coronal':
            
            max_slices = min(gt_volume.shape[1], enhanced_volume.shape[1])
            slice_indices = range(max_slices // 4, 3 * max_slices // 4, 3)  
            
            for y in slice_indices:
                try:
                    gt_slice = gt_volume[:, y, :]
                    enhanced_slice = enhanced_volume[:, y, :]
                    
                    
                    enhanced_slice = self.resize_to_match_2d(enhanced_slice, gt_slice.shape)
                    
                    data_range = gt_slice.max() - gt_slice.min()
                    if data_range > 0:
                        p = psnr(gt_slice, enhanced_slice, data_range=data_range)
                        s = ssim(gt_slice, enhanced_slice, data_range=data_range)
                        
                        if not (np.isnan(p) or np.isnan(s) or np.isinf(p) or np.isinf(s)):
                            psnr_values.append(p)
                            ssim_values.append(s)
                except Exception:
                    continue
                    
        elif view_type == 'sagittal':

            max_slices = min(gt_volume.shape[2], enhanced_volume.shape[2])
            slice_indices = range(max_slices // 4, 3 * max_slices // 4, 3)  
            
            for x in slice_indices:
                try:
                    gt_slice = gt_volume[:, :, x]
                    enhanced_slice = enhanced_volume[:, :, x]
                    
                    enhanced_slice = self.resize_to_match_2d(enhanced_slice, gt_slice.shape)
                    
                    data_range = gt_slice.max() - gt_slice.min()
                    if data_range > 0:
                        p = psnr(gt_slice, enhanced_slice, data_range=data_range)
                        s = ssim(gt_slice, enhanced_slice, data_range=data_range)
                        
                        if not (np.isnan(p) or np.isnan(s) or np.isinf(p) or np.isinf(s)):
                            psnr_values.append(p)
                            ssim_values.append(s)
                except Exception:
                    continue
        
        if len(psnr_values) == 0:
            return {'psnr': 0.0, 'ssim': 0.0}
        
        return {
            'psnr': np.mean(psnr_values),
            'ssim': np.mean(ssim_values)
        }
    
    def evaluate_original_baseline(self):
        print("Evaluating Original (baseline)...")
        
        cor_results = self.calculate_metrics_for_view(self.test_volume, 'coronal', self.gt_volume)
          
        sag_results = self.calculate_metrics_for_view(self.test_volume, 'sagittal', self.gt_volume)
        
        original_results = {
            'method': 'Original',
            'cor_psnr': cor_results['psnr'],
            'cor_ssim': cor_results['ssim'],
            'sag_psnr': sag_results['psnr'],
            'sag_ssim': sag_results['ssim']
        }
        
        print(f"Original - Coronal: PSNR={cor_results['psnr']:.2f}dB, SSIM={cor_results['ssim']:.3f}")
        print(f"Original - Sagittal: PSNR={sag_results['psnr']:.2f}dB, SSIM={sag_results['ssim']:.3f}")
        
        return original_results
    
    def evaluate_enhanced_method(self, method_name, output_dir):
        print(f"Evaluating {method_name}...")
        
        cor_path = Path(output_dir) / 'output_cor.npy'
        sag_path = Path(output_dir) / 'output_sag.npy'
        
        if not (cor_path.exists() and sag_path.exists()):
            print(f"{method_name}: Output files not found")
            return None
        
        enhanced_cor = np.load(cor_path)
        enhanced_sag = np.load(sag_path)
        
        cor_results = self.calculate_metrics_for_view(enhanced_cor, 'coronal', self.gt_volume)
        
        sag_results = self.calculate_metrics_for_view(enhanced_sag, 'sagittal', self.gt_volume)
        
        method_results = {
            'method': method_name,
            'cor_psnr': cor_results['psnr'],
            'cor_ssim': cor_results['ssim'],
            'sag_psnr': sag_results['psnr'],
            'sag_ssim': sag_results['ssim']
        }
        
        print(f"{method_name} - Coronal: PSNR={cor_results['psnr']:.2f}dB, SSIM={cor_results['ssim']:.3f}")
        print(f"{method_name} - Sagittal: PSNR={sag_results['psnr']:.2f}dB, SSIM={sag_results['ssim']:.3f}")
        
        return method_results
    
    def run_complete_evaluation(self, method_configs):
        print("="*20)
        print("SR4ZCT EVALUATION")
        print("="*20)
        
        all_results = []
        
        original_results = self.evaluate_original_baseline()
        all_results.append(original_results)
        
        print()  
        
        
        for config in method_configs:
            method_results = self.evaluate_enhanced_method(
                config['method_name'], 
                config['output_dir']
            )
            if method_results:
                all_results.append(method_results)
        
        self.generate_paper_table(all_results)
        
        return all_results
    
    def generate_paper_table(self, all_results):

        if not all_results:
            print("No results available!")
            return
        
        print("\n" + "="*80)
        print("RESULTS TABLE - Paper Table 1 Format")
        print("="*80)
        
        table_data = []
        original = all_results[0]  
        
        for result in all_results:
            method = result['method']
            
            if method == 'Original':
                table_data.append({
                    'Method': method,
                    'Coronal PSNR': f"{result['cor_psnr']:.2f}",
                    'Coronal SSIM': f"{result['cor_ssim']:.3f}",
                    'Sagittal PSNR': f"{result['sag_psnr']:.2f}",
                    'Sagittal SSIM': f"{result['sag_ssim']:.3f}",
                    'PSNR Improvement': "---",
                    'SSIM Improvement': "---"
                })
            else:

                cor_psnr_gain = result['cor_psnr'] - original['cor_psnr']
                sag_psnr_gain = result['sag_psnr'] - original['sag_psnr']
                cor_ssim_gain = result['cor_ssim'] - original['cor_ssim']
                sag_ssim_gain = result['sag_ssim'] - original['sag_ssim']
                avg_psnr_gain = (cor_psnr_gain + sag_psnr_gain) / 2
                avg_ssim_gain = (cor_ssim_gain + sag_ssim_gain) / 2
                
                table_data.append({
                    'Method': method,
                    'Coronal PSNR': f"{result['cor_psnr']:.2f}",
                    'Coronal SSIM': f"{result['cor_ssim']:.3f}",
                    'Sagittal PSNR': f"{result['sag_psnr']:.2f}",
                    'Sagittal SSIM': f"{result['sag_ssim']:.3f}",
                    'PSNR Improvement': f"{avg_psnr_gain:+.2f}dB",
                    'SSIM Improvement': f"{avg_ssim_gain:+.3f}"
                })
        
        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))
        
        df.to_csv('paper_format_results.csv', index=False)
        print(f"\nResults saved to: paper_format_results.csv")
        
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        
        enhanced_methods = all_results[1:]  
        if enhanced_methods:
            best_cor_psnr = max(enhanced_methods, key=lambda x: x['cor_psnr'])
            best_sag_psnr = max(enhanced_methods, key=lambda x: x['sag_psnr'])
            best_cor_ssim = max(enhanced_methods, key=lambda x: x['cor_ssim'])
            best_sag_ssim = max(enhanced_methods, key=lambda x: x['sag_ssim'])
            
            print(f"Best Coronal PSNR: {best_cor_psnr['method']} ({best_cor_psnr['cor_psnr']:.2f}dB)")
            print(f"Best Sagittal PSNR: {best_sag_psnr['method']} ({best_sag_psnr['sag_psnr']:.2f}dB)")
            print(f"Best Coronal SSIM: {best_cor_ssim['method']} ({best_cor_ssim['cor_ssim']:.3f})")
            print(f"Best Sagittal SSIM: {best_sag_ssim['method']} ({best_sag_ssim['sag_ssim']:.3f})")
            
            avg_psnrs = [(r['cor_psnr'] + r['sag_psnr'])/2 for r in enhanced_methods]
            best_overall_idx = np.argmax(avg_psnrs)
            best_overall = enhanced_methods[best_overall_idx]
            
            print(f"\n Overall Best Method: {best_overall['method']}")
            print(f"   Average PSNR: {avg_psnrs[best_overall_idx]:.2f}dB")
            print(f"   Average SSIM: {(best_overall['cor_ssim'] + best_overall['sag_ssim'])/2:.3f}")


def main():

    print("Paper Format SR4ZCT Evaluation")
    print("Generating Table 1 format results...\n")
    
    evaluator = PaperFormatEvaluator("sr4zct_exp_dataset")
    
    method_configs = [
        {'method_name': 'MSDNet', 'output_dir': 'msdnet_training_results'},
        {'method_name': 'UNet', 'output_dir': 'unet_training_results'},
        {'method_name': 'ResNet', 'output_dir': 'resnet_training_results'}
    ]
    
    results = evaluator.run_complete_evaluation(method_configs)
    
    print("\n" + "="*20)
    print("EVALUATION COMPLETED.")
    print("Results in paper_format_results.csv")
    print("="*20)


if __name__ == "__main__":
    main()