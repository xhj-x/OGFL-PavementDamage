"""
Validation script for OGFL-PavementDamage
Evaluate model performance on test datasets
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO


def validate(
    data,
    weights,
    batch_size=32,
    img_size=640,
    conf_thres=0.001,
    iou_thres=0.6,
    device='0',
    save_json=False,
    save_txt=False,
    project='runs/val',
    name='exp',
    exist_ok=False
):
    """
    Run validation on test dataset
    
    Args:
        data: Path to dataset YAML file
        weights: Path to model weights
        batch_size: Batch size for validation
        img_size: Input image size
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        device: CUDA device (0, 1, etc.) or 'cpu'
        save_json: Save results to JSON
        save_txt: Save results to TXT
        project: Save results to project/name
        name: Save results to project/name
        exist_ok: Overwrite existing results
    """
    
    # Check if weights exist
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights not found: {weights}")
    
    # Check if data config exists
    if not os.path.exists(data):
        raise FileNotFoundError(f"Dataset config not found: {data}")
    
    print("=" * 60)
    print("OGFL-PavementDamage Validation")
    print("=" * 60)
    print(f"Weights: {weights}")
    print(f"Data: {data}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    model = YOLO(weights)
    
    # Run validation
    results = model.val(
        data=data,
        batch=batch_size,
        imgsz=img_size,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        save_json=save_json,
        save_txt=save_txt,
        project=project,
        name=name,
        exist_ok=exist_ok,
        plots=True,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    # Per-class results
    if hasattr(results.box, 'maps'):
        print("\nPer-class mAP50:")
        class_names = ['D00', 'D10', 'D20', 'D40']
        for i, (name, map50) in enumerate(zip(class_names, results.box.maps)):
            print(f"  {name}: {map50:.4f}")
    
    print("=" * 60)
    print(f"Results saved to: {Path(project) / name}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Validate OGFL-PavementDamage model'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset YAML file'
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to model weights (.pt file)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for validation (default: 32)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.001,
        help='Confidence threshold (default: 0.001)'
    )
    parser.add_argument(
        '--iou-thres',
        type=float,
        default=0.6,
        help='IoU threshold for NMS (default: 0.6)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='CUDA device (0, 1, etc.) or cpu (default: 0)'
    )
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save results to TXT file'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/val',
        help='Save results to project/name'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='exp',
        help='Save results to project/name'
    )
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='Overwrite existing results'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validate(
        data=args.data,
        weights=args.weights,
        batch_size=args.batch_size,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
        save_json=args.save_json,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok
    )


if __name__ == '__main__':
    main()