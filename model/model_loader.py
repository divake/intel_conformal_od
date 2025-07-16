"""
Model loading utilities for detectron2 models.
This module provides functions to build and load detectron2 models.
"""

import os
import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.model_zoo import model_zoo


def d2_build_model(cfg, logger=None):
    """
    Build a detectron2 model from configuration.
    
    Args:
        cfg: Configuration object
        logger: Logger instance (optional)
        
    Returns:
        cfg_model: Model configuration
        model: Built model instance
    """
    # Create detectron2 config
    cfg_model = get_cfg()
    
    # Check if we have a model config file path
    if hasattr(cfg, 'MODEL') and hasattr(cfg.MODEL, 'CONFIG_PATH'):
        config_file = cfg.MODEL.CONFIG_PATH
        if logger:
            logger.info(f"Loading model config from: {config_file}")
        cfg_model.merge_from_file(config_file)
    else:
        # Use default Faster R-CNN config
        if logger:
            logger.info("Using default Faster R-CNN X101-FPN config")
        cfg_model.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    
    # Set device
    if hasattr(cfg, 'MODEL') and hasattr(cfg.MODEL, 'DEVICE'):
        cfg_model.MODEL.DEVICE = cfg.MODEL.DEVICE
    else:
        cfg_model.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set other model parameters from config
    if hasattr(cfg, 'MODEL'):
        if hasattr(cfg.MODEL, 'ROI_HEADS_SCORE_THRESH_TEST'):
            cfg_model.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.MODEL.ROI_HEADS_SCORE_THRESH_TEST
        if hasattr(cfg.MODEL, 'ROI_HEADS_NMS_THRESH_TEST'):
            cfg_model.MODEL.ROI_HEADS.NMS_THRESH_TEST = cfg.MODEL.ROI_HEADS_NMS_THRESH_TEST
        if hasattr(cfg.MODEL, 'DETECTIONS_PER_IMAGE'):
            cfg_model.TEST.DETECTIONS_PER_IMAGE = cfg.MODEL.DETECTIONS_PER_IMAGE
    
    # Build model
    model = build_model(cfg_model)
    
    if logger:
        logger.info(f"Model built successfully on device: {cfg_model.MODEL.DEVICE}")
    
    return cfg_model, model


def d2_load_model(cfg_model, model, logger=None):
    """
    Load model weights from checkpoint.
    
    Args:
        cfg_model: Model configuration
        model: Model instance to load weights into
        logger: Logger instance (optional)
    """
    # Create checkpointer
    checkpointer = DetectionCheckpointer(model)
    
    # Determine checkpoint path
    if hasattr(cfg_model.MODEL, 'WEIGHTS') and cfg_model.MODEL.WEIGHTS:
        checkpoint_path = cfg_model.MODEL.WEIGHTS
    else:
        # Try to find checkpoint in standard locations
        possible_paths = [
            "checkpoints/faster_rcnn_X_101_32x8d_FPN_3x.pth",
            "checkpoints/faster_rcnn_R_50_FPN_3x.pkl",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                         "checkpoints/faster_rcnn_X_101_32x8d_FPN_3x.pth"),
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if not checkpoint_path:
            # Try model zoo
            checkpoint_path = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    
    if logger:
        logger.info(f"Loading model weights from: {checkpoint_path}")
    
    # Load checkpoint
    checkpointer.load(checkpoint_path)
    
    # Set model to eval mode
    model.eval()
    
    if logger:
        logger.info("Model weights loaded successfully")