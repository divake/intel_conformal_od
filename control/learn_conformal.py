import os
import torch
from tqdm import tqdm

from detectron2.structures.boxes import Boxes
from detectron2.data.detection_utils import annotations_to_instances

from .abstract_risk_control import RiskControl
from model import matching
from data.data_collector import DataCollector, _default_dict_fields
from calibration import random_split, pred_intervals
from calibration.conformal_scores import abs_res, one_sided_res, learned_score
from evaluation import metrics, results_table, ap_evaluation
from util.util import set_seed
from control.classifier_sets import get_label_set_generator


class LearnConformal(RiskControl):
    """
    This class is used to run the conformal bounding box procedure with
    learnable scoring function nonconformity scores (Box-Learn).
    
    The preceeding class label set step is integrated by the label_set_generator
    attribute, which is a callable object that generates label sets and computes
    the relevant metrics, and is called in the __call__ method.
    """

    def __init__(self, cfg, args, nr_class, filedir, log, logger):
        self.seed = cfg.PROJECT.SEED
        self.nr_class = nr_class
        self.filedir = filedir
        self.log = log
        self.logger = logger

        self.device = cfg.MODEL.DEVICE
        self.box_matching = cfg.MODEL.BOX_MATCHING
        self.class_matching = cfg.MODEL.CLASS_MATCHING
        self.iou_thresh = cfg.MODEL.IOU_THRESH_TEST
        self.nr_metrics = 12

        self.calib_fraction = cfg.CALIBRATION.FRACTION
        self.calib_trials = cfg.CALIBRATION.TRIALS
        
        # Check if we should use box correction from command line args
        if hasattr(args, 'learn_use_correction') and args.learn_use_correction:
            # Use box correction method specified in command line
            self.calib_box_corr = args.box_correction_method
            self.logger.info(f"Using box correction method: {self.calib_box_corr}")
        else:
            # Default behavior: no box correction for learnable method
            self.calib_box_corr = "none"
            self.logger.info("Using no box correction (default for learnable method)")
        
        self.calib_alpha = args.alpha

        self.ap_eval = cfg.MODEL.AP_EVAL
        if self.ap_eval:
            self.ap_evaluator = ap_evaluation.APEvaluator(nr_class=nr_class)

        self.label_alpha = args.label_alpha
        self.label_set_generator = get_label_set_generator(cfg, args, logger)

        # Learnable scoring model path (can be configured)
        self.learnable_model_path = getattr(args, 'learnable_model_path', None)
        
        # If not provided via args, check config file
        if self.learnable_model_path is None and hasattr(cfg, 'LEARNABLE_SCORING') and hasattr(cfg.LEARNABLE_SCORING, 'MODEL_PATH'):
            self.learnable_model_path = cfg.LEARNABLE_SCORING.MODEL_PATH
        
        # Log that we're using learnable scoring
        self.logger.info("Initialized LearnConformal with learnable scoring function")
        if self.learnable_model_path:
            self.logger.info(f"Using learnable model from: {self.learnable_model_path}")
        else:
            self.logger.info("No specific learnable model path provided, will use default path")

    def set_collector(self, nr_class: int, nr_img: int, dict_fields: list = []):
        """Set a LearnConformalDataCollector instance."""
        self.collector = LearnConformalDataCollector(
            nr_class, nr_img, dict_fields, self.logger, self.label_set_generator
        )
        # Pass the learnable model path to the collector
        self.collector.learnable_model_path = self.learnable_model_path

    def raw_prediction(self, model, img):
        """Generates model prediction for given image(s)."""
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            pred = model([img])
                
        return pred[0]["instances"]

    def collect_predictions(self, model, dataloader, verbose: bool = False):
        """Generates predictions for given dataloader and passes to data collection."""
        
        # Check if prediction files already exist and load them if they do
        img_list_file = os.path.join(self.filedir, f"{os.path.basename(self.filedir)}_img_list.json")
        ist_list_file = os.path.join(self.filedir, f"{os.path.basename(self.filedir)}_ist_list.json")
        
        if os.path.exists(img_list_file) and os.path.exists(ist_list_file):
            self.logger.info("Found existing prediction files. Loading from cache...")
            self.logger.info(f"Cache files found:")
            self.logger.info(f"  - {img_list_file} (size: {os.path.getsize(img_list_file)/1024/1024:.2f} MB)")
            self.logger.info(f"  - {ist_list_file} (size: {os.path.getsize(ist_list_file)/1024/1024:.2f} MB)")
            self.logger.info(f"Loading {img_list_file}")
            self.logger.info(f"Loading {ist_list_file}")
            
            import json
            with open(img_list_file, 'r') as f:
                img_list = json.load(f)
            with open(ist_list_file, 'r') as f:
                ist_list = json.load(f)
            
            # Set the collector's data
            self.collector.img_list = img_list
            self.collector.ist_list = ist_list
            
            self.logger.info("Successfully loaded existing predictions. Skipping prediction collection.")
            return img_list, ist_list
        
        self.logger.info("Prediction files not found. Running prediction collection...")
        self.logger.info("Collecting predictions...")
        self.logger.info(
            f"""
            Running 'collect_predictions' with {self.iou_thresh=}...
            Box matching: '{self.box_matching}', class matching: {self.class_matching}.
            """
        )
        
        model.eval()  # Set model to evaluation mode
        with torch.no_grad(), tqdm(dataloader, desc="Images") as loader:
            for i, img in enumerate(loader):
                # BoxMode.XYWH to BoxMode.XYXY and correct formatting
                gt = annotations_to_instances(
                    img[0]["annotations"], (img[0]["height"], img[0]["width"])
                )
                gt_box, gt_class = gt.gt_boxes, gt.gt_classes

                pred = model(img)
                pred_ist = pred[0]["instances"].to("cpu")
                pred_box = pred_ist.pred_boxes
                pred_class = pred_ist.pred_classes
                pred_score = pred_ist.scores
                pred_score_all = pred_ist.scores_all
                pred_logits_all = pred_ist.logits_all

                # Collect and store AP eval info
                if self.ap_eval:
                    self.ap_evaluator.collect(
                        gt_box, pred_box, gt_class, pred_class, pred_score
                    )

                # Object matching process (predictions to ground truths)
                (
                    gt_box,
                    pred_box,
                    gt_class,
                    pred_class,
                    pred_score,
                    pred_score_all,
                    pred_logits_all,
                    matches,
                ) = matching.matching(
                    gt_box,
                    pred_box,
                    gt_class,
                    pred_class,
                    pred_score,
                    pred_score_all,
                    pred_logits_all,
                    box_matching=self.box_matching,
                    class_match=self.class_matching,
                    thresh=self.iou_thresh,
                )
                
                # Collect and store risk control info
                if matches:
                    self.collector(
                        gt_box,
                        gt_class,
                        pred_box,
                        pred_score,
                        pred_score_all,
                        pred_logits_all,
                        img_id=i,
                        verbose=verbose,
                    )

                if self.log is not None:
                    self.log.define_metric("nr_matches", summary="mean")
                    self.log.log({"nr_matches": len(gt_class)})

                if verbose:
                    self.logger.info(f"\n{gt_class=}\n{pred_class=},\n{pred_score=}")

                del gt, pred, pred_ist

        # Save predictions for caching
        import json
        with open(img_list_file, 'w') as f:
            json.dump(self.collector.img_list, f)
        with open(ist_list_file, 'w') as f:
            json.dump(self.collector.ist_list, f)
        self.logger.info(f"Saved predictions to cache:")
        self.logger.info(f"  - {img_list_file}")
        self.logger.info(f"  - {ist_list_file}")
        
        if self.ap_eval:
            self.ap_evaluator.to_file(
                self.ap_evaluator.ap_info, "ap_info", self.filedir
            )
        return self.collector.img_list, self.collector.ist_list

    def __call__(self, img_list: list, ist_list: list):
        set_seed(self.seed, self.logger)
        self.logger.info("Running conformal risk control...")

        # Initialize data collection
        data = torch.zeros(
            (
                self.calib_trials,
                self.nr_class,
                self.collector.nr_scores,
                self.nr_metrics,
            )
        )
        test_indices = torch.zeros(
            (self.calib_trials, self.nr_class, len(img_list[0])), dtype=torch.bool
        )

        # Collect label set information
        self.label_set_generator.collect(
            img_list,
            ist_list,
            self.nr_class,
            self.nr_metrics,
            self.collector.nr_scores,
            self.collector.score_fields,
            self.collector.coord_fields,
        )

        # Run conformal procedure
        for t in tqdm(range(self.calib_trials)):
            for c in range(self.nr_class):
                # Skip classes with no instances
                if ist_list[c] is None:
                    continue

                if len(ist_list[c]["gt_x0"]) == 0:
                    continue

                # Convert instance data to tensors (needed for JSON loading compatibility)
                ist = {k: torch.tensor(v) for k, v in ist_list[c].items()}

                # Split data into calibration and test sets
                try:
                    calib_mask, calib_idx, test_idx = random_split.random_split(
                        imgs=torch.tensor(img_list[c]), 
                        ist_img_id=torch.tensor(ist_list[c]["img_id"]), 
                        calib_fraction=self.calib_fraction, 
                        verbose=False
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to split data for class {c}: {e}")
                    continue

                # Process each score type (learned scores)
                for s in range(self.collector.nr_scores // 4):
                    i = s * 4
                    # Load scores for this coordinate set
                    scores = torch.zeros(
                        (len(ist_list[c]["gt_x0"]), 4), dtype=torch.float32
                    )
                    gt = torch.zeros((len(ist_list[c]["gt_x0"]), 4), dtype=torch.float32)
                    pred = torch.zeros((len(ist_list[c]["gt_x0"]), 4), dtype=torch.float32)

                    # Convert instance data to tensors and compute proper scores
                    for coord_idx in range(4):
                        score_field = self.collector.score_fields[i + coord_idx]
                        coord_field = self.collector.coord_fields[coord_idx]
                        
                        # Load ground truth and predictions
                        gt[:, coord_idx] = torch.tensor(ist_list[c]["gt_" + coord_field])
                        pred[:, coord_idx] = torch.tensor(ist_list[c]["pred_" + coord_field])
                        
                        # PROPER NONCONFORMITY SCORE COMPUTATION
                        if "learn_res" in score_field:
                            # Learned models output width predictions
                            learned_widths = torch.tensor(ist_list[c][score_field])
                            
                            # Compute actual errors
                            actual_errors = torch.abs(gt[:, coord_idx] - pred[:, coord_idx])
                            
                            # Proper nonconformity score: error normalized by predicted width
                            # Higher score = larger error relative to predicted uncertainty
                            scores[:, coord_idx] = actual_errors / (learned_widths + 1e-6)
                        else:
                            # For other score types (abs_res), use directly
                            scores[:, coord_idx] = torch.tensor(ist_list[c][score_field])

                    if scores[calib_mask].shape[0] == 0:  # degenerate case
                        continue

                    # Compute quantiles
                    # Note: learnable scoring functions can work with or without box correction
                    # - Without correction: relies on learned scores optimized for box-level coverage
                    # - With correction: applies traditional multiple testing corrections like other methods
                    quant = pred_intervals.compute_quantile(
                        scores=scores[calib_mask],
                        box_correction=self.calib_box_corr,
                        alpha=self.calib_alpha,
                    )

                    # PROPER FIX: Log diagnostic information
                    if t == 0 and c == 0 and s == 0:
                        self.logger.info("=" * 60)
                        self.logger.info("USING PROPER NONCONFORMITY SCORES")
                        self.logger.info(f"Target coverage: {(1 - self.calib_alpha) * 100:.0f}%")
                        self.logger.info(f"Computed quantiles: {quant}")
                        self.logger.info("Scores are properly normalized errors (error/width)")
                        self.logger.info("=" * 60)

                    # Compute label set quantiles and store info
                    self.label_set_generator.calib_masks[t][c] = calib_mask
                    self.label_set_generator.box_quantiles[t, c, i:(i+4)] = quant
                    self.label_set_generator.label_quantiles[
                        t, c
                    ] = pred_intervals.get_quantile(
                        scores=torch.tensor(ist_list[c]["label_score"])[calib_mask],
                        alpha=self.label_alpha,
                        n=calib_mask.sum(),
                    )

                    # Compute other relevant quantities coordinate/score-wise
                    nr_calib_samp = calib_mask.sum().repeat(4)  # Only repeat for current 4 coordinates
                    
                    # PROPER INTERVAL COMPUTATION FOR LEARNED SCORES
                    if "learn_res" in self.collector.score_fields[i]:
                        # For learned scores, quantile is a ratio (error/width)
                        # We need to multiply by predicted widths to get actual intervals
                        
                        # Get the predicted widths from the model
                        predicted_widths = torch.zeros((len(ist_list[c]["gt_x0"]), 4), dtype=torch.float32)
                        for coord_idx in range(4):
                            width_field = self.collector.score_fields[i + coord_idx]
                            predicted_widths[:, coord_idx] = torch.tensor(ist_list[c][width_field])
                        
                        # Compute calibrated interval widths using quantile ratio
                        calibrated_widths = predicted_widths * quant  # Broadcasting: [N, 4] * [4]
                        
                        # Create prediction intervals properly
                        # pred has shape [N, 4], coverage expects pi with shape [N, 2, 4]
                        # where dim 1 is [lower, upper] bounds
                        lower = pred - calibrated_widths
                        upper = pred + calibrated_widths
                        pi = torch.stack([lower, upper], dim=1)  # Shape: [N, 2, 4]
                    else:
                        # For other scores (abs_res), use standard fixed intervals
                        pi = pred_intervals.fixed_pi(pred, quant)
                    cov_coord, cov_box = metrics.coverage(gt[~calib_mask], pi[~calib_mask])
                    cov_area, cov_iou = metrics.stratified_coverage(gt, pi, calib_mask, ist)
                    mpiw = metrics.mean_pi_width(pi[~calib_mask])
                    stretch = metrics.box_stretch(
                        pi[~calib_mask], ist["pred_area"][~calib_mask]
                    )

                    # Store information
                    # NOTE: order of metrics fed into tensor matters for results tables and plotting
                    metr = torch.stack(
                        (nr_calib_samp, quant, mpiw, stretch, cov_coord, cov_box), dim=1
                    )
                    metr = torch.cat((metr, cov_area, cov_iou), dim=1)

                    data[t, c, i:(i+4), :] = metr  # Store at specific slice for current coordinates
                    test_indices[t, c, test_idx.sort()[0]] = True

                    del scores, gt, pred, pi  # Don't delete calib_mask and test_idx as they're used across score iterations

        # run label set loop, which also returns box_set_data containing results with label sets
        label_sets, label_data, box_set_data = self.label_set_generator()

        return data, test_indices, label_sets, label_data, box_set_data

    def evaluate(
        self,
        data: torch.Tensor,
        label_data: torch.Tensor,
        box_set_data: torch.Tensor,
        metadata: dict,
        filedir: str,
        save_file: bool,
        load_collect_pred,
    ):
        self.logger.info("Collecting and computing results...")

        # AP eval
        if self.ap_eval:
            self.ap_evaluator.evaluate(
                metadata["thing_classes"],
                filedir,
                load_collect_pred,
                logger=self.logger,
            )

        # Risk control eval
        for s in range(self.collector.nr_scores // 4):
            i = s * 4
            self.logger.info(
                f"Evaluating for scores {self.collector.score_fields[i:(i+4)]}"
            )

            results_table.get_results_table(
                data=data[:, :, i : (i + 4), :],
                class_names=metadata["thing_classes"],
                to_file=save_file,
                filename=f"{os.path.split(filedir)[-1]}_res_table_{self.collector.score_fields[i][:-3]}",
                filedir=filedir,
                logger=self.logger,
            )

            results_table.get_label_results_table(
                data=label_data,
                class_names=metadata["thing_classes"],
                to_file=save_file,
                filename=f"{os.path.split(filedir)[-1]}_label_table",
                filedir=filedir,
                logger=self.logger,
            )

            results_table.get_box_set_results_table(
                data=box_set_data[:, :, i : (i + 4), :],
                class_names=metadata["thing_classes"],
                to_file=save_file,
                filename=f"{os.path.split(filedir)[-1]}_box_set_table_{self.collector.score_fields[i][:-3]}",
                filedir=filedir,
                logger=self.logger,
            )


class LearnConformalDataCollector(DataCollector):
    """
    Subclass of DataCollector for the LearnConformal risk control procedure.
    Uses learnable scoring function instead of absolute residual scoring.
    """
    def __init__(
        self,
        nr_class: int,
        nr_img: int,
        dict_fields: list = [],
        logger=None,
        label_set_generator=None,
    ):
        if not dict_fields:
            dict_fields = _default_dict_fields.copy()
            self.coord_fields = ["x0", "y0", "x1", "y1"]
            # Learnable conformal scores
            self.score_fields = [
                "learn_res_x0",
                "learn_res_y0", 
                "learn_res_x1",
                "learn_res_y1",
                "abs_res_x0",  # Keep abs_res as fallback
                "abs_res_y0",
                "abs_res_x1", 
                "abs_res_y1",
            ]
            self.nr_scores = len(self.score_fields)
            dict_fields += self.score_fields
        super().__init__(nr_class, nr_img, dict_fields, logger, label_set_generator)
        
        # Path for learnable model (set by parent class)
        self.learnable_model_path = None

    def __call__(
        self,
        gt_box: Boxes,
        gt_class: torch.Tensor,
        pred_box: Boxes,
        pred_score: torch.Tensor,
        pred_score_all: torch.Tensor = None,
        pred_logits_all: torch.Tensor = None,
        img_id: int = None,
        verbose: bool = False,
    ):
        # If pred_score_all is None, use pred_score
        if pred_score_all is None:
            pred_score_all = pred_score
        
        # Ensure all tensors are on the same device
        device = gt_box.tensor.device
        gt_class = gt_class.to(device)
        pred_box = Boxes(pred_box.tensor.to(device))
        pred_score = pred_score.to(device)
        if pred_score_all is not None:
            pred_score_all = pred_score_all.to(device)
        if pred_logits_all is not None:
            pred_logits_all = pred_logits_all.to(device)
            
        for c in torch.unique(gt_class).cpu().numpy():
            # img has instances of class
            self.img_list[c][img_id] = 1
            # indices for matching instances
            idx = torch.nonzero(gt_class == c, as_tuple=True)[0]
            
            # Add base infos - call parent method for standard data collection
            super()._add_instances(
                c,
                img_id,
                idx,
                gt_box,
                pred_box,
                pred_score,
                pred_score_all,
                pred_logits_all,
            )

            # Add learnable conformal scores - OPTIMIZED VERSION!
            # Process all coordinates together for better efficiency and consistency
            if len(idx) > 0:
                gt_coords_full = gt_box[idx].tensor  # [N, 4]
                pred_coords_full = pred_box[idx].tensor  # [N, 4] 
                pred_scores_full = pred_score[idx]  # [N]
                
                # Get learned nonconformity scores for all coordinates at once
                # This returns [N, 4] - one score per coordinate per instance
                all_learned_scores = learned_score(
                    gt=gt_coords_full,
                    pred=pred_coords_full,
                    pred_score=pred_scores_full,
                    model_path=self.learnable_model_path
                )
                
                # Convert to lists and add to instance lists
                coord_names = ["x0", "y0", "x1", "y1"]
                for coord_idx, coord_name in enumerate(coord_names):
                    coord_scores = all_learned_scores[:, coord_idx].tolist()
                    self.ist_list[c][f"learn_res_{coord_name}"] += coord_scores
                
            # Also keep absolute residual scores as backup/comparison
            self.ist_list[c]["abs_res_x0"] += abs_res(
                gt_box[idx].tensor[:, 0], pred_box[idx].tensor[:, 0]
            ).tolist()
            self.ist_list[c]["abs_res_y0"] += abs_res(
                gt_box[idx].tensor[:, 1], pred_box[idx].tensor[:, 1]
            ).tolist()
            self.ist_list[c]["abs_res_x1"] += abs_res(
                gt_box[idx].tensor[:, 2], pred_box[idx].tensor[:, 2]
            ).tolist()
            self.ist_list[c]["abs_res_y1"] += abs_res(
                gt_box[idx].tensor[:, 3], pred_box[idx].tensor[:, 3]
            ).tolist()

        if verbose:
            print(f"Added all instances (with learnable scoring) for image {img_id}.") 