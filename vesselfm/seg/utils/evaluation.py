from pathlib import Path

import numpy as np
import torch
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.measure import euler_number, label
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import SimpleITK as sitk
from torch.utils.data import Dataset


class PretrainEvaluationDataset(Dataset):
    def __init__(self, data_path):
        data_dir = Path(data_path).resolve()
        self.val_data = {
            "deepvess": [
                torch.tensor(read_nifti(data_dir / "deepvess.nii"))[None],
                torch.tensor(read_nifti(data_dir / "deepvess_mask.nii"))[None],
            ],
            "deepvesselnet": [
                torch.tensor(read_nifti(data_dir / "deepvesselnet.nii"))[None],
                torch.tensor(read_nifti(data_dir / "deepvesselnet_mask.nii"))[None],
            ],
            "lightsheet": [
                torch.tensor(read_nifti(data_dir / "lightsheet.nii"))[None],
                torch.tensor(read_nifti(data_dir / "lightsheet_mask.nii"))[None],
            ],
            "minivess": [
                torch.tensor(read_nifti(data_dir / "minivess.nii"))[None],
                torch.tensor(read_nifti(data_dir / "minivess_mask.nii"))[None],
            ],
            "tubetk": [
                torch.tensor(read_nifti(data_dir / "tubetk.nii"))[None],
                torch.tensor(read_nifti(data_dir / "tubetk_mask.nii"))[None],
            ],
        }
        self._samples = list(self.val_data.keys())

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        name = self._samples[idx]
        image, mask = self.val_data[self._samples[idx]]
        return image, mask, name

# Insert your existing betti_number methods here
def betti_number(self, img):
    # Your implementation (binary image -> [b0, b1, b2])
    assert img.ndim == 3
    N6 = 1
    N26 = 3

    padded = np.pad(img, pad_width=1)
    assert set(np.unique(padded)).issubset({0, 1})

    _, b0 = label(padded, return_num=True, connectivity=N26)
    euler_char_num = euler_number(padded, connectivity=N26)
    _, b2 = label(1 - padded, return_num=True, connectivity=N6)

    b2 -= 1
    b1 = b0 + b2 - euler_char_num
    return [b0, b1, b2]


class Evaluator:
    def extract_labels(self, gt_array, pred_array):
        """
        Adapted from https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py#L18.
        """
        labels_gt = np.unique(gt_array)
        labels_pred = np.unique(pred_array)
        labels = list(set().union(labels_gt, labels_pred))
        labels = [int(x) for x in labels]
        return labels

    def betti_number_error(self, gt, pred):
        """
        Adapted from https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py#L250.
        """
        labels = self.extract_labels(gt_array=gt, pred_array=pred)
        labels.remove(0)

        if len(labels) == 0:
            return 0, 0
        assert len(labels) == 1 and 1 in labels, "Invalid binary segmentatio.n"

        gt_betti_numbers = self.betti_number(gt)
        pred_betti_numbers = self.betti_number(pred)
        betti_0_error = abs(pred_betti_numbers[0] - gt_betti_numbers[0])
        betti_1_error = abs(pred_betti_numbers[1] - gt_betti_numbers[1])
        return betti_0_error, betti_1_error

    def betti_number(self, img):
        """
        Adapted from https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py#L186.
        """
        assert img.ndim == 3
        N6 = 1
        N26 = 3

        padded = np.pad(img, pad_width=1)
        assert set(np.unique(padded)).issubset({0, 1})

        _, b0 = label(padded, return_num=True, connectivity=N26)
        euler_char_num = euler_number(padded, connectivity=N26)
        _, b2 = label(1 - padded, return_num=True, connectivity=N6)

        b2 -= 1
        b1 = b0 + b2 - euler_char_num
        return [b0, b1, b2]

    def cl_dice(self, v_p, v_l):
        """
        Adapted from https://github.com/jocpae/clDice/blob/master/cldice_metric/cldice.py.
        """
        def cl_score(v, s):
            return np.sum(v * s) / np.sum(s)

        if len(v_p.shape) == 2:
            tprec = cl_score(v_p, skeletonize(v_l))
            tsens = cl_score(v_l, skeletonize(v_p))
        elif len(v_p.shape) == 3:
            tprec = cl_score(v_p, skeletonize_3d(v_l))
            tsens = cl_score(v_l, skeletonize_3d(v_p))
        else:
            raise ValueError(f"Invalid shape for cl_dice: {v_p.shape}")
        return 2 * tprec * tsens / (tprec + tsens + np.finfo(float).eps)

    def estimate_metrics(self, pred_seg, gt_seg, threshold=0.5, fast=False):
        metrics = {}
        pred_seg_thresh = (pred_seg >= threshold).float().cpu()

        # estimate metrics
        tn, fp, fn, tp = confusion_matrix(
            gt_seg.flatten().cpu().clone().numpy(),
            pred_seg_thresh.flatten().cpu().clone().numpy(),
            labels=[0, 1],
        ).ravel()

        if fast:
            metrics["dice"] = (2 * tp) / (2 * tp + fp + fn)
            return metrics

        roc_auc = roc_auc_score(
            gt_seg.flatten().cpu().clone().detach().numpy(),
            pred_seg.flatten().cpu().clone().detach().numpy(),
        )

        pr_auc = average_precision_score(
            gt_seg.flatten().cpu().clone().detach().numpy(),
            pred_seg.flatten().cpu().clone().detach().numpy(),
        )

        cldice = self.cl_dice(
            pred_seg_thresh.squeeze().cpu().clone().detach().byte().numpy(),
            gt_seg.squeeze().cpu().clone().detach().byte().numpy(),
        )

        betti_0_error, betti_1_error = self.betti_number_error(
            gt_seg.squeeze().cpu().clone().detach().int().numpy(),
            pred_seg_thresh.squeeze().cpu().clone().detach().int().numpy(),
        )
        betti_0, betti_1, betti_2 = self.betti_number(
            pred_seg_thresh.squeeze().cpu().clone().detach().int().numpy()
        )

        metrics["recall_tpr_sensitivity"] = tp / (tp + fn)
        metrics["fpr"] = fp / (fp + tn)
        metrics["precision"] = tp / (tp + fp)
        metrics["specificity"] = tn / (tn + fp)
        metrics["jaccard_iou"] = tp / (tp + fp + fn)
        metrics["dice"] = (2 * tp) / (2 * tp + fp + fn)
        metrics["cldice"] = cldice
        metrics["accuracy"] = (tp + tn) / (tn + fp + tp + fn)
        metrics["roc_auc"] = roc_auc
        metrics["pr_auc_ap"] = pr_auc
        metrics["betti_0_error"] = betti_0_error
        metrics["betti_1_error"] = betti_1_error
        metrics["betti_0"] = betti_0
        metrics["betti_1"] = betti_1
        metrics["betti_2"] = betti_2
        return metrics


class MulticlassEvaluator(Evaluator):
    def __init__(self, ignore_index=[0], log_mode=False):
        """
        ignore_index: class to exclude from evaluation (e.g., void)
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.log_mode = log_mode

    def estimate_metrics(self, pred_logits, gt, threshold=0.5, compute_auc=True, fast=False):
        """
        pred_logits: numpy array (H,W,(D),C) or torch tensor of class scores
        gt: numpy array (H,W,(D)) integer labels in [0..num_classes-1]
        num_classes: how many classes
        """
        metrics = {}
        # Convert logits to predicted label map
        if hasattr(pred_logits, "cpu"):
            pred = pred_logits.cpu().detach().numpy()
        if hasattr(gt, "cpu"):
            gt = gt.cpu().detach().numpy()

        num_classes = np.unique(gt)
        # If last dim is classes
        pred_labels = np.argmax(pred, axis=0)

        per_class_metrics = {}
        betti_per_class = {}
        betti_err_per_class = {}
        cldice_per_class = {}
        TP_sum = FP_sum = FN_sum = TN_sum = 0
        for c in num_classes:
            if c in self.ignore_index:
                continue

            # Build one-vs-all masks
            gt_mask = (gt == c).astype(int)
            pred_mask = (pred_labels == c).astype(int)

            # Confusion counts
            tp = np.sum((pred_mask == 1) & (gt_mask == 1))
            fp = np.sum((pred_mask == 1) & (gt_mask == 0))
            fn = np.sum((pred_mask == 0) & (gt_mask == 1))
            tn = np.sum((pred_mask == 0) & (gt_mask == 0))

            TP_sum += tp;
            FP_sum += fp;
            FN_sum += fn;
            TN_sum += tn

            # Basic metrics
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
            iou = tp / (tp + fp + fn + 1e-7)

            roc_auc = roc_auc_score(gt_mask.flatten(), pred[c].flatten())
            pr_auc = average_precision_score(gt_mask.flatten(), pred[c].flatten())

            per_class_metrics[c] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "dice": dice,
                "iou": iou,
                "precision": tp / (tp + fp + 1e-7),
                "recall": tp / (tp + fn + 1e-7),
                'auroc': roc_auc,
                'aupr': pr_auc,
            }

            # Betti numbers per class
            # (use inherited functions)
            try:
                b0_gt, b1_gt, b2_gt = self.betti_number(gt_mask)
            except Exception:
                # If 2D, b2 is zero
                b0_gt, b1_gt, b2_gt = [None] * 3

            try:
                b0_pred, b1_pred, b2_pred = self.betti_number(pred_mask)
            except Exception:
                b0_pred, b1_pred, b2_pred = [None] * 3

            betti_per_class[c] = {
                "b0_gt": b0_gt,
                "b1_gt": b1_gt,
                "b2_gt": b2_gt,
                "b0_pred": b0_pred,
                "b1_pred": b1_pred,
                "b2_pred": b2_pred,
            }

            # Betti errors
            betti_err_per_class[c] = {
                "betti_0_error": abs(b0_pred - b0_gt)
                if b0_gt is not None
                else None,
                "betti_1_error": abs(b1_pred - b1_gt)
                if b1_gt is not None
                else None,
                "betti_2_error": abs(b2_pred - b2_gt)
                if b2_gt is not None
                else None,
            }

            # clDice
            # rely on inherited cl_dice; pass binary masks
            try:
                cldice_val = self.cl_dice(
                    pred_mask.astype(np.uint8),
                    gt_mask.astype(np.uint8),
                )
            except Exception:
                cldice_val = None

            cldice_per_class[c] = cldice_val

        # Collect across classes
        if not self.log_mode:
            metrics["per_class"] = per_class_metrics
            metrics["betti_per_class"] = betti_per_class
            metrics["betti_error_per_class"] = betti_err_per_class
            metrics["cldice_per_class"] = cldice_per_class

        # Macro averages
        metrics["dice_macro"] = np.mean(
            [v["dice"] for v in per_class_metrics.values()]
        )
        metrics["iou_macro"] = np.mean(
            [v["iou"] for v in per_class_metrics.values()]
        )
        metrics["cldice_macro"] = np.mean(
            [v for v in cldice_per_class.values() if v is not None]
        )

        # Macro Betti errors (ignoring None)
        b0_errs = [
            v["betti_0_error"]
            for v in betti_err_per_class.values()
            if v["betti_0_error"] is not None
        ]
        b1_errs = [
            v["betti_1_error"]
            for v in betti_err_per_class.values()
            if v["betti_1_error"] is not None
        ]
        b2_errs = [
            v["betti_2_error"]
            for v in betti_err_per_class.values()
            if v["betti_2_error"] is not None
        ]

        metrics["betti_0_error_macro"] = (
            np.mean(b0_errs) if len(b0_errs) > 0 else None
        )
        metrics["betti_1_error_macro"] = (
            np.mean(b1_errs) if len(b1_errs) > 0 else None
        )
        metrics["betti_2_error_macro"] = (
            np.mean(b2_errs) if len(b2_errs) > 0 else None
        )

        # Optional: overall AUC & PR-AUC per class
        # Only if pred_logits are continuous and shapes match
        if compute_auc:
            try:
                # Flatten for one-hot
                gt_onehot = self._one_hot(gt, len(num_classes))
                pred_soft = pred / np.sum(pred, axis=0, keepdims=True)
                # roc_auc (macro)
                metrics["roc_auc_macro"] = roc_auc_score(
                    gt_onehot.reshape(-1, len(num_classes)),
                    pred_soft.reshape(-1, len(num_classes)),
                    average="macro",
                    multi_class="ovo",
                )
                metrics["pr_auc_macro"] = average_precision_score(
                    gt_onehot.reshape(-1, len(num_classes)),
                    pred_soft.reshape(-1, len(num_classes)),
                )
            except Exception:
                metrics["roc_auc_macro"] = None
                metrics["pr_auc_macro"] = None

        def _macro(key):
            vals = [per_class_metrics[c][key] for c in per_class_metrics.keys()]
            return float(np.mean(vals)) if len(vals) else 0.0

        if not self.log_mode:
            metrics['macro_avg'] = {
                'dice': _macro('dice'),
                'recall': _macro('recall'),
                'precision': _macro('precision'),
            }

            # Micro average (pool TP/FP/FN/TN over classes)
            dice_micro = (2 * TP_sum) / (2 * TP_sum + FP_sum + FN_sum) if (2 * TP_sum + FP_sum + FN_sum) > 0 else 0.0
            tpr_micro = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) > 0 else 0.0
            fpr_micro = FP_sum / (FP_sum + TN_sum) if (FP_sum + TN_sum) > 0 else 0.0
            ppv_micro = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) > 0 else 0.0
            npv_micro = TN_sum / (TN_sum + FN_sum) if (TN_sum + FN_sum) > 0 else 0.0

            metrics['micro_avg'] = {
                'dice': float(dice_micro),
                'recall': float(tpr_micro),
                #'FPR': float(fpr_micro),
                'precision': float(ppv_micro),
                #'NPV': float(npv_micro),
            }

        metrics["dice"] = metrics["dice_macro"]

        return metrics
    def _one_hot(self, arr, num_classes):
        """
        Helper: one-hot encode integer labels
        """
        oh = np.eye(num_classes)[arr.reshape(-1)]
        return oh.reshape(*arr.shape, num_classes)

def read_nifti(path: str):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def calculate_mean_metrics(results, round_to=2):
    mean = {}
    for k in results[0].keys():
        numbers = [r[k] for r in results]
        numbers = [n for n in numbers if np.isnan(n) == False]
        mean[k] = np.mean(numbers)

        if "dice" in k:
            mean[k] = mean[k] * 100
        mean[k] = np.round(mean[k], round_to)
    return mean
