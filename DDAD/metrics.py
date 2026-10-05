import torch
from torchmetrics import ROC, AUROC, F1Score
import os
from torchvision.transforms import transforms
from skimage import measure
import pandas as pd
from statistics import mean
import numpy as np
from sklearn.metrics import auc
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve


class Metric:
    def __init__(self,labels_list, predictions, anomaly_map_list, gt_list, config) -> None:
        self.labels_list = labels_list
        self.predictions = predictions
        self.anomaly_map_list = anomaly_map_list
        self.gt_list = gt_list
        self.config = config
        self.threshold = 0.5
    
    def image_auroc(self):
        auroc_image = roc_auc_score(self.labels_list, self.predictions)
        return auroc_image
    def snr(self):
        """
        SNR = (mean anomaly score in GT-positive pixels) /
            (std  anomaly score in GT-negative pixels)
        Higher = better separation between anomalous and normal regions.
        """
        import torch
        maps  = torch.cat(self.anomaly_map_list, dim=0)  # (N,1,H,W)
        masks = torch.cat(self.gt_list,          dim=0)  # (N,1,H,W)

        maps  = maps.squeeze(1).cpu().numpy()   # (N,H,W)
        masks = masks.squeeze(1).cpu().numpy()  # (N,H,W)

        anomalous = maps[masks > 0]
        normal    = maps[masks == 0]

        if len(anomalous) == 0 or len(normal) == 0:
            return float("nan")

        return float(anomalous.mean() / (normal.std() + 1e-8))
    def pixel_ap_per_image_mean(self):
        """
        Average Precision per image, then averaged over defective images.
        Same structure as pixel_auroc_per_image_mean — skips good images.
        AP = area under precision-recall curve (sklearn average_precision_score).
        """
        from sklearn.metrics import average_precision_score

        per_image_ap = []
        skipped = 0

        for amap, gt in zip(self.anomaly_map_list, self.gt_list):
            gt_flat = gt.flatten().cpu().numpy().astype(int)

            # Skip images with no defect pixels — AP undefined
            if gt_flat.sum() == 0:
                skipped += 1
                continue

            amap_np = amap.flatten().cpu().numpy().astype(np.float32)
            mn, mx  = amap_np.min(), amap_np.max()
            amap_norm = (amap_np - mn) / (mx - mn + 1e-8)

            try:
                ap_val = average_precision_score(gt_flat, amap_norm)
                per_image_ap.append(ap_val)
            except Exception as e:
                print(f"  [WARN] AP skipping image: {e}")
                skipped += 1

        print(f"  Per-image pixel AP: n={len(per_image_ap)} defective, "
            f"skipped={skipped} good/invalid images")
        if not per_image_ap:
            return float("nan")

        print(list(enumerate(per_image_ap)))
        print(f"  Individual AP scores: min={min(per_image_ap):.3f}, "
            f"max={max(per_image_ap):.3f}, "
            f"mean={np.mean(per_image_ap):.3f}")
        return float(np.mean(per_image_ap))
    def mse(self):
        import torch.nn.functional as F

        per_image_mse = []
        for a, g in zip(self.anomaly_map_list, self.gt_list):
            # Ensure 4D: (N, C, H, W)
            if a.dim() == 3: a = a.unsqueeze(0)   # (C,H,W) → (1,C,H,W)
            if g.dim() == 3: g = g.unsqueeze(0)

            # Already 256×256 per ddad.py transform — resize only if needed
            if a.shape[-2:] != (256, 256):
                a = F.interpolate(a.float(), size=(256, 256),
                                mode="bilinear", align_corners=False)
            if g.shape[-2:] != (256, 256):
                g = F.interpolate(g.float(), size=(256, 256), mode="nearest")

            g_np = g.squeeze().cpu().numpy().astype(np.float32)
            if g_np.sum() == 0:
                continue

            a_np = a.squeeze().cpu().numpy().astype(np.float32)
            mn, mx = a_np.min(), a_np.max()
            a_norm = (a_np - mn) / (mx - mn + 1e-8)

            per_image_mse.append(float(np.mean((a_norm - g_np) ** 2)))

        if not per_image_mse:
            return float("nan")

        mean_mse = float(np.mean(per_image_mse))
        print(f"  MSE (defective images): n={len(per_image_mse)}, mean={mean_mse:.4f}")
        return mean_mse
    def pixel_auroc(self):
        resutls_embeddings = self.anomaly_map_list[0]
        for feature in self.anomaly_map_list[1:]:
            resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
        resutls_embeddings =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min())) 

        gt_embeddings = self.gt_list[0]
        for feature in self.gt_list[1:]:
            gt_embeddings = torch.cat((gt_embeddings, feature), 0)

        resutls_embeddings = resutls_embeddings.clone().detach().requires_grad_(False)
        gt_embeddings = gt_embeddings.clone().detach().requires_grad_(False)

        auroc_p = AUROC(task="binary")
        
        gt_embeddings = torch.flatten(gt_embeddings).type(torch.bool).cpu().detach()
        resutls_embeddings = torch.flatten(resutls_embeddings).cpu().detach()
        auroc_pixel = auroc_p(resutls_embeddings, gt_embeddings)
        return auroc_pixel
    
    def optimal_threshold(self):
        if len(self.labels_list) == 0 or len(set(self.labels_list)) < 2:
            print("[WARN] Skipping optimal_threshold — no image-level labels collected.")
            self.threshold = 0.5  # dummy fallback
            return
        fpr, tpr, thresholds = roc_curve(self.labels_list, self.predictions)

        # Calculate Youden's J statistic for each threshold
        youden_j = tpr - fpr

        # Find the optimal threshold that maximizes Youden's J statistic
        optimal_threshold_index = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_threshold_index]
        self.threshold = optimal_threshold
        return optimal_threshold
    
    def pixel_auroc_per_image_mean(self):
        from sklearn.metrics import roc_auc_score
        per_image_aurocs = []
        skipped = 0

        for amap, gt in zip(self.anomaly_map_list, self.gt_list):
            gt_flat = gt.flatten().cpu().numpy().astype(int)

            # Skip images with no defect pixels — AUROC undefined
            if gt_flat.sum() == 0:
                skipped += 1
                continue

            # Normalize per image independently
            amap_np = amap.flatten().cpu().numpy().astype(np.float32)
            mn, mx = amap_np.min(), amap_np.max()
            amap_norm = (amap_np - mn) / (mx - mn + 1e-8)

            try:
                auroc_val = roc_auc_score(gt_flat, amap_np)
                per_image_aurocs.append(auroc_val)
            except Exception as e:
                print(f"  [WARN] Skipping image: {e}")
                skipped += 1

        print(f"  Per-image pixel AUROC: n={len(per_image_aurocs)} defective, "
            f"skipped={skipped} good/invalid images")
        if not per_image_aurocs:
            return float("nan")
        print(list(enumerate(per_image_aurocs)))
        print(f"  Individual scores: min={min(per_image_aurocs):.3f}, "
            f"max={max(per_image_aurocs):.3f}, "
            f"mean={np.mean(per_image_aurocs):.3f}")
        return float(np.mean(per_image_aurocs))
    # def pixel_auroc_per_image_mean(self):
    #     """
    #     Compute pixel AUROC independently for each image, then average.
    #     Skips 'good' images (no positive GT pixels) since AUROC is undefined there.
    #     """
    #     from torchmetrics import AUROC as TM_AUROC
    #     auroc_fn = TM_AUROC(task="binary")
        
    #     per_image_aurocs = []
    #     skipped = 0

    #     for amap, gt in zip(self.anomaly_map_list, self.gt_list):
    #         gt_flat = gt.flatten().bool().cpu()
            
    #         # Skip images with no defect pixels (good images — AUROC undefined)
    #         if gt_flat.sum() == 0:
    #             skipped += 1
    #             continue

    #         # Normalize per image independently
    #         mn, mx = amap.min(), amap.max()
    #         amap_norm = (amap - mn) / (mx - mn + 1e-8)
    #         scores_flat = amap_norm.flatten().cpu()

    #         try:
    #             auroc_val = auroc_fn(scores_flat, gt_flat).item()
    #             per_image_aurocs.append(auroc_val)
    #         except Exception as e:
    #             skipped += 1

    #     if not per_image_aurocs:
    #         return float("nan")

    #     mean_auroc = float(np.mean(per_image_aurocs))
    #     print(list(enumerate(per_image_aurocs)))
    #     print(f"  Per-image pixel AUROC: n={len(per_image_aurocs)} defective images, "
    #         f"skipped={skipped} good images")
    #     return mean_auroc

    def pixel_pro(self):
        #https://github.com/hq-deng/RD4AD/blob/main/test.py#L337
        def _compute_pro(masks, amaps, num_th = 200):
            resutls_embeddings = amaps[0]
            for feature in amaps[1:]:
                resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
            amaps =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min())) 
            amaps = amaps.squeeze(1)
            amaps = amaps.cpu().detach().numpy()
            gt_embeddings = masks[0]
            for feature in masks[1:]:
                gt_embeddings = torch.cat((gt_embeddings, feature), 0)
            masks = gt_embeddings.squeeze(1).cpu().detach().numpy()
            min_th = amaps.min()
            max_th = amaps.max()
            delta = (max_th - min_th) / num_th
            binary_amaps = np.zeros_like(amaps)
            df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])

            for th in np.arange(min_th, max_th, delta):
                binary_amaps[amaps <= th] = 0
                binary_amaps[amaps > th] = 1

                pros = []
                for binary_amap, mask in zip(binary_amaps, masks):
                    for region in measure.regionprops(measure.label(mask)):
                        axes0_ids = region.coords[:, 0]
                        axes1_ids = region.coords[:, 1]
                        tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                        pros.append(tp_pixels / region.area)

                inverse_masks = 1 - masks
                fp_pixels = np.logical_and(inverse_masks , binary_amaps).sum()
                fpr = fp_pixels / inverse_masks.sum()
                # print(f"Threshold: {th}, FPR: {fpr}, PRO: {mean(pros)}")

                df = pd.concat([df, pd.DataFrame({"pro": mean(pros), "fpr": fpr, "threshold": th}, index=[0])], ignore_index=True)
                # df = df.concat({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

            # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
            df = df[df["fpr"] < 0.3]
            df["fpr"] = df["fpr"] / df["fpr"].max()

            pro_auc = auc(df["fpr"], df["pro"])
            return pro_auc
        
        pro = _compute_pro(self.gt_list, self.anomaly_map_list, num_th = 200)
        return pro
    

    def miscalssified(self):
        predictions = torch.tensor(self.predictions)
        labels_list = torch.tensor(self.labels_list)
        predictions0_1 = (predictions > self.threshold).int()
        for i,(l,p) in enumerate(zip(labels_list, predictions0_1)):
            print('Sample : ', i, ' predicted as: ',p.item() ,' label is: ',l.item(),'\n' ) if l != p else None

