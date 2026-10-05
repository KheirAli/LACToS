from asyncio import constants
from typing import Any
import os
import torch
from unet import *
from dataset import *
from visualize import *
from anomaly_map import *
from metrics import *
from feature_extractor import *
from reconstruction import *
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

class DDAD:
    def __init__(self, unet, config) -> None:
        self.test_dataset = Dataset_maker(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=False,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size= config.data.test_batch_size,
            shuffle=False,
            num_workers= config.model.num_workers,
            drop_last=False,
        )
        self.unet = unet
        self.config = config
        self.reconstruction = Reconstruction(self.unet, self.config)
        # self.transform = transforms.Compose([
        #                     transforms.CenterCrop((256)), #224
        #                 ])
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
        ])

    def __call__(self) -> Any:
        # Env-guarded runtime instrumentation (DDAD_TIMING_JSON=/path.json).
        # Records per-image reconstruction / heat-map wall clock and peak CUDA
        # memory for the runtime cost table; inert when the var is unset.
        import json as _json
        import time as _time
        _tpath = os.environ.get("DDAD_TIMING_JSON")
        _trec = {"per_image": []} if _tpath else None
        if _tpath:
            torch.cuda.reset_peak_memory_stats()

        _t0 = _time.perf_counter()
        feature_extractor = domain_adaptation(self.unet, self.config, fine_tune=False)
        feature_extractor.eval()
        if _tpath:
            torch.cuda.synchronize()
            _trec["feature_extractor_setup_s"] = _time.perf_counter() - _t0

        labels_list = []
        predictions= []
        anomaly_map_list = []
        gt_list = []
        reconstructed_list = []
        forward_list = []



        # with torch.no_grad():
        #     for i, (input, gt, labels) in enumerate(self.testloader):
        #         input = input.to(self.config.model.device)
        #         x0 = self.reconstruction(input, input, self.config.model.w)[-1]
        #         anomaly_map = heat_map(x0, input, feature_extractor, self.config)
        #         if i < 3:
        #             print(f"\n[DEBUG] Image {i}")
        #             print(f"  input shape:       {input.shape}")
        #             print(f"  x0 shape:          {x0.shape}")
        #             print(f"  anomaly_map shape: {anomaly_map.shape}  "
        #                 f"min={anomaly_map.min():.4f} max={anomaly_map.max():.4f}")
        #             print(f"  gt shape (pre-crop):  {gt.shape}  "
        #                 f"positive pixels={gt.sum().item():.0f}  "
        #                 f"label={labels}")

        with torch.no_grad():
            for i, (input, gt, labels) in enumerate(self.testloader):
                input = input.to(self.config.model.device)
                if _tpath:
                    torch.cuda.synchronize(); _t0 = _time.perf_counter()
                x0 = self.reconstruction(input, input, self.config.model.w)[-1]
                if _tpath:
                    torch.cuda.synchronize()
                    _t_recon = _time.perf_counter() - _t0
                    _t0 = _time.perf_counter()
                anomaly_map = heat_map(x0, input, feature_extractor, self.config)
                if _tpath:
                    torch.cuda.synchronize()
                    _trec["per_image"].append({
                        "i": i, "reconstruction_s": _t_recon,
                        "heat_map_s": _time.perf_counter() - _t0})
                anomaly_map = self.transform(anomaly_map)
                gt = self.transform(gt)

                # ── DEEP DEBUG for first few images ────────────────────────
                # if i < 3:
                #     from sklearn.metrics import roc_auc_score
                    
                #     amap_np = anomaly_map.flatten().cpu().numpy().astype(np.float32)
                #     gt_np   = gt.flatten().cpu().numpy().astype(int)
                    
                #     print(f"\n[VERIFY] Image {i}")
                #     print(f"  GT positive pixels: {gt_np.sum()} / {len(gt_np)} "
                #         f"({100*gt_np.sum()/len(gt_np):.2f}%)")
                    
                #     # Stats split by GT class
                #     scores_in_gt  = amap_np[gt_np == 1]
                #     scores_out_gt = amap_np[gt_np == 0]
                #     print(f"  Scores INSIDE GT:  mean={scores_in_gt.mean():.4f}  "
                #         f"median={np.median(scores_in_gt):.4f}  "
                #         f"min={scores_in_gt.min():.4f}  max={scores_in_gt.max():.4f}")
                #     print(f"  Scores OUTSIDE GT: mean={scores_out_gt.mean():.4f}  "
                #         f"median={np.median(scores_out_gt):.4f}  "
                #         f"min={scores_out_gt.min():.4f}  max={scores_out_gt.max():.4f}")
                    
                #     # Compute AUROC with sklearn directly
                #     auc = roc_auc_score(gt_np, amap_np)
                #     print(f"  sklearn AUROC: {auc:.4f}")
                    
                #     # Manually verify: fraction of (in,out) pairs where in > out
                #     # AUROC == P(score_in > score_out)
                #     n_sample = min(2000, len(scores_in_gt), len(scores_out_gt))
                #     ins  = np.random.choice(scores_in_gt,  n_sample, replace=False)
                #     outs = np.random.choice(scores_out_gt, n_sample, replace=False)
                #     wins = np.mean(ins > outs)
                #     ties = np.mean(ins == outs)
                #     print(f"  Manual P(in > out) sampled: {wins:.4f} (ties: {ties:.4f})")
                    
                #     # What threshold would catch all GT pixels?
                #     # i.e. threshold = min score inside GT
                #     t_recall = scores_in_gt.min()
                #     fp_at_recall = (scores_out_gt >= t_recall).sum()
                #     print(f"  Threshold for 100% recall: {t_recall:.4f}")
                #     print(f"  False positives at that threshold: {fp_at_recall} "
                #         f"({100*fp_at_recall/len(scores_out_gt):.1f}% of background)")
                anomaly_map = self.transform(anomaly_map)
                gt = self.transform(gt)
                if i < 3:
                    print(f"  anomaly_map shape (post-crop): {anomaly_map.shape}")
                    print(f"  gt shape (post-crop):          {gt.shape}  "
                        f"positive pixels={gt.sum().item():.0f}")
                    # Check if crop ate the defect
                    lost = gt.sum().item() - gt.sum().item()
                    if lost > 0:
                        print(f"  *** CenterCrop removed {lost:.0f} GT defect pixels! ***")
                forward_list.append(input)
                anomaly_map_list.append(anomaly_map)


                gt_list.append(gt)
                reconstructed_list.append(x0)
                for pred, label in zip(anomaly_map, labels):
                    labels_list.append(0 if label == 'good' else 1)
                    predictions.append(torch.max(pred).item())

        if os.environ.get("DDAD_SAVE_MAPS"):
            torch.save({"maps": torch.cat(anomaly_map_list, 0).cpu(),
                        "gts": torch.cat(gt_list, 0).cpu(),
                        "files": list(self.test_dataset.image_files)},
                       os.environ["DDAD_SAVE_MAPS"])
            print(f"[DDAD_SAVE_MAPS] saved {len(self.test_dataset.image_files)} maps "
                  f"-> {os.environ['DDAD_SAVE_MAPS']}")

        metric = Metric(labels_list, predictions, anomaly_map_list, gt_list, self.config)
        try:
            metric.optimal_threshold()
            
        except (ValueError, Exception) as e:
            print(f"[WARN] Image-level metrics skipped: {e}")
            pass
        if self.config.metrics.auroc:
            try:
                print('AUROC: ({:.1f},{:.1f})'.format(metric.image_auroc() * 100, metric.pixel_auroc() * 100))
            except ValueError as e:
                print(f"[WARN] Image AUROC skipped (single class): {e}")
                print('Pixel AUROC: ({:.1f})'.format(metric.pixel_auroc() * 100))
            per_img_px_auroc = metric.pixel_auroc_per_image_mean()
        print('Per-image Pixel AUROC (mean): {:.1f}'.format(per_img_px_auroc * 100))
        mse_score = metric.mse()
        print(f"MSE: {mse_score:.4f}")
        ap_score = metric.pixel_ap_per_image_mean()                        # ← add
        print(f"Per-image Pixel AP (mean): {ap_score * 100:.1f}")
        if self.config.metrics.pro:
            print('PRO: {:.1f}'.format(metric.pixel_pro() * 100))
        # metric.optimal_threshold()
        if self.config.metrics.pro:
            print('PRO: {:.1f}'.format(metric.pixel_pro() * 100))
        if self.config.data.mask:   # only meaningful when GT masks are loaded
            snr_val = metric.snr()
            print(f'SNR: {snr_val:.4f}')
        else:
            print('[WARN] SNR skipped — set mask: True in config to enable')
        if self.config.metrics.misclassifications:
            metric.miscalssified()
        reconstructed_list = torch.cat(reconstructed_list, dim=0)
        forward_list = torch.cat(forward_list, dim=0)
        anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
        pred_mask = (anomaly_map_list > metric.threshold).float()
        gt_list = torch.cat(gt_list, dim=0)
        if not os.path.exists('results'):
                os.mkdir('results')
        if self.config.metrics.visualisation:
            visualize(forward_list, reconstructed_list, gt_list, pred_mask, anomaly_map_list, self.config.data.category)

        if _tpath:
            _trec["peak_cuda_alloc_mb"] = torch.cuda.max_memory_allocated() / 2**20
            _trec["peak_cuda_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
            _trec["n_images"] = len(_trec["per_image"])
            _trec["denoise_steps_per_image"] = len(
                range(0, self.config.model.test_trajectoy_steps, self.config.model.skip))
            _trec["test_batch_size"] = self.config.data.test_batch_size
            _trec["image_size"] = self.config.data.image_size
            with open(_tpath, "w") as _fh:
                _json.dump(_trec, _fh, indent=1)
            print(f"[timing] wrote {_trec['n_images']} records -> {_tpath}")
