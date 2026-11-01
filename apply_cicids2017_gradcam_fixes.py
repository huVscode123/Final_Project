# -*- coding: utf-8 -*-
"""Apply targeted CICIDS2017 CNN-VAE Grad-CAM fixes.

Run from the project root:
    python apply_cicids2017_gradcam_fixes.py

Only core/cnn_gradcam/gradcam_core.py and core/run_gradcam_cicids2017.py are
modified. A .pre_gradcam_fix.bak backup is created beside each source file.
This patch does not train the model, alter its checkpoint, or change the data.
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path.cwd()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"[{label}] 預期 1 個符合片段，實際 {count} 個；停止修改。")
    return source.replace(old, new, 1)


def patch_gradcam_core(s: str) -> str:
    s = replace_once(s,
'''        self._activation: Optional[torch.Tensor] = None
        self._gradient: Optional[torch.Tensor] = None
        self._forward_handle = self._target_module.register_forward_hook(
''',
'''        self._activation: Optional[torch.Tensor] = None
        self._gradient: Optional[torch.Tensor] = None
        # Per-sample diagnostics distinguish an empty positive-evidence CAM
        # from a missing/zero gradient hook.
        self.last_diagnostics: list[dict[str, float]] = []
        self._forward_handle = self._target_module.register_forward_hook(
''', 'core:init diagnostics')

    s = replace_once(s,
'''    def _resolve_target_layer(self, layer_name: str) -> str:
        modules = dict(self.model.named_modules())
        if layer_name in modules:
            module = modules[layer_name]
            if isinstance(module, nn.Sequential):
                candidates = [
                    name
                    for name, child in self.model.named_modules()
                    if name.startswith(layer_name + ".")
                    and isinstance(child, nn.Conv2d)
                ]
                if candidates:
                    return candidates[-1]
            if not isinstance(module, nn.Conv2d):
                raise TypeError(
                    f"目標層 {layer_name!r} 不是 Conv2d；請指定實際卷積層名稱。"
                )
            return layer_name

        candidates = [
            name
            for name, module in self.model.named_modules()
            if layer_name in name and isinstance(module, nn.Conv2d)
        ]
        if candidates:
            return candidates[-1]

        raise ValueError(
            f"找不到目標層 {layer_name!r}。可用 Conv2d 層："
            f"{self.list_target_layers()}"
        )
''',
'''    def _resolve_target_layer(self, layer_name: str) -> str:
        modules = dict(self.model.named_modules())
        if layer_name in modules:
            module = modules[layer_name]
            if isinstance(module, nn.Sequential):
                candidates = [
                    name for name, child in self.model.named_modules()
                    if name.startswith(layer_name + ".")
                    and isinstance(child, nn.Conv2d)
                ]
                if candidates:
                    return candidates[-1]
                raise TypeError(f"Sequential 目標層 {layer_name!r} 內找不到 Conv2d。")
            # Exact spatial modules including the post-BN/ReLU feature layer
            # are accepted. The forward hook validates the 4-D output shape.
            return layer_name

        supported = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
                     nn.MaxPool2d, nn.AdaptiveAvgPool2d)
        candidates = [
            name for name, module in self.model.named_modules()
            if layer_name in name and isinstance(module, supported)
        ]
        if candidates:
            return candidates[-1]
        raise ValueError(
            f"找不到目標層 {layer_name!r}。可用空間層："
            f"{self.list_spatial_target_layers()}"
        )
''', 'core:allow post-ReLU target')

    s = replace_once(s,
'''    def list_target_layers(self) -> list[str]:
        return [
            name
            for name, module in self.model.named_modules()
            if name and isinstance(module, nn.Conv2d)
        ]
''',
'''    def list_target_layers(self) -> list[str]:
        # Backward-compatible list containing Conv2d modules only.
        return [
            name for name, module in self.model.named_modules()
            if name and isinstance(module, nn.Conv2d)
        ]

    def list_spatial_target_layers(self) -> list[str]:
        supported = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
                     nn.MaxPool2d, nn.AdaptiveAvgPool2d)
        return [
            name for name, module in self.model.named_modules()
            if name and isinstance(module, supported)
        ]
''', 'core:list spatial layers')

    s = replace_once(s,
'''    def _forward_hook(self, _module, _inputs, output) -> None:
        activation = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(activation):
            raise TypeError("目標卷積層的輸出不是 Tensor。")
        self._activation = activation
''',
'''    def _forward_hook(self, _module, _inputs, output) -> None:
        activation = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(activation) or activation.ndim != 4:
            shape = tuple(activation.shape) if torch.is_tensor(activation) else type(activation)
            raise TypeError(
                "Grad-CAM 目標層必須輸出 (B,C,H,W) 空間特徵，"
                f"目前輸出={shape}。請指定 Conv2d/BN/ReLU 等空間層。"
            )
        self._activation = activation
''', 'core:4d hook validation')

    s = replace_once(s,
'''    # ------------------------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------------------------
    def _gradcam(
''',
'''    @staticmethod
    def _gradient_diagnostics(
        activation: torch.Tensor,
        gradient: torch.Tensor,
        signed_cam: torch.Tensor,
    ) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        with torch.no_grad():
            for index in range(activation.shape[0]):
                a, g, c = activation[index], gradient[index], signed_cam[index]
                rows.append({
                    "activation_min": float(a.min().item()),
                    "activation_max": float(a.max().item()),
                    "activation_positive_fraction": float((a > 0).float().mean().item()),
                    "gradient_min": float(g.min().item()),
                    "gradient_max": float(g.max().item()),
                    "gradient_abs_mean": float(g.abs().mean().item()),
                    "gradient_positive_fraction": float((g > 0).float().mean().item()),
                    "gradient_negative_fraction": float((g < 0).float().mean().item()),
                    "signed_cam_min": float(c.min().item()),
                    "signed_cam_max": float(c.max().item()),
                    "positive_evidence_fraction": float((c > 0).float().mean().item()),
                    "negative_evidence_fraction": float((c < 0).float().mean().item()),
                })
        return rows

    # ------------------------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------------------------
    def _gradcam(
''', 'core:gradient diagnostics helper')

    s = replace_once(s,
'''        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation.detach()).sum(dim=1))
        return cam
''',
'''        weights = gradient.mean(dim=(2, 3), keepdim=True)
        signed_cam = (weights * activation.detach()).sum(dim=1)
        self.last_diagnostics = self._gradient_diagnostics(
            activation.detach(), gradient.detach(), signed_cam.detach()
        )
        # Keep standard positive-evidence Grad-CAM. Do not use abs(): that
        # would mislabel negative contribution as positive anomaly evidence.
        return F.relu(signed_cam)
''', 'core:gradcam signed diagnostics')

    s = replace_once(s,
'''        cam = F.relu(
            (channel_weights * activation.detach()).sum(dim=1)
        )
        return cam
''',
'''        signed_cam = (channel_weights * activation.detach()).sum(dim=1)
        self.last_diagnostics = self._gradient_diagnostics(
            activation.detach(), gradient.detach(), signed_cam.detach()
        )
        return F.relu(signed_cam)
''', 'core:gradcam++ signed diagnostics')

    s = replace_once(s,
'''        channel_weights = torch.softmax(normalized_delta, dim=1)

        cam = (
''',
'''        channel_weights = torch.softmax(normalized_delta, dim=1)
        # Softmax gives every finite channel a positive weight. Record its
        # entropy/effective channel count to quantify Score-CAM diffuseness.
        with torch.no_grad():
            probability = channel_weights.clamp_min(1e-12)
            entropy = -(probability * probability.log()).sum(dim=1)
            effective_channels = entropy.exp()
            self.last_diagnostics = [
                {
                    "score_delta_min": float(score_delta[i].min().item()),
                    "score_delta_max": float(score_delta[i].max().item()),
                    "score_delta_std": float(score_delta[i].std(unbiased=False).item()),
                    "scorecam_weight_entropy": float(entropy[i].item()),
                    "scorecam_effective_channels": float(effective_channels[i].item()),
                    "scorecam_total_channels": float(channels),
                }
                for i in range(batch_size)
            ]

        cam = (
''', 'core:scorecam entropy diagnostics')
    return s


def patch_runner(s: str) -> str:
    s = replace_once(s,
'''    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="推論裝置。",
    )
''',
'''    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="推論裝置。",
    )
    parser.add_argument(
        "--target-layer", default="encoder_conv.13",
        help=("CAM 目標層。預設使用最後卷積後的 ReLU 特徵 "
              "encoder_conv.13；指定 encoder_conv.11 可重現舊設定。"),
    )
''', 'runner:target-layer CLI')

    s = replace_once(s,
'''def score_samples(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    threshold: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    tensor = torch.from_numpy(X).to(device)
    model.eval()
    with torch.no_grad():
        scores = model.reconstruction_error(tensor).detach().cpu().numpy().astype(float)
    if not np.isfinite(scores).all():
        raise RuntimeError("CNN-VAE 異常分數含有 NaN 或 Inf。")

    details: List[Dict[str, Any]] = []
    for spec, score in zip(SAMPLE_MANIFEST, scores):
        actual_label = str(spec["expected_label"])
        # The actual label is replaced by CSV label by the caller afterwards.
        details.append({"score": float(score), "predicted_attack": bool(score > threshold)})
    return scores, details
''',
'''def score_samples(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    threshold: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    tensor = torch.from_numpy(X).to(device)
    model.eval()
    with torch.no_grad():
        deployed_scores = model.reconstruction_error(tensor).detach()
        recon, kl = model.loss_per_sample(tensor)
        cam_target_scores = (recon + float(model.kl_weight) * kl).detach()

    max_abs_diff = float((deployed_scores - cam_target_scores).abs().max().item())
    if max_abs_diff > 1e-7:
        raise RuntimeError(
            "正式 anomaly score 與 Grad-CAM 目標分數不一致："
            f"max_abs_diff={max_abs_diff:.3e}。請檢查 checkpoint、"
            "model class、kl_weight 與 eval mode。"
        )

    scores = deployed_scores.cpu().numpy().astype(np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError("CNN-VAE 異常分數含有 NaN 或 Inf。")

    details = [
        {
            "score": float(score),
            "predicted_attack": bool(score > threshold),
            "score_minus_threshold": float(score - threshold),
            "absolute_threshold_distance": float(abs(score - threshold)),
        }
        for score in scores
    ]
    return scores, details
''', 'runner:score parity and threshold details')

    s = replace_once(s,
'''def jaccard_top_k(left: np.ndarray, right: np.ndarray, k: int = 10) -> float:
    k = min(k, len(left), len(right))
    left_set = set(np.argsort(left)[-k:].tolist())
    right_set = set(np.argsort(right)[-k:].tolist())
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else float("nan")
''',
'''def has_informative_signal(values: np.ndarray, eps: float = 1e-8) -> bool:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        return False
    return bool(float(np.max(array) - np.min(array)) > eps)


def jaccard_top_k(left: np.ndarray, right: np.ndarray, k: int = 10) -> float:
    # Constant/all-zero CAMs have no meaningful Top-k set. Avoid the arbitrary
    # tie ordering returned by argsort from inflating method agreement.
    if not has_informative_signal(left) or not has_informative_signal(right):
        return float("nan")
    k = min(k, len(left), len(right))
    left_set = set(np.argsort(left)[-k:].tolist())
    right_set = set(np.argsort(right)[-k:].tolist())
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else float("nan")
''', 'runner:ignore constant CAM in Jaccard')

    old = '''            part = feature_df[
                (feature_df["variant"] == variant)
                & (feature_df["expected_group"] == group)
            ]
            if part.empty:
                continue
            means = part.groupby("feature_name")["heatmap_value"].mean()
            ranked = means.sort_values(ascending=False)
            group_sample_ids = set(part["sample_id"].unique())
            runtime_part = runtime_df[
                (runtime_df["variant"] == variant)
                & (runtime_df["sample_id"].isin(group_sample_ids))
            ]
            summary_rows.append({
                "variant": variant,
                "case_group": group,
                "sample_count": int(part["sample_id"].nunique()),
                "mean_feature_heatmap": float(part["heatmap_value"].mean()),
                "max_mean_feature_heatmap": float(ranked.iloc[0]),
                "top1_feature": str(ranked.index[0]),
                "top5_features": " | ".join(ranked.head(5).index.astype(str)),
                "mean_runtime_seconds_per_sample": float(runtime_part["runtime_seconds"].mean()),
            })
'''
    new = '''            # Group by the newly recalculated result, not the historic label.
            group_column = "observed_group" if "observed_group" in feature_df else "expected_group"
            part = feature_df[
                (feature_df["variant"] == variant)
                & (feature_df[group_column] == group)
            ]
            if part.empty:
                continue
            all_group_ids = set(part["sample_id"].unique())
            valid_group_ids = {
                sid for sid in all_group_ids
                if variant in sample_heatmaps.get(sid, {})
                and has_informative_signal(
                    sample_heatmaps[sid][variant].reshape(-1)[:len(feature_columns)]
                )
            }
            valid_part = part[part["sample_id"].isin(valid_group_ids)]
            invalid_count = len(all_group_ids - valid_group_ids)
            runtime_part = runtime_df[
                (runtime_df["variant"] == variant)
                & (runtime_df["sample_id"].isin(valid_group_ids))
            ]
            if valid_part.empty:
                summary_rows.append({
                    "variant": variant, "case_group": group,
                    "sample_count": len(all_group_ids),
                    "valid_sample_count": 0, "invalid_sample_count": invalid_count,
                    "mean_feature_heatmap": float("nan"),
                    "max_mean_feature_heatmap": float("nan"),
                    "top1_feature": "", "top5_features": "",
                    "mean_runtime_seconds_per_sample": float("nan"),
                })
                continue
            means = valid_part.groupby("feature_name")["heatmap_value"].mean()
            ranked = means.sort_values(ascending=False)
            summary_rows.append({
                "variant": variant, "case_group": group,
                "sample_count": len(all_group_ids),
                "valid_sample_count": len(valid_group_ids),
                "invalid_sample_count": invalid_count,
                "mean_feature_heatmap": float(valid_part["heatmap_value"].mean()),
                "max_mean_feature_heatmap": float(ranked.iloc[0]),
                "top1_feature": str(ranked.index[0]),
                "top5_features": " | ".join(ranked.head(5).index.astype(str)),
                "mean_runtime_seconds_per_sample": (
                    float(runtime_part["runtime_seconds"].mean())
                    if not runtime_part.empty else float("nan")
                ),
            })
'''
    s = replace_once(s, old, new, 'runner:observed group summary')

    s = replace_once(s,
'''            case_group = next(
                row["expected_group"] for row in feature_rows
                if row["sample_id"] == sample_id and row["variant"] == left_name
            )
''',
'''            case_group = next(
                row["observed_group"] for row in feature_rows
                if row["sample_id"] == sample_id and row["variant"] == left_name
            )
''', 'runner:comparison actual group')

    s = replace_once(s,
'''        for axis, variant in zip(axes[0], available_variants):
            subset = feature_df[feature_df["variant"] == variant]
            means = subset.groupby("feature_name")["heatmap_value"].mean().sort_values().tail(10)
            axis.barh(means.index.astype(str), means.values)
            axis.set_title(variant)
            axis.set_xlabel("Mean normalized heatmap value")
            axis.grid(axis="x", alpha=0.25)
''',
'''        for axis, variant in zip(axes[0], available_variants):
            valid_ids = [
                sid for sid, maps in sample_heatmaps.items()
                if variant in maps
                and has_informative_signal(
                    maps[variant].reshape(-1)[:len(feature_columns)]
                )
            ]
            subset = feature_df[
                (feature_df["variant"] == variant)
                & (feature_df["sample_id"].isin(valid_ids))
            ]
            axis.set_title(variant)
            axis.set_xlabel("Mean normalized heatmap value")
            axis.grid(axis="x", alpha=0.25)
            if subset.empty:
                axis.text(0.5, 0.5, "No informative CAM maps", ha="center", va="center",
                          transform=axis.transAxes)
            else:
                means = subset.groupby("feature_name")["heatmap_value"].mean().sort_values().tail(10)
                axis.barh(means.index.astype(str), means.values)
''', 'runner:exclude invalid maps from global top10')

    # The first occurrence is in run_feature_occlusion(); the second is in
    # scaled_feature_inputs. Patch each independently.
    row_fragment = '''                "expected_group": record["expected_group"],
                "feature_index": feature_index,
'''
    row_replacement = '''                "expected_group": record["expected_group"],
                "observed_group": record.get("observed_group", record["expected_group"]),
                "feature_index": feature_index,
'''
    row_count = s.count(row_fragment)
    if row_count != 2:
        raise RuntimeError(
            f"[runner:actual groups] 預期兩處 feature_index 欄位，實際 {row_count} 處；停止修改。"
        )
    s = s.replace(row_fragment, row_replacement)

    # Use explicit post-ReLU target and persist its diagnostic fields per run.
    s = replace_once(s,
'''            target_layer=bundle.gradcam_target_layer,
''',
'''            target_layer=args.target_layer,
''', 'runner:use explicit post-ReLU layer')

    s = replace_once(s,
'''                runtime_rows.append({
                    "variant": variant,
                    "sample_id": sample_id,
                    "expected_group": record["expected_group"],
                    "observed_group": record["observed_group"],
                    "runtime_seconds": float(elapsed),
''',
'''                diagnostic = (
                    cam.last_diagnostics[0]
                    if getattr(cam, "last_diagnostics", None) else {}
                )
                runtime_rows.append({
                    "variant": variant,
                    "sample_id": sample_id,
                    "expected_group": record["expected_group"],
                    "observed_group": record["observed_group"],
                    "anomaly_score": float(record["actual_score"]),
                    "threshold": float(model_threshold),
                    "score_minus_threshold": float(record["actual_score"] - model_threshold),
                    "target_layer": cam._resolved_layer,
                    **diagnostic,
                    "runtime_seconds": float(elapsed),
''', 'runner:export diagnostic and score fields')

    s = replace_once(s,
'''        "target_layer": "auto-detected last Conv2d in encoder",
        "resolved_target_layers": resolved_target_layers,
''',
'''        "target_layer_requested": args.target_layer,
        "resolved_target_layers": resolved_target_layers,
''', 'runner:metadata target layer')

    s = replace_once(s,
'''            "Each heatmap is normalized per sample to [0,1]; compare ranks/correlation rather than absolute intensity across samples.",
            "Score-CAM channel weights use standardized anomaly-score differences because VAE scores are small-valued.",
''',
'''            "Each heatmap is normalized per sample to [0,1]; compare ranks/correlation rather than absolute intensity across samples.",
            "Grad-CAM/Grad-CAM++ retain only positive evidence for increasing anomaly score; an all-zero map is not force-filled and must be interpreted with gradient_diagnostics.",
            "Score-CAM softmax gives all finite channels positive weights; scorecam_effective_channels quantifies weight diffuseness.",
            "Group summaries and method comparisons use observed_group recalculated from the CSV label and exact checkpoint score.",
''', 'runner:metadata interpretation notes')

    return s


def patch_vae_check(s: str) -> str:
    return replace_once(
        s,
        """    cam = GradCAM(
        model=model,
        target_layer=bundle.gradcam_target_layer,
        variant="gradcam",
        target_type="mse",
        device=device,
    )
    heatmaps = cam.generate(x, smooth=False)
""",
        """    cam = GradCAM(
        model=model,
        target_layer="encoder_conv.13",
        variant="gradcam",
        target_type="mse",
        score_mode="vae",
        device=device,
        smooth_sigma=0.0,
    )
    heatmaps = cam.generate(x, smooth=False)
""",
        "vae-check:exercise post-ReLU layer",
    ).replace(
        '    print("heatmap range:", float(heatmaps.min()), float(heatmaps.max()))\n',
        '    print("heatmap range:", float(heatmaps.min()), float(heatmaps.max()))\n'
        '    print("gradient diagnostics:", cam.last_diagnostics)\n',
        1,
    )


def main() -> int:
    patchers = {
        ROOT / "core/cnn_gradcam/gradcam_core.py": patch_gradcam_core,
        ROOT / "core/run_gradcam_cicids2017.py": patch_runner,
        ROOT / "core/run_gradcam_vae_check.py": patch_vae_check,
    }
    pending: dict[Path, str] = {}
    # Validate all replacements before writing anything.
    for path, patcher in patchers.items():
        if not path.is_file():
            raise FileNotFoundError(f"找不到專案檔案：{path}")
        pending[path] = patcher(path.read_text(encoding="utf-8"))

    backups = {
        path: path.with_suffix(path.suffix + ".pre_gradcam_fix.bak")
        for path in pending
    }
    existing = [backup for backup in backups.values() if backup.exists()]
    if existing:
        raise FileExistsError(
            "以下備份檔已存在；先檢查上次修改，不覆寫："
            + ", ".join(str(path) for path in existing)
        )

    for path, updated in pending.items():
        backup = backups[path]
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(updated, encoding="utf-8")
        print(f"PATCHED: {path.relative_to(ROOT)}")
        print(f"BACKUP : {backup.relative_to(ROOT)}")
    print("\n修正已套用。此腳本沒有重新執行模型或驗證真實資料輸出。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
