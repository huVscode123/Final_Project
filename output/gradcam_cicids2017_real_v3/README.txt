Grad-CAM CICIDS2017 outputs
- selected_samples.csv: selected sample, score, threshold, prediction and image names
- feature_heatmap_values.csv: CAM pixel values sampled at the raster positions of the 78 features;
  this is NOT a valid standalone feature-importance ranking. The final-layer CAM is upsampled,
  and convolutional receptive fields may spread activation into padded locations.
- feature_occlusion_sensitivity.csv: one feature at a time is set to 0.0 after scaling/clipping;
  score_delta = original score - perturbed score. Larger absolute deltas indicate greater
  local sensitivity for this sample, not causality or global feature importance.
- source_row_index_zero_based is local to each source CSV; combined_row_index_zero_based
  is the row index after input CSVs have been concatenated. Row index excludes the header.
- *_input.png, *_gradcam.png, *_overlay.png: feature-image input, CAM, overlay
- The 32x32 image is a row-major raster of CSV flow features padded to 1024 values; it is not raw packet bytes.
- The CAM is normalized per sample; magnitudes should not be compared across samples.
