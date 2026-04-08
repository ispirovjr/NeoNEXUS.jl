# Workflow

NeoNEXUS has two broad stages: multi-scale signature extraction and hierarchical thresholding.

```@raw html
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

mermaid.initialize({
  startOnLoad: true,
  securityLevel: "loose",
});
</script>

<div class="mermaid">
flowchart LR
    subgraph Extraction["Multi-Scale Signature Extraction"]
        direction TB
        Scale["For each scale R"] --> Filter["Filter field"]
        Filter --> Hessian["Compute Hessian eigenvalues"]
        Hessian --> Feature["For each feature"]
        Feature --> Signature["Compute signature and update max map"]
        Signature -.-> Feature
        Feature -.-> Scale
    end

    Extraction ==> Cleaning

    subgraph Cleaning["Hierarchical Cleaning and Thresholding"]
        direction TB
        CleanFeature["For each feature"] --> Mask["Mask voxels claimed by earlier features"]
        Mask --> Threshold["Apply threshold rule"]
        Threshold --> Store["Store threshold map"]
        Store -.-> CleanFeature
    end
</div>
```

## Stage 1: Signature Extraction

At each smoothing scale, the field is filtered first and then passed through the Hessian pipeline.

- The Hessian eigenvalues are computed once per filtered field.
- Each requested feature converts those eigenvalues into a signature map.
- Feature objects keep the voxel-wise maximum across scales in `significanceMap`.

This stage is where cache reuse matters most. When several features are evaluated on the same filtered field, a shared [`HessianEigenCache`](@ref) avoids recomputing eigenvalues.

## Stage 2: Hierarchical Thresholding

After the scale loop, the accumulated signature maps are turned into binary `thresholdMap`s.

- Thresholding can be flat, mass-based, density-based, `deltaMSquaredThreshold!`, or connected-component based.
- Later features can be masked by already-thresholded earlier features with [`maskSignatureMap!`](@ref).
- Connected-component utilities operate on 6-connectivity.

## Runner-Specific Behavior

### `MMFClassic`

[`MMFClassic`](@ref) is the configurable runner.

- You provide the filter, the feature list, and the scales.
- The runner thresholds features in the order they are supplied.
- Each feature is masked by all earlier thresholded features.
- The built-in thresholding rule is [`componentErosionPlateauThreshold!`](@ref).

### `NEXUSPlus`

[`NEXUSPlus`](@ref) is the opinionated node -> filament -> wall pipeline.

- The input field is normalized to mean density 1 inside [`NeoNEXUS.run`](@ref).
- Nodes use the linear filter dispatch.
- Filaments and walls use the log-filtered dispatch of the selected filter.
- Nodes are thresholded with [`findComponentPercentageThreshold!`](@ref).
- Filaments and walls are thresholded with [`deltaMSquaredThreshold!`](@ref).

[`NeoNEXUS.runMultithreaded`](@ref) parallelizes the scale loop for `NEXUSPlus`, with one set of temporary arrays per scale.

## Stateful Objects

Features and runners are stateful.

- `significanceMap` stores the running max across calls.
- `thresholdMap` stores the latest thresholded classification.
- Recreate the feature or runner, or clear these arrays manually, before reusing an object for a new dataset.
