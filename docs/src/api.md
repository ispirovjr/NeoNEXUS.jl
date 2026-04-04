# API Reference

## Types

```@docs
AbstractScaleFilter
AbstractFeature
AbstractMorphologicalFeature
CacheMode
```

## Filters

```@docs
GaussianFourierFilter
TopHatFourierFilter
```

## Hessian

```@docs
HessianEigenCache
computeHessianEigenvalues
computeHessianEigenvalues!
```

## Features

```@docs
SheetFeature
LineFeature
NodeFeature
computeSignature
```

## Thresholds

```@docs
flatThreshold!
volumeThreshold!
massThreshold!
massCutoffThreshold!
thresholdedAverageDensity
averageDensityThreshold!
calculateΔM²
deltaMSquaredThreshold!
maskSignatureMap!
findComponentPercentageThreshold!
componentErosionPercentileThreshold!
componentErosionPlateauThreshold!
```

## Connected Components

```@docs
ConnectedComponent
findConnectedComponents
labelConnectedComponents
componentAverageDensity
componentDensityThreshold!
pruneSmallComponents!
pruneSmallMassComponents!
```

## Pipelines

```@docs
MMFClassic
NEXUSPlus
NeoNEXUS.run
runMultithreaded
```
