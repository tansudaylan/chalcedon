# Chalcedon

## Scientific purpose
Chalcedon is a gravitational-lensing simulation and modeling library. It provides the tooling needed to create and inspect lensing geometries, source configurations, and derived image-plane diagnostics.

## Scope
The repository is focused on lensing calculations and their visualization rather than on broader astronomical utility code. It fits in the ecosystem as a domain-specific simulation layer used by imaging and inference workflows that require physically interpretable lensing diagnostics.

## Installation

```bash
cd /path/to/chalcedon
python -m pip install -e .
```

## Minimal usage

```python
import chalcedon
# Use the package-level lensing functions and simulation entry points.
```

## Output and diagnostics
A useful run should produce the input lensing configuration, relevant intermediate geometric quantities, and the final simulated or diagnostic image output in a reproducible and inspectable form.

## Development status
Chalcedon is maintained as a focused gravitational-lensing simulation library. It is not a generic astronomy toolbox and should remain structured around its specific lensing workflows and visual diagnostics.
