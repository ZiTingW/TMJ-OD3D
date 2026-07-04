# TMJ CBCT Dataset Usage Example

This folder contains two public scripts:

- `tmj_dataset_read_example.py`: reads one sample and prints DICOM/CBCT plus annotation-box information.
- `tmj_dataset_viewer.py`: exports PNG visualizations of the same labels.

Run examples from an environment with `pydicom`, `numpy`, `Pillow`, and `matplotlib`. Locally this is available through:

```powershell
conda run -n base python ...
```

## Read One Sample

```powershell
conda run -n base python tmj_dataset_read_example.py `
  --dataset-root E:\tmj `
  --sample TMJ00622 `
  --side all `
  --output-json TMJ00622_summary.json
```

The script prints:

- CBCT DICOM slice count and `InstanceNumber` range.
- Raw DICOM pixel array shape, dtype, min, and max for an example slice.
- One row per annotation box with `side`, `class_id`, `class_name`, `slice_start`, `slice_end`, `bbox_xyxy`, and reference DICOM slice.

The optional JSON output contains the same fields plus raw pixel summaries for each annotation reference slice.

## Annotation Format

Each 3D bounding box is represented as:

```text
3D box = 2D rectangle in pixel coordinates + axial slice_start/slice_end
```

The JSON label format is:

```text
<class_id(s)>-<side>-<slice_start>-<slice_end>
```

For example:

- `4-L-295-325`: left TMJ, condylar flattening, visible from DICOM `InstanceNumber` 295 to 325.
- `1,5-R-240-270`: right TMJ, cortical erosion plus osteophyte, visible from 240 to 270.

The class names are:

| ID | Class name |
| --- | --- |
| 0 | normal |
| 1 | cortical erosion |
| 2 | subchondral sclerosis |
| 3 | subchondral cystic changes |
| 4 | condylar flattening |
| 5 | osteophyte |
| 6 | other |

## Access Raw CBCT Data In Python

```python
from pathlib import Path
import pydicom
import numpy as np

sample_dir = Path(r"E:\tmj\TMJ00622")
dicom_files = sorted(
    sample_dir.glob("*.dcm"),
    key=lambda p: int(pydicom.dcmread(str(p), stop_before_pixels=True).InstanceNumber),
)
volume = np.stack([pydicom.dcmread(str(p)).pixel_array for p in dicom_files], axis=0)
print(volume.shape, volume.dtype)
```

Use DICOM `InstanceNumber` to match annotation `slice_start` and `slice_end`.

## Visualize Labels

Reference layer visualization, using the DICOM slice named by each JSON `image_path`:

```powershell
conda run -n base python tmj_dataset_viewer.py `
  --dataset-root E:\tmj `
  --sample TMJ00622 `
  --mode reference `
  --side all `
  --output-dir tmj_preview
```

Specific DICOM layer visualization, drawing boxes whose 3D slice range contains the requested `InstanceNumber`:

```powershell
conda run -n base python tmj_dataset_viewer.py `
  --dataset-root E:\tmj `
  --sample TMJ00622 `
  --mode slice `
  --slice 315 `
  --side all `
  --output-dir tmj_preview
```
