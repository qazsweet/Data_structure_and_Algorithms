## Chessboard corner detection (saddle points + subpixel)

This repo contains a small script (`detect_corners.py`) for detecting chessboard corners using:

- **Saddle-point candidates** from the Hessian determinant response (custom implementation)
- Optional **subpixel refinement**
- Optional OpenCV **saddle-based** chessboard detector (`findChessboardCornersSB`) when the pattern size is known

### Install

```bash
python3 -m pip install -r requirements.txt
```

### Usage

- **Saddle candidates (no pattern size required)**:

```bash
python3 detect_corners.py /path/to/image.png --method saddle
```

- **Cross template center (subpixel)**:
  - If you already have a cross template image:

```bash
python3 detect_cross_template.py /path/to/image.png --template /path/to/cross_template.png
```

  - Or using a synthetic “+” template (tune arm/thickness/scales as needed):

```bash
python3 detect_cross_template.py /path/to/image.png --arm-len 18 --thickness 5 --scales 0.75,1.0,1.25
```

- **OpenCV saddle-based ordered grid (pattern size required; returns subpixel corners)**:

```bash
python3 detect_corners.py /path/to/image.png --method sb --rows 31 --cols 31
```

- **Classic OpenCV + `cornerSubPix` (pattern size required)**:

```bash
python3 detect_corners.py /path/to/image.png --method classic --rows 31 --cols 31
```

Outputs are written as `corners_<method>_<image-stem>.png` unless overridden with `--out`.
