import numpy as np
import sys
# The PV calibration data is strange... I looked into the hl2ss/viewer examples as well as functions like
# hl2ss_3dcv.pv_fix_calibration to figure it out. 
def PVCalibrationToOpenCVFormat(hl2ss_calibration):
    fx, fy, = hl2ss_calibration.focal_length
    cx, cy = hl2ss_calibration.principal_point
    
    intrinsics_opencv = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,   1]
    ])
    
    R = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=hl2ss_calibration.extrinsics.dtype)

    extrinsics = hl2ss_calibration.extrinsics @ R
    extrinsics_opencv = extrinsics.T

    return intrinsics_opencv, extrinsics_opencv


def deep_getsizeof(o, seen=None):
    """Recursively finds the memory footprint of a Python object and its contents."""
    if seen is None:
        seen = set()
    obj_id = id(o)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(o)
    
    if isinstance(o, dict):
        return size + sum(deep_getsizeof(k, seen) + deep_getsizeof(v, seen) for k, v in o.items())
    elif isinstance(o, (list, tuple, set, frozenset)):
        return size + sum(deep_getsizeof(i, seen) for i in o)
    # For other objects, assume they don't contain further references
    return size



def bytes2human(n: int, decimals: int = 2) -> str:
    """
    Convert a byte count into a human-readable string (KiB, MiB, etc.).

    :param n: Number of bytes.
    :param decimals: Number of decimal places to include.
    :return: Human-readable string, e.g. "1.50 MiB".
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    suffixes = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB']
    idx = 0
    value = float(n)
    while value >= 1024 and idx < len(suffixes) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.{decimals}f} {suffixes[idx]}"


def memsize(o):
    return bytes2human(deep_getsizeof(o))

def printHl2ssCalibration(calibrationData):
    print('================Calibration================')
    print(f'Focal length: {calibrationData.focal_length}')
    print(f'Principal point: {calibrationData.principal_point}')
    print(f'Radial distortion: {calibrationData.radial_distortion}')
    print(f'Tangential distortion: {calibrationData.tangential_distortion}')
    print('\nProjection')
    print(calibrationData.projection)
    print('\nIntrinsics')
    print(calibrationData.intrinsics)
    print('\nRigNode Extrinsics')
    print(calibrationData.extrinsics)
    print(f'\nIntrinsics MF: {calibrationData.intrinsics_mf}')
    print(f'Extrinsics MF: {calibrationData.extrinsics_mf}')
    print("")