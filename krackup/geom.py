"""Small transform and rotation helpers. Meshes stay in millimeters."""

import numpy as np


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    if length < 1e-12:
        raise ValueError("zero vector")
    return vector / length


def extents(bounds):
    return np.array(
        [bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2]],
        dtype=np.float64,
    )


def fits(ext, limit):
    return (
        ext[0] <= limit[0] + 0.4
        and ext[1] <= limit[1] + 0.4
        and ext[2] <= limit[2] + 0.4
    )


def rot_from_to(source, target):
    source = unit(source)
    target = unit(target)
    cross = np.cross(source, target)
    cosine = float(np.dot(source, target))
    if cosine > 1.0 - 1e-10:
        return np.eye(3)
    if cosine < -1.0 + 1e-10:
        helper = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = unit(np.cross(source, helper))
        skew = _skew(axis)
        return np.eye(3) + 2.0 * (skew @ skew)
    skew = _skew(cross)
    scale = (1.0 - cosine) / float(np.dot(cross, cross))
    return np.eye(3) + skew + (skew @ skew) * scale


def plane_frame(normal):
    """Rotation whose third row is the unit normal, so z' = n · p."""
    normal = unit(normal)
    helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x_axis = unit(np.cross(helper, normal))
    y_axis = np.cross(normal, x_axis)
    frame = np.stack([x_axis, y_axis, normal], axis=0)
    if np.linalg.det(frame) < 0:
        frame[0] *= -1
    return frame


def apply_frame(manifold, frame):
    matrix = np.zeros((3, 4), dtype=np.float64)
    matrix[:, :3] = frame
    return manifold.transform(matrix)


def _skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def pca(points):
    center = points.mean(axis=0)
    shifted = points - center
    covariance = (shifted.T @ shifted) / max(len(points) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    projected = shifted @ axes
    ext = projected.max(axis=0) - projected.min(axis=0)
    return center, axes, ext
