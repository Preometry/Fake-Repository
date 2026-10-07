#!/usr/bin/env python3
"""Independent NumPy checks for d(x,y)=acos(clamp(x·y,-1,1)) on S^19."""
import math
import numpy as np


def distance(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    dot = np.sum(x * y, axis=-1)
    # Match the recovered attention's guard band, which avoids infinite acos
    # derivatives at the endpoints but means even identical vectors have a
    # small positive distance.
    return np.arccos(np.clip(dot, -1.0 + 1e-7, 1.0 - 1e-7))


def main():
    rng = np.random.default_rng(20261004)
    x = rng.normal(size=(128, 20))
    y = rng.normal(size=(128, 20))
    dx = distance(x, x)
    dxy, dyx = distance(x, y), distance(y, x)
    endpoint_floor = math.acos(1.0 - 1e-7)
    assert np.max(np.abs(dx-endpoint_floor)) < 4e-8
    assert np.max(np.abs(dxy-dyx)) < 2e-8
    assert np.all((dxy >= 0) & (dxy <= math.pi))
    assert abs(float(distance([1, 0], [1, 0])) - endpoint_floor) < 1e-12
    assert abs(float(distance([1, 0], [-1, 0])) - (math.pi-endpoint_floor)) < 1e-12
    # A content perturbation changes geodesic distance for a generic point pair.
    a = np.array([1., 0., 0.]); b = np.array([0., 1., 0.]); c = np.array([1., 1., 0.1])
    assert abs(float(distance(a, b)) - float(distance(a, c))) > 0.1
    # Verify the recovered attention transform: exp(-d^2/T), normalized by softmax.
    points = np.array([[1., 0.], [0., 1.], [1., 1.]])
    ds = np.array([distance(points[0], q) for q in points])
    logits = -(ds ** 2) / 1.0
    weights = np.exp(logits-logits.max())
    weights /= weights.sum()
    assert np.allclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]  # identical direction is closer than orthogonal
    # On a unit sphere, arccos(dot) preserves cosine nearest-neighbor ranking,
    # while the squared-distance softmax changes the relative weight scale.
    cosines = np.array([0.9, 0.2, -0.4])
    angles = np.arccos(np.clip(cosines, -1.0+1e-7, 1.0-1e-7))
    cosine_logits = cosines / math.sqrt(20.0)
    geo_logits = -(angles ** 2)
    assert np.array_equal(np.argsort(cosines), np.argsort(-angles))
    cweights = np.exp(cosine_logits-cosine_logits.max()); cweights /= cweights.sum()
    gweights = np.exp(geo_logits-geo_logits.max()); gweights /= gweights.sum()
    assert not np.allclose(cweights, gweights)
    print("PASS: metric identities, guarded endpoints, content sensitivity, geodesic attention transform")
    print("NOTE: cosine and geodesic scores share nearest-neighbor ranking on normalized vectors; attention weights differ.")


if __name__ == "__main__":
    main()
