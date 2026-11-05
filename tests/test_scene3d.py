import numpy as np
import pytest

from quantum_ray_tracing.grover import AerGroverBackend
from quantum_ray_tracing.scene3d import (
    Rect,
    Scene,
    intersect_all,
    nrmse,
    qornell_box,
    render_classical,
    render_quantum,
)


def test_ray_rect_intersection():
    s = Scene(rects=[Rect(2, 5.0, (0, 0), (2, 2)), Rect(0, 1.0, (0, 0), (1, 1))], light=np.zeros(3))
    o = np.array([[1.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.5, 0.5]])
    d = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    t = intersect_all(s, o, d)
    assert t[0, 0] == pytest.approx(5.0)
    assert np.isinf(t[1, 0])  # misses the rectangle bounds
    assert t[2, 1] == pytest.approx(1.0)
    assert np.isinf(t[0, 1])


def test_dummies_do_not_change_image():
    a, _ = render_classical(qornell_box(16), 24, 24)
    b, st = render_classical(qornell_box(128), 24, 24)
    assert np.array_equal(a, b)
    assert st.int_per_ray == 128


def test_quantum_render_close_to_classical_and_cheaper():
    scene = qornell_box(256)
    ref, cst = render_classical(scene, 32, 32)
    img, qst = render_quantum(scene, 32, 32, seed=0)
    assert nrmse(img, ref) < 0.05
    assert qst.int_per_ray < cst.int_per_ray / 1.5


@pytest.mark.slow
def test_render_with_real_circuits():
    scene = qornell_box(16)
    ref, _ = render_classical(scene, 10, 10)
    img, _ = render_quantum(scene, 10, 10, backend=AerGroverBackend(seed=2), seed=2)
    assert nrmse(img, ref) < 0.1
