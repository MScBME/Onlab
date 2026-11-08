import numpy as np
import pytest

from swimmertracker.detection.onnx_detector import OnnxDetector, decode, letterbox, to_image_coords, to_tensor
from swimmertracker.detection.types import Detection, highest_confidence


def classic_output(rows, num_classes=1, anchors=50):
    """Build a (1, 4 + classes, anchors) output from (cx, cy, w, h, class, score) rows. Other anchors score 0."""
    out = np.zeros((1, 4 + num_classes, anchors), np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(rows):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    return out


def test_letterbox_is_a_no_op_for_the_exact_input_size():
    image = np.random.default_rng(0).integers(0, 255, (1024, 256, 3), dtype=np.uint8)
    padded, gain, pad = letterbox(image, (1024, 256))
    assert gain == 1.0 and pad == (0, 0)
    assert np.array_equal(padded, image)


def test_letterbox_pads_with_gray_and_maps_boxes_back():
    image = np.zeros((100, 200, 3), np.uint8)
    padded, gain, pad = letterbox(image, (64, 64))
    assert padded.shape == (64, 64, 3) and gain == pytest.approx(0.32)
    assert pad == (0, 16)
    assert tuple(padded[0, 0]) == (114, 114, 114)
    rows = np.array([[0, 16, 64, 48, 0.9, 0]], np.float32)
    back = to_image_coords(rows, gain, pad, image.shape)
    assert back[0, :4] == pytest.approx([0, 0, 200, 100])


def test_to_tensor_is_rgb_nchw_unit_range():
    image = np.zeros((2, 3, 3), np.uint8)
    image[..., 0] = 255  # blue in BGR
    tensor = to_tensor(image)
    assert tensor.shape == (1, 3, 2, 3) and tensor.dtype == np.float32
    assert tensor[0, 2].max() == 1.0 and tensor[0, 0].max() == 0.0


def test_decode_classic_thresholds_suppresses_and_sorts():
    out = classic_output([
        (100, 100, 40, 40, 0, 0.80),
        (102, 101, 40, 40, 0, 0.85),  # overlaps the first one and suppresses it
        (300, 500, 30, 60, 0, 0.60),
        (50, 900, 20, 20, 0, 0.30),   # under the threshold
    ])
    rows = decode(out, confidence=0.5)
    assert rows[:, 4] == pytest.approx([0.85, 0.60])
    assert rows[0, :4] == pytest.approx([82, 81, 122, 121])


def test_decode_score_must_be_strictly_above_threshold():
    rows = decode(classic_output([(100, 100, 40, 40, 0, 0.5)]), confidence=0.5)
    assert len(rows) == 0


def test_decode_nms_is_per_class():
    out = classic_output([(100, 100, 40, 40, 0, 0.9), (101, 100, 40, 40, 1, 0.8)], num_classes=3)
    rows = decode(out, confidence=0.5)
    assert len(rows) == 2 and sorted(rows[:, 5]) == [0, 1]


def test_decode_end_to_end_output():
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = (10, 20, 50, 80, 0.9, 0)
    out[0, 1] = (100, 200, 150, 260, 0.4, 0)
    rows = decode(out, confidence=0.5)
    assert rows.tolist() == [pytest.approx([10, 20, 50, 80, 0.9, 0])]


def test_highest_confidence():
    a, b = Detection(0, 0, 1, 1, 0.5), Detection(0, 0, 2, 2, 0.7)
    assert highest_confidence([a, b]) is b
    assert highest_confidence([]) is None
    assert b.center == (1.0, 1.0)


def test_detector_runs_an_onnx_model(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    constant = classic_output([(128, 512, 40, 60, 0, 0.9), (128, 100, 40, 60, 0, 0.3)], anchors=20)
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),  # use the input so it is kept
            helper.make_node("Mul", ["mean", "zero"], ["nothing"]),
            helper.make_node("Add", ["boxes", "nothing"], ["output0"]),
        ],
        "constant_detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 1024, 256])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(constant.shape))],
        [numpy_helper.from_array(constant, "boxes"), numpy_helper.from_array(np.array(0, np.float32), "zero")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    path = tmp_path / "model.onnx"
    onnx.save(model, path)

    detector = OnnxDetector(path, confidence=0.5)
    assert detector.input_size == (1024, 256)
    detections = detector.detect(np.zeros((1024, 256, 3), np.uint8))
    assert len(detections) == 1
    assert detections[0].bbox == pytest.approx((108, 482, 148, 542))
    assert detections[0].confidence == pytest.approx(0.9)
