"""Task 3 RED/GREEN: decode_image — RGB order, EXIF-8, committed expected."""

import pytest

from facecore.conformance.fixtures import make_exif_case, upright_expected
from facecore.errors import InputDecodeError
from facecore.pipeline.decode import decode_image


def test_exif_orientation_6_decodes_upright_not_sideways() -> None:
    """Failing case from the plan: orientation-6 must not decode sideways."""
    data = make_exif_case(6)
    decoded = decode_image(data)
    assert decoded.pixels == upright_expected()


def test_all_eight_orientations_produce_same_upright_array() -> None:
    expected = upright_expected()
    for orientation in range(1, 9):
        assert decode_image(make_exif_case(orientation)).pixels == expected


def test_color_order_is_rgb_against_committed_expected() -> None:
    decoded = decode_image(make_exif_case(1))
    assert decoded.color_order == "RGB"
    assert decoded.pixels == upright_expected()


def test_undecodable_bytes_raise_exit_2_never_invalid_input() -> None:
    with pytest.raises(InputDecodeError) as exc_info:
        decode_image(b"not an image at all")
    assert type(exc_info.value).exit_code == 2
