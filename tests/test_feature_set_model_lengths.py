import pytest


@pytest.mark.parametrize(
    ("length", "classes", "params", "flops", "w8_bytes"),
    [(8, 12, 38328, 133056, 46284), (93, 2, 35438, 3358656, 43324)],
)
def test_model_length_check_counts_grouped_convolutions_and_pooled_head(length, classes, params, flops, w8_bytes):
    from scripts.diagnostics.model_lengths import check_length

    result = check_length(length, classes)

    assert result["output_shape"] == [1, classes]
    assert result["trainable_parameters"] == params
    assert result["conv_and_linear_flops"] == flops
    assert result["w8_model_array_bytes"] == w8_bytes
