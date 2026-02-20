import numpy as np
from lab1 import query_key_value_attention as scaled_dot_product_attention


def test_output_shape():
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[1.0, 2.0], [3.0, 4.0]])

    output, weights = scaled_dot_product_attention(Q, K, V)

    assert output.shape == V.shape, f"Expected shape {V.shape}, got {output.shape}"
    assert weights.shape == (Q.shape[0], K.shape[0]), (
        f"Expected shape {(Q.shape[0], K.shape[0])}, got {weights.shape}"
    )


def test_attention_weights_sum_to_one():
    np.random.seed(42)
    Q = np.random.randn(3, 4)
    K = np.random.randn(3, 4)
    V = np.random.randn(3, 4)

    _, weights = scaled_dot_product_attention(Q, K, V)

    row_sums = weights.sum(axis=-1)
    assert np.allclose(row_sums, 1.0), f"Rows do not sum to 1: {row_sums}"


def test_numerical_example():
    Q = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    K = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0]])

    d_k = Q.shape[-1]
    raw_scores = Q @ K.T / np.sqrt(d_k)

    shifted = raw_scores - raw_scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    expected_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    expected_output = expected_weights @ V

    output, weights = scaled_dot_product_attention(Q, K, V)

    assert np.allclose(output, expected_output), (
        f"Output mismatch.\nExpected:\n{expected_output}\nGot:\n{output}"
    )
    assert np.allclose(weights, expected_weights), (
        f"Weights mismatch.\nExpected:\n{expected_weights}\nGot:\n{weights}"
    )


def test_identity_query_key():
    I = np.eye(3)
    V = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    output, weights = scaled_dot_product_attention(I, I, V)

    assert output.shape == V.shape
    assert np.allclose(weights.sum(axis=-1), 1.0)


if __name__ == "__main__":
    tests = [
        test_output_shape,
        test_attention_weights_sum_to_one,
        test_numerical_example,
        test_identity_query_key,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            print(f"PASSED: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {test.__name__} -> {e}")

    print(f"\n{passed}/{len(tests)} tests passed.")

    Q = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    K = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0]])

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("\n--- Example ---")
    print("Q:\n", Q)
    print("K:\n", K)
    print("V:\n", V)
    print("\nAttention Weights:\n", np.round(weights, 4))
    print("Output:\n", np.round(output, 4))
