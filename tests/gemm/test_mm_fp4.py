import pytest
import torch
import torch.nn.functional as F
from flashinfer import (
    SfLayout,
    autotune,
    mm_fp4,
    nvfp4_quantize,
    mxfp4_quantize,
)
from flashinfer.utils import get_compute_capability, LibraryError
from flashinfer.gemm.gemm_base import CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR
from flashinfer.gemm.kernels.utils import (
    _SM100_MM_FP4_TACTIC_CACHE,
    _select_sm100_mm_fp4_cute_dsl_tactic,
)


def _test_mm_fp4(
    m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
):
    use_nvfp4 = fp4_type == "nvfp4"

    compute_capability = get_compute_capability(torch.device(device="cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if not mm_fp4.is_backend_supported(backend, compute_capability_number):
        pytest.skip(
            f"Skipping test for {backend} because it is not supported on compute capability {compute_capability_number}."
        )

    if backend == "trtllm":
        if res_dtype == torch.float16:
            pytest.skip("Skipping test for trtllm fp4 with float16")
        if compute_capability[0] in [11, 12]:
            pytest.skip("trtllm gemm does not support SM110/SM120/SM121 GPUs.")
    if backend == "cute-dsl":
        if not use_128x4_sf_layout:
            pytest.skip("cute_dsl backend only supports 128x4 SF layout")
        if compute_capability[0] not in [10]:
            pytest.skip("cute_dsl backend only supports SM100/SM103 GPUs.")
    if backend == "b12x":
        if not use_128x4_sf_layout:
            pytest.skip("b12x backend only supports 128x4 SF layout")
        if compute_capability[0] != 12:
            pytest.skip("b12x backend only supports SM120/SM121 GPUs.")
        if not use_nvfp4:
            pytest.skip("b12x backend only supports NVFP4 (sf_vec_size=16).")
        if torch.version.cuda and int(torch.version.cuda.split(".")[0]) < 13:
            pytest.skip("b12x backend requires CUDA 13+.")
    if not use_128x4_sf_layout and backend != "trtllm":
        pytest.skip("Skipping test for non-trtllm fp4 with use_128x4_sf_layout=False")
    if not use_nvfp4 and backend not in ["cudnn", "auto", "cute-dsl"]:
        pytest.skip("mx_fp4 is only supported for cudnn, cute-dsl, and auto backends")

    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_sf_layout = SfLayout.layout_128x4 if use_128x4_sf_layout else SfLayout.layout_8x4

    global_sf_input = (448 * 6) / input.float().abs().nan_to_num().max()
    global_sf_mat2 = (448 * 6) / mat2.float().abs().nan_to_num().max()

    # for trtllm, we need to shuffle mat2 because we swap A, B.
    do_shuffle_b = backend == "trtllm"

    block_size = 16 if use_nvfp4 else 32
    has_alpha = fp4_type == "mxfp4_alpha" or fp4_type == "nvfp4"

    if use_nvfp4:
        input_fp4, input_inv_s = nvfp4_quantize(
            input, global_sf_input, sfLayout=a_sf_layout, do_shuffle=False
        )
        mat2_fp4, mat2_inv_s = nvfp4_quantize(
            mat2,
            global_sf_mat2,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=do_shuffle_b,
        )
    else:
        input_fp4, input_inv_s = mxfp4_quantize(input)
        mat2_fp4, mat2_inv_s = mxfp4_quantize(mat2)

    alpha = 1.0 / (global_sf_input * global_sf_mat2) if has_alpha else None

    reference = torch.mm(input, mat2.T)

    res = torch.empty([m, n], device="cuda", dtype=res_dtype)

    try:
        with autotune(auto_tuning):
            mm_fp4(
                input_fp4,
                mat2_fp4.T,
                input_inv_s,
                mat2_inv_s.T,
                alpha,
                res_dtype,
                res,
                block_size=block_size,
                use_8x4_sf_layout=not use_128x4_sf_layout,
                backend=backend,
                use_nvfp4=use_nvfp4,
                skip_check=False,
            )

        cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
        assert cos_sim > 0.97
    except LibraryError as e:
        # TODO: Remove this check once cuDNN backend version is updated to 9.14.0
        if str(e) == CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR:
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))


# TODO: Consdier splitting this function up for the various backends
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 48, 64, 128, 256, 512])
@pytest.mark.parametrize("n", [128, 256, 512])
@pytest.mark.parametrize("k", [128, 256, 512])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["trtllm", "cudnn", "cutlass", "cute-dsl", "b12x"])
@pytest.mark.parametrize("use_128x4_sf_layout", [False, True])
@pytest.mark.parametrize("auto_tuning", [False, True])
@pytest.mark.parametrize("fp4_type", ["nvfp4", "mxfp4", "mxfp4_alpha"])
def test_mm_fp4(
    m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
):
    # Non-auto backends
    _test_mm_fp4(
        m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
    )


@pytest.mark.parametrize("m", [3, 7, 9, 15, 17, 31])
def test_sm100_mm_fp4_cute_dsl_heuristic_swaps_non_8_m_when_n_aligned(m):
    _SM100_MM_FP4_TACTIC_CACHE.clear()

    tactic = _select_sm100_mm_fp4_cute_dsl_tactic(
        m=m,
        n=128,
        real_k=256,
        sm_count=148,
    )

    assert tactic[2] is True
    assert tactic[0][1] <= 32
    assert tactic[0][1] >= m


def test_sm100_mm_fp4_cute_dsl_heuristic_rounds_m_bucket_up():
    _SM100_MM_FP4_TACTIC_CACHE.clear()

    tactic_m9 = _select_sm100_mm_fp4_cute_dsl_tactic(9, 128, 256, 148)
    tactic_m15 = _select_sm100_mm_fp4_cute_dsl_tactic(15, 128, 256, 148)
    tactic_m16 = _select_sm100_mm_fp4_cute_dsl_tactic(16, 128, 256, 148)

    cache = _SM100_MM_FP4_TACTIC_CACHE[(128, 256, 148)]
    assert set(cache) == {16}
    assert tactic_m9 == tactic_m15 == tactic_m16
    assert tactic_m9[2] is True
    assert tactic_m9[0][1] == 16


def test_sm100_mm_fp4_cute_dsl_heuristic_rejects_unaligned_n():
    _SM100_MM_FP4_TACTIC_CACHE.clear()

    with pytest.raises(ValueError, match="N.*multiple of 8"):
        _select_sm100_mm_fp4_cute_dsl_tactic(
            m=8,
            n=127,
            real_k=256,
            sm_count=148,
        )


def test_sm100_mm_fp4_cute_dsl_tactics_allow_swap_for_non_8_m_when_n_aligned():
    cutlass = pytest.importorskip("cutlass")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to enumerate SM100 CuTe-DSL tactics")

    from flashinfer.gemm.gemm_base import _get_sm100_block_scaled_tactics

    tactics = _get_sm100_block_scaled_tactics(
        m=7,
        n=128,
        real_k=256,
        ab_dtype=cutlass.Float4E2M1FN,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=16,
        c_cutlass_dtype=cutlass.BFloat16,
        device=torch.device("cuda"),
    )

    assert any(tactic[2] for tactic in tactics)


def test_sm100_mm_fp4_cute_dsl_tactics_reject_unaligned_n():
    cutlass = pytest.importorskip("cutlass")

    from flashinfer.gemm.gemm_base import _get_sm100_block_scaled_tactics

    tactics = _get_sm100_block_scaled_tactics(
        m=8,
        n=127,
        real_k=256,
        ab_dtype=cutlass.Float4E2M1FN,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=16,
        c_cutlass_dtype=cutlass.BFloat16,
        device=torch.device("cuda"),
    )

    assert tactics == []


@pytest.mark.parametrize("m", [7, 9, 17])
def test_mm_fp4_cute_dsl_nvfp4_heuristic_handles_non_8_m(m):
    _test_mm_fp4(m, 128, 256, torch.bfloat16, "cute-dsl", True, False, "nvfp4")


# Split tests for checking auto functionality
@pytest.mark.parametrize("m", [1, 48, 256, 512])
@pytest.mark.parametrize("n", [256, 512])
@pytest.mark.parametrize("k", [256, 512])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_128x4_sf_layout", [True])
@pytest.mark.parametrize("auto_tuning", [False, True])
@pytest.mark.parametrize("fp4_type", ["nvfp4", "mxfp4", "mxfp4_alpha"])
def test_mm_fp4_backend_auto(
    m, n, k, res_dtype, use_128x4_sf_layout, auto_tuning, fp4_type
):
    # Some test cases for auto backend.
    _test_mm_fp4(m, n, k, res_dtype, "auto", use_128x4_sf_layout, auto_tuning, fp4_type)


if __name__ == "__main__":
    pytest.main([__file__])
