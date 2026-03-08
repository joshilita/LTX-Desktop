"""LTX Pro (non-distilled) video pipeline wrapper."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import ClassVar, Literal, cast

import torch

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8


# Default guider params from LTX-2.3 non-distilled pipeline constants.
_DEFAULT_NUM_INFERENCE_STEPS = 30
_DEFAULT_CFG_SCALE = 3.0
_DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
    "proportions, unnatural skin tones, deformed facial features, asymmetrical face, "
    "missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts "
    "around text, inconsistent perspective, camera shake, incorrect depth of field"
)


class LTXProVideoPipeline:
    pipeline_kind: ClassVar[Literal["fast", "pro"]] = "pro"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
    ) -> "LTXProVideoPipeline":
        return LTXProVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
    ) -> None:
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

        self.num_inference_steps = _DEFAULT_NUM_INFERENCE_STEPS
        self.cfg_scale = _DEFAULT_CFG_SCALE
        self.negative_prompt = _DEFAULT_NEGATIVE_PROMPT

        self.pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=[],
            spatial_upsampler_path=upsampler_path,
            gemma_root=cast(str, gemma_root),
            loras=[],
            device=device,
            quantization=QuantizationPolicy.fp8_cast() if device_supports_fp8(device) else None,
        )

        self._device = device
        self._video_guider_params = MultiModalGuiderParams(
            cfg_scale=3.0,
            stg_scale=1.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[28],
        )
        self._audio_guider_params = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=1.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[28],
        )

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

        video_guider_params = MultiModalGuiderParams(
            cfg_scale=self.cfg_scale,
            stg_scale=self._video_guider_params.stg_scale,
            rescale_scale=self._video_guider_params.rescale_scale,
            modality_scale=self._video_guider_params.modality_scale,
            skip_step=self._video_guider_params.skip_step,
            stg_blocks=self._video_guider_params.stg_blocks,
        )

        return self.pipeline(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=self.num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=self._audio_guider_params,
            images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
            tiling_config=tiling_config,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        tiling_config = default_tiling_config()
        video, audio = self._run_inference(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        warmup_frames = 9
        tiling_config = default_tiling_config()

        try:
            video, audio = self._run_inference(
                prompt="test warmup",
                seed=42,
                height=256,
                width=384,
                num_frames=warmup_frames,
                frame_rate=8,
                images=[],
                tiling_config=tiling_config,
            )
            chunks = video_chunks_number(warmup_frames, tiling_config)
            encode_video_output(video=video, audio=audio, fps=8, output_path=output_path, video_chunks_number_value=chunks)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        transformer = self.pipeline.stage_1_model_ledger.transformer()

        compiled = cast(
            torch.nn.Module,
            torch.compile(transformer, mode="reduce-overhead", fullgraph=False),  # type: ignore[reportUnknownMemberType]
        )
        setattr(self.pipeline.stage_1_model_ledger, "transformer", lambda: compiled)
