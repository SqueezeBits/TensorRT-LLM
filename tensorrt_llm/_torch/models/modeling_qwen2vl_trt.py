import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from tensorrt_llm.functional import RopeEmbeddingUtils, RotaryScalingType
from tensorrt_llm.inputs import (ExtraProcessedInputs, InputProcessor, TextPrompt)
from tensorrt_llm.inputs.registry import INPUT_PROCESSOR_REGISTRY

from tensorrt_llm.sampling_params import SamplingParams



class PatchedQwen2VLInputProcessorBase(InputProcessor):

    def __init__(self, model_path, model_config: PretrainedConfig, tokenizer: AutoTokenizer):
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.device = 'cpu'
        self.tokenizer = tokenizer
        self.model_config = AutoConfig.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

        self._post_init_()


    @classmethod
    def get_model_class(cls) -> type[PreTrainedModel]:
        raise NotImplementedError()

    @classmethod
    def get_rope_index(
        cls,
        model_config: PretrainedConfig,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        This is a generalized implementation that can be used by both Qwen2VL and Qwen2_5_VL models.
        The main difference between the two implementations is how temporal position IDs are calculated.

        Args:
            model_config: The model configuration
            input_ids: Indices of input sequence tokens in the vocabulary
            image_grid_thw: The temporal, height and width of feature shape of each image in LLM
            video_grid_thw: The temporal, height and width of feature shape of each video in LLM
            attention_mask: Mask to avoid performing attention on padding token indices
            second_per_grid_ts: The time interval (in seconds) for each grid along the temporal dimension

        Returns:
            position_ids: A tensor of shape (3, batch_size, sequence_length)
            mrope_position_deltas: A tensor of shape (batch_size)
        """
        spatial_merge_size = model_config.vision_config.spatial_merge_size
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        mrope_position_deltas = []

        # Handle case with no vision inputs
        if image_grid_thw is None and video_grid_thw is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(
                    input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[
                    -1]
            else:
                position_ids = (torch.arange(input_ids.shape[1],
                                             device=input_ids.device).view(
                                                 1, 1, -1).expand(
                                                     3, input_ids.shape[0], -1))
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

        # Handle case with vision inputs
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                    llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # Calculate temporal position IDs based on model type
                if hasattr(model_config.vision_config, 'tokens_per_second'):
                    # Qwen2_5_VL style temporal position calculation
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(
                        -1, llm_grid_h * llm_grid_w)
                    time_tensor = expanded_range * second_per_grid_t * model_config.vision_config.tokens_per_second
                    t_index = time_tensor.long().flatten()
                else:
                    # Qwen2VL style temporal position calculation
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                        -1, llm_grid_h * llm_grid_w).flatten()

                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                    llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                    llm_grid_t, llm_grid_h, -1).flatten()

                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len +
                    st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                    llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 -
                                         len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def _post_init_(self):
        _, rotary_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
            num_pos=self.model_config.max_position_embeddings,
            dim=int(self.model_config.hidden_size /
                    self.model_config.num_attention_heads),
            theta=float(self.model_config.rope_theta),
            scale_type=RotaryScalingType.mrope)
        self.rotary_cos_sin = torch.from_numpy(rotary_cos_sin).to(self.device)
        self.rotary_cos_sin = self.rotary_cos_sin.reshape(
            self.model_config.max_position_embeddings,
            int(self.model_config.hidden_size /
                self.model_config.num_attention_heads / 2), 2)

        self.cos_ori = self.rotary_cos_sin[:, :, 0]
        self.sin_ori = self.rotary_cos_sin[:, :, 1]

    def _preprocess(self, text: dict[str, Any], mm_data: dict[str, Any],
                    mm_processor_kwargs: Dict[str, Any]):
        images = mm_data.get("image", None)
        inputs = self.processor(text=[text],
                            images=images,
                            videos=mm_data.get("video", None),
                            padding=True,
                            return_tensors='pt',
                            **mm_processor_kwargs)
        inputs["pixel_values"] = torch.stack(images, dim=0).to(torch.bfloat16)
        return inputs


    def get_mrope_config(
            self,
            input_ids: torch.LongTensor,
            image_grid_thw: torch.LongTensor,
            video_grid_thw: torch.LongTensor,
            attention_mask: torch.Tensor,
            second_per_grid_ts: torch.Tensor = None) -> dict[str, torch.Tensor]:
        mrope_position_ids, mrope_position_deltas = self.__class__.get_rope_index(
            self.model_config, input_ids, image_grid_thw, video_grid_thw,
            attention_mask, second_per_grid_ts)

        mrope_position_ids = mrope_position_ids.transpose(1, 0)
        mrope_position_ids_padding = torch.zeros(
            mrope_position_ids.shape[:-1] +
            (self.model_config.max_position_embeddings, ),
            dtype=torch.int32,
            device=input_ids.device)
        mrope_position_ids_padding[:, :, :mrope_position_ids.
                                   shape[-1]] = mrope_position_ids
        cos = self.cos_ori[mrope_position_ids_padding]
        sin = self.sin_ori[mrope_position_ids_padding]

        mrope_section = [16, 24, 24]
        cos = torch.cat([
            m[:, i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))
        ],
                        dim=-1).unsqueeze(-1)
        sin = torch.cat([
            m[:, i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))
        ],
                        dim=-1).unsqueeze(-1)
        concat_cos_sin = torch.concatenate((cos, sin), axis=-1)
        concat_cos_sin = concat_cos_sin.reshape(concat_cos_sin.shape[0], -1)
        mrope_config = {}
        mrope_config['mrope_rotary_cos_sin'] = concat_cos_sin[0]
        mrope_config['mrope_position_deltas'] = mrope_position_deltas[0]
        return mrope_config

    @torch.inference_mode()
    def __call__(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data, mm_processor_kwargs = inputs.get("prompt"), \
                        inputs.get("multi_modal_data"), inputs.get("mm_processor_kwargs", {})

        # NOTE: Since we are passed in Tensor images, we don't need to rescale them.
        mm_processor_kwargs['do_rescale'] = False
        processed_inputs = self._preprocess(text_prompt, mm_data,
                                            mm_processor_kwargs)
        
        mm_features = processed_inputs.get('pixel_values', None).to(torch.float16)

        input_ids = processed_inputs['input_ids']

        mrope_config = self.get_mrope_config(
            input_ids, processed_inputs.get('image_grid_thw', None),
            processed_inputs.get('video_grid_thw', None),
            processed_inputs.get('attention_mask', None),
            processed_inputs.get('second_per_grid_ts', None))

        return input_ids[0].to(torch.int32).tolist(), {
            "prompt_tuning_config": [mm_features.contiguous(), None, None],
            "mrope_config": mrope_config
        }



INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type["DITTO"] = PatchedQwen2VLInputProcessorBase