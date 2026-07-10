"""Prompt/conditioning nodes.

Registered nodes:
  Prompt With Guidance (Flux)  Prompt with Flux Guidance MXD
  QwenImageEditSingleMXD       Qwen Image Edit + Latent MXD   (needs comfy_api)
  QwenImageEditTripleMXD       Qwen Image Edit Prompt MXD (Triple)  (needs comfy_api)
"""
from __future__ import annotations
import torch, comfy, math, node_helpers, comfy.model_management, comfy.utils
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
try:
    from comfy_api.latest import io
    HAVE_COMFY_API = True
except Exception as _e:
    io = None
    HAVE_COMFY_API = False
    print(f"[ComfyUI-MaxedOut] comfy_api not available in prompts: {_e}")

########################################################################################################################
# Prompt with Guidance (Flux)
class PromptWithGuidance(ComfyNodeABC):
    DESCRIPTION = """Encode text and apply Flux guidance in one node."""
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1})
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    FUNCTION = "encode_and_guide"
    CATEGORY = "MXD/conditioning"

    def encode_and_guide(self, text, clip, guidance):
        if clip is None:
            raise RuntimeError("CLIP model is None. Your checkpoint may not contain a text encoder.")

        tokens = clip.tokenize(text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return (conditioning,)

########################################################################################################################
if HAVE_COMFY_API:
    class QwenImageEditSingleMXD(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="QwenImageEditSingleMXD",
                display_name="Qwen Image Edit + Latent MXD",
                category="MXD/conditioning",
                description="Encode prompt/image and output a matching empty latent.",
                inputs=[
                    io.Clip.Input("clip"),
                    io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                    io.Vae.Input("vae", optional=True),
                    io.Image.Input("image", optional=True),
                    io.Int.Input("batch_size", default=1, min=1, max=4096),
                ],
                outputs=[
                    io.Conditioning.Output(),
                    io.Latent.Output(), # New Output
                ],
            )

        @classmethod
        def execute(cls, clip, prompt, vae=None, image=None, batch_size=1) -> io.NodeOutput:
            ref_latents = []
            images_vl = []
            llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            image_prompt = ""

            # Default fallback size if no image is provided (1024x1024)
            final_width, final_height = 1024, 1024

            if image is not None:
                samples = image.movedim(-1, 1)

                # --- VISION SCALING (384px area) ---
                total_vl = int(384 * 384)
                scale_vl = math.sqrt(total_vl / (samples.shape[3] * samples.shape[2]))
                width_vl = round(samples.shape[3] * scale_vl)
                height_vl = round(samples.shape[2] * scale_vl)

                s_vl = comfy.utils.common_upscale(samples, width_vl, height_vl, "area", "disabled")
                images_vl.append(s_vl.movedim(1, -1))

                # --- LATENT/VAE SCALING (1024px area) ---
                total_lat = int(1024 * 1024)
                scale_lat = math.sqrt(total_lat / (samples.shape[3] * samples.shape[2]))
                # Calculate final dimensions to be multiples of 8
                final_width = round(samples.shape[3] * scale_lat / 8.0) * 8
                final_height = round(samples.shape[2] * scale_lat / 8.0) * 8

                if vae is not None:
                    s_lat = comfy.utils.common_upscale(samples, final_width, final_height, "area", "disabled")
                    ref_latents.append(vae.encode(s_lat.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"

            # 1. Generate the Empty Latent (SD3 Style: 16 channels, 1/8th resolution)
            # This replaces the need for the separate EmptySD3LatentImage node
            latent_tensor = torch.zeros(
                [batch_size, 16, final_height // 8, final_width // 8],
                device=comfy.model_management.intermediate_device()
            )
            latent_output = {"samples": latent_tensor}

            # 2. Process Conditioning
            tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
            conditioning = clip.encode_from_tokens_scheduled(tokens)

            if len(ref_latents) > 0:
                conditioning = node_helpers.conditioning_set_values(
                    conditioning,
                    {"reference_latents": ref_latents},
                    append=True,
                )

            return io.NodeOutput(conditioning, latent_output)

    ########################################################################################################################
    class QwenImageEditTripleMXD(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="QwenImageEditTripleMXD",
                display_name="Qwen Image Edit Prompt MXD (Triple)",
                category="advanced/conditioning",
                inputs=[
                    io.Clip.Input("clip"),
                    io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                    io.Vae.Input("vae", optional=True),
                    io.Image.Input("image1", optional=True),
                    io.Image.Input("image2", optional=True),
                    io.Image.Input("image3", optional=True),
                    io.Int.Input("batch_size", default=1, min=1, max=4096),
                ],
                outputs=[
                    io.Conditioning.Output(),
                    io.Latent.Output(),
                ],
            )

        @classmethod
        def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None, batch_size=1) -> io.NodeOutput:
            ref_latents = []
            images = [image1, image2, image3]
            images_vl = []
            llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            image_prompt = ""

            # Default fallback
            latent_width = 1024
            latent_height = 1024

            for i, image in enumerate(images):
                if image is not None:
                    samples = image.movedim(-1, 1)

                    # 1. VL Model Scaling (LLM Vision)
                    total_vl = int(384 * 384)
                    scale_by_vl = math.sqrt(total_vl / (samples.shape[3] * samples.shape[2]))
                    width_vl = round(samples.shape[3] * scale_by_vl)
                    height_vl = round(samples.shape[2] * scale_by_vl)
                    s_vl = comfy.utils.common_upscale(samples, width_vl, height_vl, "area", "disabled")
                    images_vl.append(s_vl.movedim(1, -1))

                    # 2. VAE Scaling (Synchronized to 16-step for SD3 compatibility)
                    if vae is not None:
                        total_ref = int(1024 * 1024)
                        scale_by_ref = math.sqrt(total_ref / (samples.shape[3] * samples.shape[2]))

                        # Pixels as multiple of 16 ensures Latent (Pixels/8) is always even
                        width_ref = round(samples.shape[3] * scale_by_ref / 16.0) * 16
                        height_ref = round(samples.shape[2] * scale_by_ref / 16.0) * 16

                        if i == 0:
                            latent_width = width_ref
                            latent_height = height_ref

                        s_ref = comfy.utils.common_upscale(samples, width_ref, height_ref, "area", "disabled")
                        ref_latents.append(vae.encode(s_ref.movedim(1, -1)[:, :, :, :3]))

                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

            # Process tokens and conditioning
            tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
            conditioning = clip.encode_from_tokens_scheduled(tokens)

            if len(ref_latents) > 0:
                conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)

            # Create Output Latent
            latent = torch.zeros([batch_size, 16, latent_height // 8, latent_width // 8], device=comfy.model_management.intermediate_device())

            # FIXED: Return outputs positionally to match the schema defined above
            # Output 1: Conditioning, Output 2: Latent Dictionary
            return io.NodeOutput(conditioning, {"samples": latent})

########################################################################################################################

NODE_CLASS_MAPPINGS = {
    "Prompt With Guidance (Flux)": PromptWithGuidance,
}

if HAVE_COMFY_API:
    NODE_CLASS_MAPPINGS.update({
        "QwenImageEditSingleMXD": QwenImageEditSingleMXD,
        "QwenImageEditTripleMXD": QwenImageEditTripleMXD,
    })

NODE_DISPLAY_NAME_MAPPINGS = {
    "Prompt With Guidance (Flux)": "Prompt with Flux Guidance MXD",
}

if HAVE_COMFY_API:
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "QwenImageEditSingleMXD": "Qwen Image Edit + Latent MXD",
        "QwenImageEditTripleMXD": "Qwen Image Edit Prompt MXD (Triple)",
    })
