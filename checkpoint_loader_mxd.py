import folder_paths
import comfy.sd


class LoadCheckpointMXD:
    DESCRIPTION = (
        "Loads a diffusion model checkpoint, same as the core Load Checkpoint node, "
        "with the MXD info-icon UI (CivitAI lookup, cached metadata, local notes)."
    )
    TITLE = "Load Checkpoint MXD"
    CATEGORY = "MXD/Loaders"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return out[:3]


NODE_CLASS_MAPPINGS = {
    "LoadCheckpointMXD": LoadCheckpointMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadCheckpointMXD": "Load Checkpoint MXD",
}
