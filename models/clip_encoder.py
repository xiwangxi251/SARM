import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from typing import List
from PIL import Image

class FrozenCLIPEncoder(nn.Module):
    def __init__(self, ckpt: str, device: torch.device):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(ckpt).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(ckpt)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _as_feature_tensor(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "image_embeds"):
            return output.image_embeds
        if hasattr(output, "text_embeds"):
            return output.text_embeds
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0]
        if isinstance(output, (tuple, list)) and output:
            return FrozenCLIPEncoder._as_feature_tensor(output[0])
        raise TypeError(f"Unsupported CLIP output type: {type(output).__name__}")

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        texts: list[str], length B
        returns: (B, 512) CLIP text embeddings
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)
        return self._as_feature_tensor(text_embeds)

    def encode_image(self, images: List[Image.Image], do_rescale=False) -> torch.Tensor:
        """
        images: list of PIL Images, length B
        returns: (B, 512) CLIP image embeddings
        """
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=do_rescale).to(self.device)
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
        return self._as_feature_tensor(image_embeds)
