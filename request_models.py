from typing import Optional, List

from PIL import Image
from pydantic import BaseModel, Field, validator

from consts import ImageFormat, MIN_SEED, MAX_SEED, ScalingMode, ESRGANModel, GFPGANModel
from utils import image_to_base64url


class BaseDiffusionRequest(BaseModel):
    prompt: str = Field(...)
    output_format: ImageFormat = ImageFormat.PNG
    num_inference_steps: int = Field(50, gt=0)
    guidance_scale: float = Field(7.5)
    seed: Optional[int] = Field(None, ge=MIN_SEED, le=MAX_SEED)
    batch_size: int = Field(7, gt=0)
    try_smaller_batch_on_fail: bool = True


class BaseImageGenerationRequest(BaseDiffusionRequest):
    num_variants: int = Field(4, gt=0)
    scaling_mode: ScalingMode = ScalingMode.GROW


class TextToImageRequest(BaseImageGenerationRequest):
    aspect_ratio: float = Field(1., gt=0)  # width/height


class ImageToImageRequest(BaseImageGenerationRequest):
    source_image: bytes
    strength: float = Field(0.8, ge=0, le=1)


class InpaintingRequest(ImageToImageRequest):
    mask: Optional[bytes] = None
    noise_q: float = Field(0.99, ge=0)
    color_variation: float = Field(0.01, ge=0, le=1)
    mask_blend_factor: float = Field(1.0, ge=1)
    parlance_zz_noise: bool = False
    edge_connect: bool = False
    edge_connect_mode: int = 3
    edge_threshold: float = Field(0.5, ge=0, le=1)
    

class GoBigRequest(BaseDiffusionRequest):
    image: bytes
    use_real_esrgan: bool = True
    esrgan_model: ESRGANModel = ESRGANModel.GENERAL_X4_V3
    maximize: bool = True
    strength: float = Field(.5, ge=0, le=1)
    target_width: int = Field(..., gt=0)
    target_height: int = Field(..., gt=0)
    overlap: int = Field(64, gt=0, lt=512)


class MakeTilableRequest(BaseImageGenerationRequest):
    source_image: bytes
    border_width: int = Field(50, gt=0, lt=256)
    border_softness: float = Field(.5, ge=0, le=1)
    strength: float = Field(0.8, ge=0, le=1)


class UpscaleRequest(BaseModel):
    image: bytes
    model: ESRGANModel = ESRGANModel.GENERAL_X4_V3
    target_width: int = Field(..., gt=0)
    target_height: int = Field(..., gt=0)
    maximize: bool = True


class ImageArrayResponse(BaseModel):
    images: List[bytes]

    @validator('images', pre=True)
    def images_to_bytes(cls, v: List):
        return [
            image_to_base64url(image) if isinstance(image, Image.Image) else image for image in v
        ]


class MakeTilableResponse(ImageArrayResponse):
    mask: bytes

    @validator('mask', pre=True)
    def mask_to_bytes(cls, v):
        if isinstance(v, Image.Image):
            v = image_to_base64url(v)
        return v


class ImageResponse(BaseModel):
    image: bytes

    @validator('image', pre=True)
    def image_to_bytes(cls, v):
        if isinstance(v, Image.Image):
            v = image_to_base64url(v)
        return v


class FaceRestorationRequest(BaseModel):
    image: bytes
    model_type: GFPGANModel
    use_real_esrgan: bool = True
    bg_tile: int = 400
    upscale: int = 2
    aligned: bool = False
    only_center_face: bool = False
