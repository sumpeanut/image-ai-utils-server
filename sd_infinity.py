from infinity.utils import functbl

import numpy as np
from request_models import InpaintingRequest
from PIL import Image, ImageOps

class SdInfinityFilter:
    def __init__(
        self
    ):
        super().__init__()

    def applyTo(self, request: InpaintingRequest, source_image: Image.Image, mask: Image.Image): 
        if request.sd_infinity:
            np_source_image = np.array( source_image.convert("RGB") )
            np_mask = np.array( ImageOps.invert( mask ) )
            np_source_image, np_mask = functbl[request.sd_infinity_mode]( np_source_image, np_mask )
            source_image = Image.fromarray(np_source_image)
        return source_image
