from edge.src.edge_connect import EdgeConnect
from edge.src.config import Config

from request_models import InpaintingRequest
from PIL import Image

eConfig = Config("edge/checkpoints/places2/config.yml")
edgeConnect = EdgeConnect(eConfig)
edgeConnect.load()

class EdgeConnectFilter:
    def __init__(
        self
    ):
        super().__init__()

    def applyTo(self, request: InpaintingRequest, source_image: Image.Image, mask: Image.Image): 
        if request.edge_connect:
            source_image = edgeConnect.test( source_image, mask )
        return source_image
