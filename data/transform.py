# module to add custom transforms
import torch
import math


class ToTiles:
    """
    transforms image to to tiles (C, H, W) -> (num_tile, -1)
    """

    def __init__(self, image_size: int = 36, num_tile: int = 9):
        div_term = int(math.sqrt(num_tile))

        assert div_term * div_term == num_tile, "only can transfer square images"

        self.num_tiles = num_tile
        self.div_term = div_term

    def __call__(self, image):
        H = image.size(1)
        assert H % self.div_term == 0
        stride = int(H / self.div_term)

        fixed = torch.zeros_like(image).view(3, self.num_tiles, stride, stride)
        k = 0
        for i in range(self.div_term):
            for j in range(self.div_term):
                width = i * stride
                height = j * stride
                fixed[:, k] = image[:, width:width + stride, height:height + stride]
                k += 1

        return fixed.permute(1, 2, 3, 0).reshape(9, -1)
