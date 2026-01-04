import math
import os
import uuid
from io import BytesIO
from typing import List, Union

import requests
from PIL import Image

from qwen_agent.llm.schema import ContentItem
from qwen_agent.log import logger
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import extract_images_from_messages


@register_tool('image_rotate_tool')
class ImageRotateToolQwen3VL(BaseToolWithFileAccess):

    description = 'Rotate an image by a specified angle (counter-clockwise).'
    parameters = {
        'type': 'object',
        'properties': {
            'img_idx': {
                'type': 'number',
                'description': 'The index of the image to rotate (starting from 0, including images from user inputs and tool-calling returns)'
            },
            'angle': {
                'type': 'number',
                'description': 'The angle to rotate the image in degrees (0-360). Positive values rotate counter-clockwise.'
            },
        },
        'required': ['img_idx', 'angle']
    }

    # Image resizing functions (Keeping these helpers consistent with the original style
    # in case the rotated image needs to be normalized for model input)
    def round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    def floor_by_factor(self, number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor
    
    def ceil_by_factor(self, number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    def smart_resize(self,
                     height: int,
                     width: int,
                     factor: int = 32,
                     min_pixels: int = 56 * 56,
                     max_pixels: int = 12845056) -> tuple[int, int]:
        """Smart resize image dimensions based on factor and pixel constraints"""
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def call(self, params: Union[str, dict], **kwargs) -> List[ContentItem]:
        params = self._verify_json_format_args(params)

        img_idx = params['img_idx']
        angle = params['angle']
        
        images = extract_images_from_messages(kwargs.get('messages', []))
        os.makedirs(self.work_dir, exist_ok=True)

        try:
            # open image, currently only support the first image
            # (Logic identical to reference for consistency)
            image_arg = images[img_idx]
            if image_arg.startswith('file://'):
                image_arg = image_arg[len('file://'):]

            if image_arg.startswith('http'):
                response = requests.get(image_arg)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            elif os.path.exists(image_arg):
                image = Image.open(image_arg)
            else:
                image = Image.open(os.path.join(self.work_dir, image_arg))
        except Exception as e:
            logger.warning(f'{e}')
            return [ContentItem(text=f'Error: Invalid input image {images}')]

        try:
            # Rotate the image
            # expand=True ensures the output image is large enough to hold the entire rotated image
            # (i.e., corners won't be cut off)
            rotated_image = image.rotate(angle, expand=True, resample=Image.BICUBIC)

            # Optional: Check if we need to resize specifically for model constraints
            # (Similar logic to the original tool, ensuring output isn't too massive or odd-sized)
            current_w, current_h = rotated_image.size
            new_h, new_w = self.smart_resize(current_h, current_w, factor=32)
            
            if (new_w, new_h) != (current_w, current_h):
                 rotated_image = rotated_image.resize((new_w, new_h), resample=Image.BICUBIC)

            output_path = os.path.abspath(os.path.join(self.work_dir, f'{uuid.uuid4()}.png'))
            rotated_image.save(output_path)

            return [ContentItem(image=output_path)]
        except Exception as e:
            obs = f'Tool Execution Error {str(e)}'
            return [ContentItem(text=obs)]