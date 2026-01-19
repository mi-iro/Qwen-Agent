import math
import os
import uuid
import json
import time
from io import BytesIO
from math import ceil, floor, cos, sin, radians
from typing import List, Union, Tuple, Any

import requests
from PIL import Image

from qwen_agent.llm.schema import ContentItem
from qwen_agent.log import logger
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import extract_images_from_messages

# Try importing MinerUClient
try:
    from mineru_vl_utils import MinerUClient
except ImportError:
    logger.warning("mineru_vl_utils not found. MinerU functionality will fail if called.")
    MinerUClient = None

class ImageMinerUBaseTool(BaseToolWithFileAccess):
    """
    Base tool containing shared logic for Image Layout Detection and OCR tools.
    Handles image loading, preprocessing (crop/rotate/resize), and coordinate mapping.
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.mineru_model_path = "/root/checkpoints/MinerU2.5-2509-1.2B/"
        self.mineru_server_url = "http://10.102.250.36:8000/"
        self.mineru_client = None

    def _get_mineru_client(self):
        if self.mineru_client is None:
            if MinerUClient is None:
                raise ImportError("MinerUClient module is not installed.")
            self.mineru_client = MinerUClient(
                model_name=self.mineru_model_path,
                backend="http-client",
                server_url=self.mineru_server_url.rstrip('/')
            )
        return self.mineru_client

    # --- Image Resizing Logic ---
    def round_by_factor(self, number: int, factor: int) -> int:
        return round(number / factor) * factor

    def ceil_by_factor(self, number: int, factor: int) -> int:
        return math.ceil(number / factor) * factor

    def floor_by_factor(self, number: int, factor: int) -> int:
        return math.floor(number / factor) * factor

    def smart_resize(self, height: int, width: int, factor: int = 32, min_pixels: int = 56 * 56, max_pixels: int = 12845056) -> tuple[int, int]:
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

    # --- Coordinate Logic ---
    def safe_crop_bbox(self, left, top, right, bottom, img_width, img_height):
        left = max(0, min(left, img_width))
        top = max(0, min(top, img_height))
        right = max(0, min(right, img_width))
        bottom = max(0, min(bottom, img_height))
        if left >= right: right = left + 1
        if top >= bottom: bottom = top + 1
        right = min(right, img_width)
        bottom = min(bottom, img_height)
        return [left, top, right, bottom]

    def map_point_back(self, x, y, final_size: Tuple[int, int], rotated_size: Tuple[int, int], 
                       crop_size: Tuple[int, int], crop_offset: Tuple[int, int], 
                       rotation_angle: float, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Maps a point (0-1000 relative) from the processed final image back to the original image (0-1000 relative).
        """
        # 1. Convert relative to absolute on final image
        abs_x = x / 1000.0 * final_size[0]
        abs_y = y / 1000.0 * final_size[1]

        # 2. Undo Resize
        scale_x = final_size[0] / rotated_size[0]
        scale_y = final_size[1] / rotated_size[1]
        abs_x = abs_x / scale_x
        abs_y = abs_y / scale_y

        # 3. Undo Rotation
        cx_rot, cy_rot = rotated_size[0] / 2.0, rotated_size[1] / 2.0
        cx_crop, cy_crop = crop_size[0] / 2.0, crop_size[1] / 2.0
        dx = abs_x - cx_rot
        dy = abs_y - cy_rot
        rad = radians(-rotation_angle)
        cos_a, sin_a = cos(rad), sin(rad)
        rot_x = dx * cos_a - dy * sin_a
        rot_y = dx * sin_a + dy * cos_a
        orig_crop_x = rot_x + cx_crop
        orig_crop_y = rot_y + cy_crop

        # 4. Undo Crop
        final_abs_x = orig_crop_x + crop_offset[0]
        final_abs_y = orig_crop_y + crop_offset[1]

        # 5. Normalize
        norm_x = min(1000, max(0, int(final_abs_x / original_size[0] * 1000)))
        norm_y = min(1000, max(0, int(final_abs_y / original_size[1] * 1000)))
        return norm_x, norm_y

    def _prepare_image_input(self, params: dict, kwargs: dict) -> Tuple[Image.Image, Image.Image, dict, str]:
        """
        Common pipeline: Load -> Crop -> Rotate -> Smart Resize -> Save
        Returns: (original_image, processed_image, transform_info, output_path)
        """
        img_idx = params.get('img_idx', 0)
        bbox = params['bbox']
        angle = params.get('angle', 0)

        images = extract_images_from_messages(kwargs.get('messages', []))
        os.makedirs(self.work_dir, exist_ok=True)

        # 1. Load Image
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

        # 2. Coordinates
        img_width, img_height = image.size
        rel_x1, rel_y1, rel_x2, rel_y2 = bbox
        abs_x1 = rel_x1 / 1000.0 * img_width
        abs_y1 = rel_y1 / 1000.0 * img_height
        abs_x2 = rel_x2 / 1000.0 * img_width
        abs_y2 = rel_y2 / 1000.0 * img_height

        left, top, right, bottom = self.safe_crop_bbox(abs_x1, abs_y1, abs_x2, abs_y2, img_width, img_height)
        crop_offset = (left, top)
        crop_size = (right - left, bottom - top)

        # 3. Crop
        cropped_image = image.crop((left, top, right, bottom))

        # 4. Rotate
        rotated_image = cropped_image.rotate(angle, expand=True)
        rotated_size = rotated_image.size

        # 5. Resize
        new_h, new_w = self.smart_resize(
            height=rotated_size[1],
            width=rotated_size[0],
            factor=32,
            min_pixels=32 * 32,
            max_pixels=12845056
        )
        final_image = rotated_image.resize((new_w, new_h), resample=Image.BICUBIC)
        
        # Save
        output_filename = f'{uuid.uuid4()}.png'
        output_path = os.path.abspath(os.path.join(self.work_dir, output_filename))
        final_image.save(output_path)

        transform_info = {
            'final_size': final_image.size,
            'rotated_size': rotated_size,
            'crop_size': crop_size,
            'crop_offset': crop_offset,
            'angle': angle,
            'original_size': image.size
        }

        return image, final_image, transform_info, output_path


@register_tool('image_layout_detection_tool')
class ImageLayoutDetectionTool(ImageMinerUBaseTool):
    description = 'Detect the layout of a specific region in an image by cropping it based on a bounding box (bbox). Returns the image region and a list of detected layout elements.'
    parameters = {
        'type': 'object',
        'properties': {
            'label': {
                'type': 'string',
                'description': 'The name or label of the object in the specified bounding box'
            },
            'bbox': {
                'type': 'array',
                'items': {'type': 'number'},
                'minItems': 4,
                'maxItems': 4,
                'description': 'The bbox specified as [x1, y1, x2, y2] in 0-1000 coordinates, relative to the page image from the user.'
            },
            'angle': {
                'type': 'number',
                'description': 'The angle to rotate the image (counter-clockwise) after cropping. Default is 0.',
                'default': 0
            }
        },
        'required': ['bbox', 'label']
    }

    def call(self, params: Union[str, dict], **kwargs) -> List[ContentItem]:
        params = self._verify_json_format_args(params)
        
        try:
            # Prepare image (Crop -> Rotate -> Resize)
            original_image, final_image, t_info, output_path = self._prepare_image_input(params, kwargs)
            
            # Call MinerU Layout Detect
            client = self._get_mineru_client()
            # Note: The provided interface for layout_detect takes (image, priority)
            layout_results = client.layout_detect(final_image)

            detected_elements = []
            
            if layout_results:
                for block in layout_results:
                    # Assuming ContentBlock has 'type' and 'bbox' attributes
                    # 'bbox' from MinerU is typically normalized [0,1] relative to the input image (final_image)
                    b_type = getattr(block, 'type', 'unknown')
                    b_bbox = getattr(block, 'bbox', None)

                    if b_bbox and len(b_bbox) == 4:
                        # Convert normalized 0-1 to 0-1000 for mapping
                        x1, y1, x2, y2 = [c * 1000 for c in b_bbox]

                        # Map back to original image global coordinates
                        orig_x1, orig_y1 = self.map_point_back(
                            x1, y1, t_info['final_size'], t_info['rotated_size'], 
                            t_info['crop_size'], t_info['crop_offset'], 
                            t_info['angle'], t_info['original_size']
                        )
                        orig_x2, orig_y2 = self.map_point_back(
                            x2, y2, t_info['final_size'], t_info['rotated_size'], 
                            t_info['crop_size'], t_info['crop_offset'], 
                            t_info['angle'], t_info['original_size']
                        )

                        detected_elements.append({
                            "type": b_type,
                            "bbox": [
                                min(orig_x1, orig_x2),
                                min(orig_y1, orig_y2),
                                max(orig_x1, orig_x2),
                                max(orig_y1, orig_y2)
                            ]
                        })

            return [
                ContentItem(image=output_path),
                ContentItem(text=f"Layout Detection Result (Mapped to original coords): {json.dumps(detected_elements, ensure_ascii=False)}")
            ]

        except Exception as e:
            logger.error(f"Layout Detection Error: {e}")
            return [ContentItem(text=f'Error executing layout detection: {str(e)}')]


@register_tool('image_ocr_tool')
class ImageOCRTool(ImageMinerUBaseTool):
    description = 'Recognize content (e.g., text, formula, table) within a specific region of an image based on the element type.'
    parameters = {
        'type': 'object',
        'properties': {
            'label': {
                'type': 'string',
                'description': 'The name or label of the object in the specified bounding box'
            },
            'bbox': {
                'type': 'array',
                'items': {'type': 'number'},
                'minItems': 4,
                'maxItems': 4,
                'description': 'The bbox specified as [x1, y1, x2, y2] in 0-1000 coordinates, relative to the page image from the user.'
            },
            'angle': {
                'type': 'number',
                'description': 'The angle to rotate the image (counter-clockwise) after cropping. Default is 0.',
                'default': 0
            },
            'element_type': {
                'type': 'string',
                'description': 'The type of element to recognize (e.g., "text", "formula", "table"). Default is "text".',
                'default': 'text',
                'enum': ['text', 'formula', 'table']
            }
        },
        'required': ['bbox', 'label']
    }

    def call(self, params: Union[str, dict], **kwargs) -> List[ContentItem]:
        params = self._verify_json_format_args(params)
        element_type = params.get('element_type', 'text')

        try:
            # Prepare image (Crop -> Rotate -> Resize)
            _, final_image, _, output_path = self._prepare_image_input(params, kwargs)

            # Call MinerU Content Extract
            client = self._get_mineru_client()
            # Note: The provided interface for content_extract takes (image, type, priority)
            ocr_content = client.content_extract(final_image, type=element_type)

            if ocr_content is None:
                ocr_content = "No content recognized."

            return [
                ContentItem(image=output_path),
                ContentItem(text=f"OCR Content Result: {ocr_content}")
            ]

        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return [ContentItem(text=f'Error executing OCR: {str(e)}')]