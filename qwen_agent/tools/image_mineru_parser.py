import os
import json
import traceback
import requests
from io import BytesIO
from typing import Union, List, Optional, Dict

from PIL import Image

# Qwen-Agent 相关导入
from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.utils import extract_images_from_messages

# MinerU / vLLM 相关导入
try:
    from vllm import LLM
    from mineru_vl_utils import MinerUClient
    try:
        from mineru_vl_utils import MinerULogitsProcessor
        HAS_LOGITS_PROCESSOR = True
    except ImportError:
        HAS_LOGITS_PROCESSOR = False
except ImportError:
    print("Warning: 'vllm' or 'mineru_vl_utils' not found. MinerUParser will not work.")
    HAS_LOGITS_PROCESSOR = False


@register_tool('mineru_parser')
class MinerUParser(BaseTool):
    description = 'Parse a document image to extract layout, text, table and formula. Returns the parsed result in JSON format.'
    parameters = {
        'type': 'object',
        'properties': {
            'img_idx': {
                'type': 'integer',
                'description': 'The index of the image to be parsed (starting from 0, including images from user inputs and tool-calling returns).'
            }
        },
        'required': ['img_idx']
    }

    # 类变量用于实现单例模式
    _client_instance = None
    _model_path = "/mnt/shared-storage-user/mineru2-shared/madongsheng/modelscope/MinerU2.5-2509/"

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        if cfg and 'model_path' in cfg:
            self.model_path = cfg['model_path']
        else:
            self.model_path = self._model_path

    def _get_or_create_client(self):
        """单例模式获取 MinerUClient"""
        if MinerUParser._client_instance is not None:
            return MinerUParser._client_instance

        print(f"[MinerUParser] Initializing model from: {self.model_path} ...")
        
        init_kwargs = {}
        if HAS_LOGITS_PROCESSOR:
            init_kwargs['logits_processors'] = [MinerULogitsProcessor]

        try:
            llm = LLM(
                model=self.model_path,
                **init_kwargs
            )
            client = MinerUClient(
                backend="vllm-engine",
                vllm_llm=llm
            )
            MinerUParser._client_instance = client
            return client
        except Exception as e:
            print(f"[MinerUParser] Failed to load model: {e}")
            raise e

    def call(self, params: Union[str, dict], **kwargs) -> List[ContentItem]:
        """
        执行工具调用，通过 img_idx 从上下文中获取图片并解析
        """
        # 1. 解析参数
        params = self._verify_json_format_args(params)
        try:
            img_idx = int(params['img_idx'])
        except (ValueError, KeyError):
             return [ContentItem(text='Error: Invalid or missing "img_idx" parameter.')]

        # 2. 从消息历史中提取图片列表
        messages = kwargs.get('messages', [])
        images = extract_images_from_messages(messages)

        if img_idx < 0 or img_idx >= len(images):
            return [ContentItem(text=f'Error: Invalid img_idx {img_idx}. Found {len(images)} images in context.')]

        try:
            # 3. 初始化客户端
            client = self._get_or_create_client()

            # 4. 获取图片路径/URL并加载
            image_arg = images[img_idx]
            
            # 处理 file:// 前缀
            if image_arg.startswith('file://'):
                image_arg = image_arg[len('file://'):]

            # 加载图片 (支持 HTTP URL 和 本地路径)
            if image_arg.startswith('http'):
                response = requests.get(image_arg, timeout=30)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            elif os.path.exists(image_arg):
                image = Image.open(image_arg)
            else:
                # 尝试结合工作目录查找
                if hasattr(self, 'work_dir') and self.work_dir:
                     potential_path = os.path.join(self.work_dir, image_arg)
                     if os.path.exists(potential_path):
                         image = Image.open(potential_path)
                     else:
                         return [ContentItem(text=f'Error: Image file not found: {image_arg}')]
                else:
                    return [ContentItem(text=f'Error: Image file not found: {image_arg}')]

            # 5. 执行 MinerU 解析
            # MinerU 的 two_step_extract 可能会比较耗时，适合异步或耐心等待
            extracted_blocks = client.two_step_extract(image)
            
            for block in extracted_blocks:
                block['bbox'] = [int(cord*1000) for cord in block['bbox']]
            
            # 6. 返回 JSON 结果
            json_result = json.dumps(extracted_blocks, ensure_ascii=False, indent=2)
            return [ContentItem(text=json_result)]

        except Exception as e:
            error_msg = f"[MinerUParser] Execution Error: {str(e)}"
            traceback.print_exc()
            return [ContentItem(text=error_msg)]