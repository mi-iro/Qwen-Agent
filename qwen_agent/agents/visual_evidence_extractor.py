infer_system_prompt = """
You are an advanced Visual Document Analysis Agent capable of precise evidence extraction from document images. Your goal is to answer user queries by locating, reading, and extracting specific information from a page.

### Your Capabilities & Tools
You have access to a powerful tool named **`image_zoom_and_ocr_tool`**.

- **Functionality**:  
  Crop a specific region of the image, optionally rotate it, and perform OCR on the cropped region.

- **When to use**:  
  - Always use this tool when the user asks for **specific text, numbers, names, dates, tables, or factual details** from the page.
  - Do NOT rely solely on the global low-resolution image when reading dense or small text.
  - If the target text is rotated, estimate and set the `angle` parameter before OCR.

- **Parameters**:
  - `label`: A short description of what you are looking for.
  - `bbox`: `[xmin, ymin, xmax, ymax]` in **0–1000 normalized coordinates**, relative to the original page.
  - `angle`: Rotation angle (counter-clockwise) applied after cropping. Default is `0`.
  - `do_ocr`: Whether to perform OCR on the cropped image.

### Tool Usage Example
Use the tool strictly in the following format:

<tool_call>
{"name": "image_zoom_and_ocr_tool", "arguments": {"label": "<A short description of what you are looking for>", "bbox": [xmin, ymin, xmax, ymax], "angle":<0/90/180/270>, "do_ocr": <true/false>}}
</tool_call>

### Your Input and Task
The input includes:
1. One page image of a visual document.
2. The user's query intent.

Please execute the following steps:
1. **Semantic Matching**: Carefully observe the image to determine if the page content contains evidence information relevant to the user's query. If it is irrelevant, return an empty list.
2. **Precise Localization**: If relevant, extract the complete chain of visual evidence that helps to answer the query (text blocks, tables, charts or image regions).
3. **Speical Notes**: The page image may contain several evidence pieces. Pay attention to tables, charts and images, as they could also contain evidence.

### Output Format
After gathering information, output the list of relevant evidence in the following JSON format.  
If the page image is not relevant, return an empty list.

```json
[
  {
    "evidence": "<self-contained content, understandable without page context>",
    "bbox": [xmin, ymin, xmax, ymax] # 0-1000 normalized coordinates 
  }
  ...
]
```

Let us think step by step, using tool calling for better understanding of details!
"""

distill_system_prompt = """
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "image_zoom_and_ocr_tool", "description": "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox), optionally rotate it or perform OCR.", "parameters": {"type": "object", "properties": {"label": {"type": "string", "description": "The name or label of the object in the specified bounding box"}, "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "The bbox specified as [x1, y1, x2, y2] in 0-1000 coordinates, relative to the page image from the user."}, "angle": {"type": "number", "description": "The angle to rotate the image (counter-clockwise) after cropping. Default is 0.", "default": 0}, "do_ocr": {"type": "boolean", "description": "Whether OCR the processed image. OCR returns results with bboxes relative to the page image from user. Default is False.", "default": false}}, "required": ["bbox", "label"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""

import json
import re
import os
from typing import List, Dict, Optional, Union, Tuple
from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import multimodal_typewriter_print

class VisualEvidenceExtractor:
    def __init__(self, api_key: str, model: str = 'qwen3-vl-plus', base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1', mode: str = 'distill'):
        """
        Initializes the VLM Agent.
        
        Args:
            api_key (str): DASHSCOPE_API_KEY.
            model (str): The model name (default: qwen3-vl-plus).
            base_url (str): The API base URL.
        """
        self.llm_cfg = {
            'model_type': 'qwenvl_oai',
            'model': model,
            'model_server': base_url,
            'api_key': api_key,
            'generate_cfg': {
                'top_p': 0.8,
                'top_k': 20,
                'temperature': 1.0
            }
        }
        # Assuming 'image_zoom_and_ocr_tool' is available in your environment's qwen_agent registry
        self.tools = ['image_zoom_and_ocr_tool']
        self.mode = mode
        if self.mode == 'infer':
            self.agent = Assistant(llm=self.llm_cfg, function_list=self.tools, system_message=infer_system_prompt)
        elif self.mode == 'distill':
            self.agent = Assistant(llm=self.llm_cfg, function_list=self.tools, system_message=distill_system_prompt)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Please choose 'infer' or 'distill'.")
            
        # self.agent.llm.use_raw_api=True

    def _prepare_layout_context(self, layout_data: List[Dict]) -> str:
        """Helper to simplify and stringify MinerU layout data."""
        if not layout_data:
            return "No layout reference available."
            
        simplified_layout = []
        for item in layout_data:
            simplified_layout.append({
                "type": item.get("type"),
                "bbox": [int(cord * 1000) for cord in item.get("bbox", [])] if item.get("bbox") else [],
                # "angle": item.get("angle"),
                "content": item.get("content", "")
            })
        return json.dumps(simplified_layout, ensure_ascii=False)

    def _parse_agent_response(self, response_text: str) -> List[Dict]:
        """Parses JSON from the LLM's Markdown response."""
        try:
            # Regex to find JSON arrays inside or outside markdown blocks
            pattern = r"```(?:json)?\s*(\[.*\])\s*```|(\[.*\])"
            match = re.search(pattern, response_text, re.DOTALL)

            if match:
                clean_json_str = match.group(1) if match.group(1) else match.group(2)
            else:
                # Fallback: assume the whole text might be the JSON
                clean_json_str = response_text

            result_list = json.loads(clean_json_str.strip())
            
            if isinstance(result_list, list):
                return result_list
            else:
                print("Warning: Model return is not a list.")
                return []
        except json.JSONDecodeError:
            print(f"Failed to parse JSON. Raw response snippet: {response_text[:100]}...")
            return []

    def extract_evidence(self, image_path: str, query: str, layout_data: Optional[List[Dict]] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Main function to extract evidence from a page image based on a query.

        Args:
            image_path (str): Path or URL to the image.
            query (str): The user's question.
            layout_data (list, optional): MinerU layout data. Defaults to None.

        Returns:
            Tuple[List[Dict], List[Dict]]: 
                1. A list of dictionaries containing 'evidence', 'bbox', etc.
                2. The full list of agent messages (history).
        """
        
        # 1. Prepare Layout Context
        layout_context = self._prepare_layout_context(layout_data) if layout_data else "Not provided."

        # 2. Construct Message Payload
        distill_prompt = f"""
The input includes:
1. The page image of a visual document.
2. The user's query intent.

Please execute the following steps:
1. **Semantic Matching**: Carefully observe the image to determine if the page content contains evidence information relevant to the user's query. If it is irrelevant, return an empty list.
2. **Precise Localization**: If relevant, extract the complete chain of visual evidence that helps to answer the query (text blocks, tables, charts or image regions).
3. **Speical Notes**: The page image may contain several evidence pieces. Pay attention to tables, charts and images, as they could also contain evidence.

User Query: '{query}'

Finally, output the list of relevant evidence in the following format, return an empty list if not relevant:
```json
[
{{
"evidence": "<self-contained content, understandable without page context>",
"bbox": [xmin, ymin, xmax, ymax]  # 0-1000 normalized coordinates
}},
...
]
```

Let us think step by step, using tool calling for better understanding of details!
        """
        
        if self.mode == 'infer':
            messages = [
                {"role": "user", "content": [
                    {"image": image_path},
                    {"text": query}
                ]}
            ]
        elif self.mode == 'distill':
            messages = [
                {"role": "user", "content": [
                    {"image": image_path},
                    {"text": distill_prompt}
                ]}
            ]

        # 3. Run Agent
        final_messages = [] # Store the full history here
        last_response_content = ""
        response_plain_text = ''
        last_response_role = "assistant"
        
        # Create a generator loop to handle tool calls internally
        for ret_messages in self.agent.run(messages):
            # Update the final messages state
            final_messages = ret_messages
            # Optional: Print real-time streaming output
            # response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)
            last_response_content = ret_messages[-1]['content']
            last_response_role = ret_messages[-1]['role']
            
        if last_response_role != "assistant":
            print("Warning: The last response is not from the assistant.")
            return [], final_messages

        # 4. Handle Agent Output (Text vs List of Content)
        full_text = ""
        if isinstance(last_response_content, str):
            full_text = last_response_content
        elif isinstance(last_response_content, list):
            # Extract text parts from the list content
            full_text = "".join([item['text'] for item in last_response_content if 'text' in item])

        # 5. Parse and Return Both Results
        parsed_result = self._parse_agent_response(full_text)
        
        return parsed_result, final_messages

from PIL import Image, ImageDraw, ImageFont

def visualize_evidence(image_path: str, evidence_list: List[Dict], output_path: str = "output_viz.png"):
    """
    在图像上绘制 Agent 提取的 Evidence BBox。
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    try:
        # 1. 打开图片
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size

        # 尝试加载字体，如果失败则使用默认字体
        try:
            # 尝试加载常见字体，字号根据图片大小自适应
            font_size = max(15, int(height / 60))
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # 2. 遍历证据列表并绘图
        for i, item in enumerate(evidence_list):
            bbox = item.get("bbox", [])
            text_content = item.get("evidence", "")

            # 检查 bbox 格式 [xmin, ymin, xmax, ymax]
            if bbox and len(bbox) == 4:
                # 归一化坐标 (0-1000) 转 绝对像素坐标
                x1 = bbox[0] / 1000 * width
                y1 = bbox[1] / 1000 * height
                x2 = bbox[2] / 1000 * width
                y2 = bbox[3] / 1000 * height

                # 生成随机颜色以区分不同的证据块，或者统一使用红色
                color = "red" 
                
                # 绘制矩形框 (Width 控制线条粗细)
                line_width = max(3, int(width / 300))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

                # 准备标签文本 (索引 + 截断的内容)
                label = f"[{i+1}] {text_content[:30]}..." if len(text_content) > 30 else f"[{i+1}] {text_content}"
                
                # 计算文本背景框大小
                text_bbox = draw.textbbox((x1, y1), label, font=font)
                # 调整文本位置，防止画出图片外
                text_loc = [x1, y1 - (text_bbox[3] - text_bbox[1])]
                if text_loc[1] < 0: 
                    text_loc = [x1, y2] # 如果上方没空间，就画在框下面

                # 绘制文本背景和文本
                draw.rectangle(draw.textbbox(tuple(text_loc), label, font=font), fill=color)
                draw.text(text_loc, label, fill="white", font=font)

        # 3. 保存图片
        image.save(output_path)
        print(f"Visualization saved to: {output_path}")
    
    except Exception as e:
        print(f"Error during visualization: {e}")

# ==========================================
# 修改后的 run_demo
# ==========================================

def run_demo():
    # 1. Configuration
    # API_KEY = "sk-1e374badf38a432c86886917fd8a867a" # Replace/Load from env
    # BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" 
    API_KEY = "sk-6TGzZJkJ5HfZKwnrS1A1pMb1lH5D7EDfSVC6USq24aN2JaaR" # Replace/Load from env
    BASE_URL = "http://35.220.164.252:3888/v1"
    MODEL_NAME = "qwen3-vl-plus"
    # API_KEY = "sk-123456" # Replace/Load from env
    # BASE_URL = "http://localhost:8001/v1"
    # MODEL_NAME = "MinerU-Agent"
    # IMAGE_PATH = "/mnt/shared-storage-user/mineru2-shared/madongsheng/dataset/vidore_v3_computer_science/sample_results_images/869.png"
    IMAGE_PATH = "/root/LMUData/images/MMLongBench_DOC/0b85477387a9d0cc33fca0f4becaa0e5_1.jpg"
    LAYOUT_PATH = "/root/LMUData/parsed_results/MMLongBench_DOC/0b85477387a9d0cc33fca0f4becaa0e5_1.json"
    USER_QUERY = "Who is editor of the news?"

    # 2. Initialize Extractor
    extractor = VisualEvidenceExtractor(api_key=API_KEY, base_url=BASE_URL, model=MODEL_NAME)

    # 3. Load Layout (Optional)
    layout_data = []
    if os.path.exists(LAYOUT_PATH):
        try:
            with open(LAYOUT_PATH, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
        except Exception as e:
            print(f"Error loading layout: {e}")

    # 4. Execute Extraction
    if os.path.exists(IMAGE_PATH):
        print(f"Analyzing image for query: '{USER_QUERY}'...")
        
        # 获取结果
        evidence, full_history = extractor.extract_evidence(IMAGE_PATH, USER_QUERY, layout_data)
        
        print("\n--- Final Structured Evidence ---")
        print(json.dumps(evidence, indent=2, ensure_ascii=False))

        # -------------------------------------------------
        # 新增：调用可视化函数
        # -------------------------------------------------
        if evidence:
            output_viz_path = "result_visualization.png"
            visualize_evidence(IMAGE_PATH, evidence, output_viz_path)
            print(f"Visualization saved to: {output_viz_path}")
        else:
            print("No evidence found to visualize.")

        print("\n--- Full Agent History (Debugging) ---")
        for msg in full_history:
            print(f"\n[{msg['role'].upper()}]")
            # Handle list content (common in multimodal messages)
            content = msg['content']
            if isinstance(content, list):
                print(json.dumps(content, ensure_ascii=False))
            else:
                print(content)
            
            # Print function calls if present
            if 'function_call' in msg:
                print(f"Function Call: {msg['function_call']}")
            
    else:
        print(f"Image not found at {IMAGE_PATH}")

if __name__ == "__main__":
    run_demo()