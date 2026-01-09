import json
import re
import os
from typing import List, Dict, Optional, Union, Tuple
from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import multimodal_typewriter_print

class VisualEvidenceExtractor:
    def __init__(self, api_key: str, model: str = 'qwen3-vl-plus', base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1'):
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
        self.agent = Assistant(llm=self.llm_cfg, function_list=self.tools, system_message='')

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

        # 2. Construct Prompt
        # 2. Query-agnostic layout detection results (MinerU format). Note: This serves only as a reference and may contain noise.
        # 3. **BBox Correction**: If the MinerU bbox is too large, too small, or shifted, generate a new, more precise bbox based on the actual visual content (0-1000 normalized coordinates [xmin, ymin, xmax, ymax]).
        # MinerU layout reference:
        # {layout_context}
        prompt_text = f"""
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

        # 3. Construct Message Payload
        messages = [
            {"role": "user", "content": [
                {"image": image_path},
                {"text": prompt_text}
            ]}
        ]

        # 4. Run Agent
        final_messages = [] # Store the full history here
        last_response_content = ""
        response_plain_text = ''
        
        # Create a generator loop to handle tool calls internally
        for ret_messages in self.agent.run(messages):
            # Update the final messages state
            final_messages = ret_messages
            # Optional: Print real-time streaming output
            # response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)
            last_response_content = ret_messages[-1]['content']

        # 5. Handle Agent Output (Text vs List of Content)
        full_text = ""
        if isinstance(last_response_content, str):
            full_text = last_response_content
        elif isinstance(last_response_content, list):
            # Extract text parts from the list content
            full_text = "".join([item['text'] for item in last_response_content if 'text' in item])

        # 6. Parse and Return Both Results
        parsed_result = self._parse_agent_response(full_text)
        
        return parsed_result, final_messages

# --- Usage Example ---

def run_demo():
    # 1. Configuration
    # API_KEY = "sk-1e374badf38a432c86886917fd8a867a" # Replace/Load from env
    # BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" 
    API_KEY = "sk-6TGzZJkJ5HfZKwnrS1A1pMb1lH5D7EDfSVC6USq24aN2JaaR" # Replace/Load from env
    BASE_URL = "http://35.220.164.252:3888/v1"
    IMAGE_PATH = "/root/LMUData/images/MMLongBench_DOC/0b85477387a9d0cc33fca0f4becaa0e5_1.jpg"
    LAYOUT_PATH = "/root/LMUData/parsed_results/MMLongBench_DOC/0b85477387a9d0cc33fca0f4becaa0e5_1.json"
    USER_QUERY = "Who is editor of the news? At least three tool calls!"

    # 2. Initialize Extractor
    extractor = VisualEvidenceExtractor(api_key=API_KEY, base_url=BASE_URL)

    # 3. Load Layout (Optional, but recommended)
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
        
        # Unpack both return values
        evidence, full_history = extractor.extract_evidence(IMAGE_PATH, USER_QUERY, layout_data)
        
        print("\n--- Final Structured Evidence ---")
        print(json.dumps(evidence, indent=2, ensure_ascii=False))

        print("\n--- Full Agent History (Debugging) ---")
        # Print the last message or the whole history to see tool calls and reasoning
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