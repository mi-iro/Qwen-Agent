import json
import re
from qwen_agent.agents import Assistant

def extract_relevant_elements_with_vlm(image_path, mineru_layout, user_query, agent_instance):
    """
    使用VLM提取页面中与用户查询相关的元素，并修正MinerU的布局噪声。
    
    Args:
        image_path (str): 本地图片路径或URL。
        mineru_layout (list): MinerU生成的布局列表 (List of dicts)。
        user_query (str): 用户想要查询的内容。
        agent_instance (Assistant): 初始化的Qwen-Agent实例。
        
    Returns:
        list: 包含相关元素的列表，格式与MinerU一致但经过清洗。
    """
    
    # 1. 将MinerU布局数据转换为字符串，作为上下文提供给模型
    # 为了节省token，可以只保留 content 和 bbox，去掉 angle 等非必要字段
    simplified_layout = []
    for item in mineru_layout:
        simplified_layout.append({
            "type": item.get("type"),
            "bbox": [ int(cord*1000) for cord in item.get("bbox")],
            "angle": item.get("angle", 0),
            "content": item.get("content", "") # 截断过长文本以节省上下文
        })
    layout_context = json.dumps(simplified_layout, ensure_ascii=False)

    # 2. 构建提示词 (Prompt)
    prompt_text = f"""
    输入包含：
    1. 视觉文档的页面图像。
    2. 无查询感知的布局检测结果（MinerU格式），这仅作为参考，可能包含错误的粒度划分或不准确的边界框等噪声。
    3. 用户的查询意图。
    
    请执行以下步骤：
    1. 语义匹配：仔细观察图像，判断页面内容是否包含用户查询相关的证据信息。如果不相关，返回空列表。
    2. 精准定位：如果相关，请提取与查询匹配的完整视觉证据链（可能包含多个文本块、表格、图像区域等元素）。
    3. BBox修正：如果MinerU的 bbox 范围过大（包含查询无关内容）、过小（截断内容、过度拆分）或位置偏移，请根据图像实际视觉内容，生成新的、更精准的 bbox（0-1000的归一化坐标[xmin, ymin, xmax, ymax]）。

    供参考的MinerU布局检测结果：
    {layout_context}
    
    用户查询：'{user_query}'
    
    最终请输出相关证据列表，格式如下：
    ```json
    [
      {{
        "evidence": "<self-contained evidence point, understandable without page context>",
        "bbox": [xmin, ymin, xmax, ymax],  # 0-1000归一化坐标
        "angle": <int> 
      }},
      ...
    ]
    ```
    """

    # 3. 构造消息体
    messages = [
        {"role": "system", "content": "你是一位专业的视觉文档分析专家。你的任务是根据用户查询，结合工具调用，从页面图像中提取完整的回复证据。"},
        {"role": "user", "content": [
            {"image": image_path},
            {"text": prompt_text}
        ]}
    ]

    # 4. 调用 Agent
    # 注意：qwen_agent 的 run 返回是一个生成器，我们需要获取完整结果
    last_response = ""
    try:
        for response in agent_instance.run(messages,stream=False):
            # 获取最后一条消息的内容
            last_response = response[-1]['content']
    except Exception as e:
        print(f"Error calling agent: {e}")
        return []

    # 5. 解析输出 (清洗 Markdown 代码块标记)
    try:
        # 去除可能存在的 markdown 代码块标记 ```json ... ```
        clean_json_str = re.sub(r'^```json\s*', '', last_response.strip())
        clean_json_str = re.sub(r'^```\s*', '', clean_json_str)
        clean_json_str = re.sub(r'\s*```$', '', clean_json_str)
        
        result_list = json.loads(clean_json_str)
        
        # 简单验证返回格式
        if isinstance(result_list, list):
            return result_list
        else:
            print("Warning: Model return is not a list.")
            return []
            
    except json.JSONDecodeError:
        print(f"Failed to parse JSON response. Raw response: {last_response}")
        return []

# --- 使用示例 ---

if __name__ == "__main__":
    # 配置你的 LLM (保持你原本的配置)
    llm_cfg = {
        'model_type': 'qwenvl_oai',
        'model': 'qwen3-vl-plus',
        # 'model': 'gemini-3-pro-preview',
        # 'model_server': 'http://127.0.0.1:3888/v1', 
        'model_server': 'http://35.220.164.252:3888/v1', 
        'api_key': 'sk-ohsIxhcDUF0xwqqmFl1L1niRtEOD9LnvxFGjtjakXennNTzI',
        'generate_cfg': {'top_p': 0.8, 'top_k': 20, 'temperature': 1.0} # 建议降低 temperature 以获得稳定的 JSON
    }
    
    tools = ['image_zoom_in_tool', 'image_rotate_tool']
    # 初始化 Agent (此时不需要 tools，纯视觉分析即可，或者保留 tools 供内部思考)
    agent = Assistant(llm=llm_cfg, function_list=tools, system_message=analysis_prompt)

    # 模拟数据
    image_file = "/root/LMUData/images/MMLongBench_DOC/0b85477387a9d0cc33fca0f4becaa0e5_1.jpg"
    # query = "Who is the editor of NAVAL MEDICAL RESEARCH AND DEVELOPMENT"
    # query = "Who is the commanding officer of NAVAL MEDICAL RESEARCH AND DEVELOPMENT"
    # query = "Who is the public affairs officer of NAVAL MEDICAL RESEARCH AND DEVELOPMENT"
    # query = "What is the website link?"
    query = "What is the brand of the machine in the image?"
    layout_file = "/root/LMUData/parsed_results/MMLongBench_DOC/0b85477387a9d0cc33fca0f4becaa0e5_1.json"
    layout_data = open(layout_file, 'r', encoding='utf-8').read()
    layout_data = json.loads(layout_data)

    # 执行提取
    results = extract_relevant_elements_with_vlm(image_file, layout_data, query, agent)
    
    print("提取结果:")
    print(json.dumps(results, indent=2, ensure_ascii=False))