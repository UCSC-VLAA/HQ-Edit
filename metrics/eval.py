import json
import os
import time
from tqdm import tqdm
import base64
import json
import re
from openai import OpenAI
import io
import pyarrow.parquet as pq

from PIL import Image
from io import BytesIO
from argparse import ArgumentParser
from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed
from langchain_community.chat_models import AzureChatOpenAI
from langchain.schema import (
    HumanMessage, 
    SystemMessage
)


from prompt import (
    ALIGNMENT_PROMPT,
    COHERENCE_PROMPT,
    EDIT_ITEM_EXAMPLE
)

# openai api here
BASE_URL = ""
DEPLOYMENT_NAME = "" 
API_KEY = "" 


# Function to encode the image
def load_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)
    return data

def encode_image(image):
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=100)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

def load_jsons(input_jsons):
    if input_jsons.endswith("sorted.jsonl"):
        items = load_jsonl(input_jsons)
        datas = {f"{int(', '.join(item.keys())):03}": item for i, item in enumerate(items)}
        return datas
    if input_jsons.endswith("jsonl"):
        items = load_jsonl(input_jsons)
        datas = {f"{i:03}": item for i, item in enumerate(items)}
        return datas

    if os.path.isdir(input_jsons):
        data = {}
        for file in os.listdir(input_jsons):
            if file.endswith(".json"):
                with open(os.path.join(input_jsons, file), "r") as f:
                    data_ = json.load(f)
                    data = {**data, **data_}
    else:
        with open(input_jsons, "r") as f:
            data = json.load(f)
    return data

def dump_json(filename, odgt):
    with open(filename, 'w') as f:
        json.dump(odgt, f, indent=4)


class EditActionEvaluator():
    def __init__(self, metric_type=None):
        self.model = AzureChatOpenAI(
            azure_endpoint=BASE_URL,
            openai_api_version="2023-12-01-preview",
            deployment_name=DEPLOYMENT_NAME,
            openai_api_key=API_KEY,
            openai_api_type="azure",
            temperature=0.1,
            max_tokens=512,
        )
        # append for files
        self.metric_type = metric_type

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
    def alignment(self, images, edit_action):
        # multi image
        sys_message = ALIGNMENT_PROMPT
        hum_message = EDIT_ITEM_EXAMPLE.format(edit_action=edit_action)
        request = [{
                "type": "text",
                "text": hum_message
            }]
        for image in images:
            base64_image = encode_image(image)
            input_ip = {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            request.append({
                "type": "image_url",
                "image_url": input_ip,
            })
            
        generate_log = self.model([
            SystemMessage(content=sys_message),
            HumanMessage(content=request)])

        print(generate_log.content)
        json_string = re.findall(r'\{.*?\}', generate_log.content, re.DOTALL)[0]
        return json.loads(json_string)

    def eval_alignment(self, 
                   image_folder, 
                   ):

        table = pq.read_table(image_folder)

        df = table.to_pandas()

        results = {}
        count = 0
        scores = 0.0

        for index, row in df.iterrows():
            v = {}
            image1 = Image.open(io.BytesIO(row["input_image"])).convert('RGB')
            image2 = Image.open(io.BytesIO(row["output_image"])).convert('RGB')

            edit = row["edit"]
            v["edit"] = edit
            images = [image1, image2]
            eval_file = f"{os.path.dirname(image_folder)}/{index}_alignment.json"
            if os.path.exists(eval_file):
                gpt4v_response = load_jsons(eval_file)
            else:
                infer_func = self.alignment 
                try:
                    gpt4v_response = infer_func(images, edit)
                    v["gpt4v_response"] = gpt4v_response
                    dump_json(eval_file, gpt4v_response)
                except :
                    gpt4v_response["Score"] = None
                    print("retry error")


            if isinstance(gpt4v_response["Score"], str):
                gpt4v_response["Score"] = 0

            if gpt4v_response["Score"] is None:
                gpt4v_response["Score"] = 0

            results[index] = v
            scores += gpt4v_response["Score"]
            

        print(f"mean score: {scores / count}")
        results["score"] = scores / count
        return results


    @retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
    def coherence(self, image, edit_action):
        # multi image
        sys_message = COHERENCE_PROMPT
        request = []
        base64_image = encode_image(image)
        input_ip = {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
        request.append({
            "type": "image_url",
            "image_url": input_ip,
        })

        generate_log = self.model([
            SystemMessage(content=sys_message),
            HumanMessage(content=request)])

        print(generate_log.content)
        json_string = re.findall(r'\{.*?\}', generate_log.content, re.DOTALL)[0]
        return json.loads(json_string)

    def eval_coherence(self, 
                   image_folder, 
                   ):
        """
            Eval the full json files
        """
        table = pq.read_table(image_folder)

        df = table.to_pandas()

        results = {}
        count = 0
        scores = 0.0

        for index, row in df.iterrows():
            v = {}
            image1 = Image.open(io.BytesIO(row["input_image"])).convert('RGB')
            image2 = Image.open(io.BytesIO(row["output_image"])).convert('RGB')

            edit = row["edit"]
            v["edit"] = edit
            images = image2
            eval_file = f"{os.path.dirname(image_folder)}/{index}_harmony.json"
            if os.path.exists(eval_file):
                gpt4v_response = load_jsons(eval_file)
            else:
                infer_func = self.coherence
                try:
                    gpt4v_response = infer_func(images, edit)
                    v["gpt4v_response"] = gpt4v_response
                    dump_json(eval_file, gpt4v_response)
                except :
                    gpt4v_response["Score"] = None
                    print("retry error")


            if isinstance(gpt4v_response["Score"], str):
                gpt4v_response["Score"] = 0

            if gpt4v_response["Score"] is None:
                gpt4v_response["Score"] = 0

            results[index] = v
            scores += gpt4v_response["Score"]
            

        print(f"mean score: {scores / count}")
        results["score"] = scores / count
        return results

def eval(metric_type="", data_folder=""):

    model = EditActionEvaluator()

    for image_folder in data_folder:
        if metric_type == "alignment":
            results = model.eval_alignment(image_folder)
            output_json = os.path.join(image_folder,"gpt_alignment_results.json")
        elif metric_type == "coherence":
            results = model.eval_coherence(image_folder)
            output_json = os.path.join(image_folder,"gpt_coherence_results.json")
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--metric_type", required=False, type=str, default="coherence")
    parser.add_argument("--data_folder", required=False, type=list, default=["pathto/HQ-Edit-data-demo/data"])


    args = parser.parse_args()
    eval(args.metric_type, args.data_folder)
