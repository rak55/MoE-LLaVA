import torch
import os
import json
from PIL import Image
from tqdm import tqdm
import argparse
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex
                
def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def main(args):
    disable_torch_init()

    #model_path = 'LanguageBind/MoE-LLaVA-xxxxxxxxxxxxxxxx'  # choose a model
    device = 'cuda'
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, None, model_name, args.load_8bit, args.load_4bit, device=device)
    
    dataset = list(read_jsonl(args.dataset)) 

    demos = list(read_jsonl(args.demos))
    demos = demos[: args.num_demos]
    ex_demos = []
    for d_item in demos:
                    
        dlabels=[]                                               #get labels.
                
        if d_item["shaming"]==1:
            dlabels.append("shaming")
        if d_item["stereotype"]==1:
            dlabels.append("stereotype")
        if d_item["objectification"]==1:
            dlabels.append("objectification")
        if d_item["violence"]==1:
            dlabels.append("violence")
        ex_demos.append(
            {
                "image": load_image(os.path.join(args.images_path, d_item["file_name"])),
                "question": "What does the man in the meme feel entitled to?",                          #again same question for every demo.
                "answer": d_item["Entitlement"],
                "rationale": d_item["Entitlement Reason"],
                "text": d_item["Text"],
                "labels": dlabels,                               
            }
        )

    answers_file = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    seen_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r") as f:
            for line in f:
                line = json.loads(line)
                seen_ids.add(line["file_name"])

    r_prompt = "Explain why your answer is correct in great detail, referencing the provided image. Think step-by-step, and make sure to only draw conclusions from evidence present in the provided image."
    a_prompt = "What is the final answer to the question? Be short and concise."

    def add_r_turn(conv, text, labels, question: str, rationale: str | None = None):
        labelh= "The meme is classified among the following categories:"
        h=" ".join(labels)
        label=labelh + " " +h + "."
        qs = f"{text} {label} {question} {r_prompt}"
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv.append_message(conv.roles[0], qs)
        if rationale is not None:
            conv.append_message(
                conv.roles[1],
                rationale + "\n",
            )
        else:
            conv.append_message(
                conv.roles[1],
                None,
            )
        prompt = conv.get_prompt()
        print(prompt)
        

    def add_a_turn(conv, answer: str | None = None):
        qs = a_prompt

        conv.append_message(conv.roles[0], qs)
        if answer is not None:
            conv.append_message(
                conv.roles[1],
                answer + "\n",
            )
        else:
            conv.append_message(
                conv.roles[1],
                None,
            )

    def run(conv, images):
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        input_token_len = input_ids.shape[1]
        print(input_token_len)
        #assert input_ids.dtype==torch.long, "type is incorrect"
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = model.generate(                                                                      #is images image_tensor here?
            input_ids,
            images=images,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria],
            use_cache=True
          )
        #n_diff_input_output = ((input_ids != output_ids[:, :input_token_len]).sum().item())
        #if n_diff_input_output > 0:
        #print(f"[Warning] Sample {idx}: {n_diff_input_output} output_ids are not the same as the input_ids")
        
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(outputs)
        return outputs

    #continue from here.
    image_processor = processor['image']
    for idx in tqdm(range(len(dataset))):
        if dataset[idx]["file_name"] in seen_ids:
            continue
        ex = dataset[idx]
        image_path = ex["file_name"]                 #i think we have to load the actual image.
        image = load_image(os.path.join(args.images_path, image_path))
        question = "What does the man in the meme feel entitled to?"              #same question for every ex. change according to output expected here.
        text=ex["Text"]
        
        labels=[]                                               #get labels.
            
        if ex["shaming"]==1:
            labels.append("shaming")
        if ex["stereotype"]==1:
            labels.append("stereotype")
        if ex["objectification"]==1:
            labels.append("objectification")
        if ex["violence"]==1:
            labels.append("violence")
        
        image_tensor = image_processor.preprocess(
            [d["image"] for d in ex_demos] + [image], return_tensors="pt"
        )["pixel_values"]
        # removed .unsqueeze(0)
        images = image_tensor.half().cuda()
        conv = conv_templates["multimodal"].copy()

        for d in ex_demos:
            add_r_turn(
                conv,text=d["text"],labels=d["labels"],
                question=d["question"],
                rationale=d["rationale"],            #changed near demos. 
            )
            add_a_turn(
                conv,
                answer=d["answer"],                   #change category wrt output.
            )

        final_conv = conv.copy()

        add_r_turn(
            conv,text=text,
            labels=labels,
            question=question,
            
        )

        rationale = run(conv, images)

        add_r_turn(
            final_conv,
            text=text,
            labels=labels,
            question=question,
            rationale=rationale,
                                       #shud we add text and labels twice?
        )
        full_conv = final_conv.copy()
        add_a_turn(final_conv)

        pred = run(final_conv, images)
        add_a_turn(full_conv, answer=pred)

        with open(answers_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "id": ex["file_name"],
                        "rationale": rationale,
                        "pred": pred,
                    }
                )
                + "\n"
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--demos", type=str, required=True)
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--num_demos", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="flaviagiammarino/vqa-rad")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-bf16", action="store_true")
    
    args = parser.parse_args()
    
    main(args)
    '''
    deepspeed predict.py
    '''
