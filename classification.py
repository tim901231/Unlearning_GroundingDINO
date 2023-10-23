from groundingdino.util.inference import load_model, load_image, predict, annotate
from tqdm import tqdm
import cv2
import os, argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'ImageClassification',
                    description = 'Takes the path of images and generates classification scores')

    parser.add_argument('--folder', help='path to images', type=str, required=True)
    parser.add_argument('--output', help='path to output', type=str, required=True)
    parser.add_argument('--prompt', help='path to prompt', type=str, required=True)
    parser.add_argument('--target_class', help='target class', type=str, required=True)
    args = parser.parse_args()
    folder_path = args.folder
    output_path = args.output
    prompt_path = args.prompt
    target_class = args.target_class

    os.makedirs(output_path, exist_ok=True)

    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    TEXT_PROMPT = "airplane . automobile . bird . cat . deer . dog . frog . horse . ship . truck"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    prompt = pd.read_csv(prompt_path)
    print(prompt)
    # print(sorted(os.listdir(folder_path)))
    T = len(prompt['case_number'])
    pred = []
    for i in tqdm(range(T)):
        path = f"{prompt['case_number'][i]}_0.png"
        IMAGE_PATH = os.path.join(folder_path, path)

        image_source, image = load_image(IMAGE_PATH)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        image_class = prompt['class'][i]
        su = False
        for s in phrases:
            if image_class in s:
                pred.append(image_class)
                su = True
                break
        if not su:
            pred.append("fail")
        
        
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        # cv2.imwrite(os.path.join(output_path, path), annotated_frame)
    
    prompt['category_top1'] = pred
    prompt.to_csv(os.path.join(output_path, f"{target_class}.csv"))