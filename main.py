import logging
import oaas_sdk_py as oaas
import uvicorn
import aiohttp
import uuid



from yolov5 import detect
from fastapi import Request, FastAPI, HTTPException
from oaas_sdk_py import OaasInvocationCtx
import os

# Class name dictionary for analyzing the objects 
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
    7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
    13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
    21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
    28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
    53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

IMAGE_KEY = os.getenv("IMAGE_KEY", "image")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
level = logging.getLevelName(LOG_LEVEL)
logging.basicConfig(level=level)


async def write_to_file(resp, file_path):
    with open(file_path, "wb") as f:
        async for chunk in resp.content.iter_chunked(1024):
            f.write(chunk)


class image_Handler(oaas.Handler):
    async def handle(self, ctx: OaasInvocationCtx):

        fmt = ctx.task.main_obj.data.get('format', 'jpg')
        tmp_in = f"in-{uuid.uuid4()}.{fmt}"
      
        
        try:
            async with aiohttp.ClientSession() as session:
                async with await ctx.load_main_file(session, IMAGE_KEY) as resp:
                    await write_to_file(resp, tmp_in)
                    

                # Run YOLOv5 detection
                img_name = detect.run(source=tmp_in, project='runs/detect', name='exp', exist_ok=True, save_txt=True) 
                

                # Extract the image file and the label file after object detection
                base_name = os.path.basename(img_name)
                detect_img_file = os.path.splitext(base_name)[0] + '.jpg'
                label_txt_file = os.path.splitext(base_name)[0] + '.txt'
                
                label_txt_file_path = os.path.join('runs', 'detect', 'exp', 'labels', label_txt_file)
               
                # Modify the label text file to analyze the information of the objects 
                # detected from the image
                with open(label_txt_file_path, "r") as label_file:
                    lines = label_file.readlines()

                modified_lines = []
                object_counts = {name: 0 for name in COCO_CLASSES.values()}
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    object_name = COCO_CLASSES[class_id]
                    object_counts[object_name] += 1

                    # Add explanations for the bounding box coordinates
                    modified_line = f"{object_name} (class id {class_id}) center_x: {parts[1]}, center_y: {parts[2]}, width: {parts[3]}, height: {parts[4]}"
                    modified_lines.append(modified_line)

                with open("label_analysis.txt", "w") as label_output:
                    for line in modified_lines:
                        label_output.write(line + "\n")

                objects_with_counts = [{"object_name": name, "count": count} for name, count in object_counts.items() if count > 0]

                ctx.resp_body = { "image": detect_img_file, 
                                  "label": label_txt_file,
                                  "objects_with_counts": objects_with_counts
                                }
                

        finally:
            if os.path.isfile(tmp_in):
                os.remove(tmp_in)


app = FastAPI()
router = oaas.Router()
router.register(image_Handler())


@app.post('/')
async def handle(request: Request):
    body = await request.json()
    logging.debug("request %s", body)
    resp = await router.handle_task(body)
    logging.debug("completion %s", resp)
    if resp is None:
        raise HTTPException(status_code=404, detail="No handler matched")
    return resp

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
