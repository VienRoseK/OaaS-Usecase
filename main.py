import logging
import oaas_sdk_py as oaas
import uvicorn
import aiohttp
import uuid



from ultralytics import YOLO
from fastapi import Request, FastAPI, HTTPException
from oaas_sdk_py import OaasInvocationCtx
import os



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
                    
                model = YOLO("yolov8n.pt")

                # Run batched inference on a list of images
                results = model([tmp_in])  # return a list of Results objects

                # Process results list
                for result in results:
                    boxes = result.boxes  # Boxes object for bounding box outputs
                    masks = result.masks  # Masks object for segmentation masks outputs
                    keypoints = result.keypoints  # Keypoints object for pose outputs
                    probs = result.probs  # Probs object for classification outputs
                    obb = result.obb  # Oriented boxes object for OBB outputs
                    result.show()  # display to screen
                    result.save(filename="result.jpg")  # save to disk
                
                

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
