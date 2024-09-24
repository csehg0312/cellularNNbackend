import asyncio
from aiohttp import web
from aiohttp import ClientSession
import random
from pathlib import Path
import numpy as np
import cv2
from worker import cnnCall
import base64
from datatobinary import save_saved

# Assuming dist_path is defined elsewhere in your code
dist_path = Path(__file__).parent / "dist"

# List of backend servers
BACKENDS = ['http://localhost:8001', 'http://localhost:8002', 'http://localhost:8003']

async def load_balancer(request):
    # Simple round-robin load balancing
    backend = random.choice(BACKENDS)
    async with ClientSession() as session:
        async with session.request(
            method=request.method,
            url=f"{backend}{request.path}",
            headers=request.headers,
            data=await request.read()
        ) as resp:
            return web.Response(
                body=await resp.read(),
                status=resp.status,
                headers=resp.headers
            )

import gc

async def handle_upload(request):
    try:
        # Reading the multipart data
        reader = await request.multipart()
        field = await reader.next()

        # Validate uploaded file
        if field.name != 'file':
            return web.Response(status=400, text="No file uploaded")

        print("Image received")

        # Read the raw binary data from the field in one go
        data = await field.read(decode=True)

        # Base64 decoding (optimized, avoid intermediate string creation)
        try:
            if data.startswith(b'data:image'):
                base64_start = data.find(b',') + 1
                image_data = base64.b64decode(data[base64_start:])  # Directly decode from bytes
            else:
                return web.Response(status=400, text="Invalid image format")

            # Decode the binary data into a numpy array
            np_array = np.frombuffer(image_data, np.uint8)

            # Decode the numpy array into an image
            image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

            if image is None:
                return web.Response(status=400, text="Invalid image file")
            
            print("Image successfully decoded")

        finally:
            # Free memory of intermediate variables
            del data, image_data, np_array
            gc.collect()

        # Process additional fields in the form
        fields = {}
        while True:
            field = await reader.next()
            if field is None:
                break
            if field.name in ['keep_original_size', 'size', 'invert_size', 'mode']:
                value = await field.read(decode=True)
                fields[field.name] = value.decode('utf-8')

        # Handle image resizing logic if needed
        keep_original_size = fields.get('keep_original_size', 'true').lower()
        selected_size = fields.get('size')
        invert_size = fields.get('invert_size', 'false').lower()

        if keep_original_size == 'false' and selected_size:
            width, height = map(int, selected_size.split('x'))
            if invert_size == 'true':
                width, height = height, width  # Swap width and height
            image = cv2.resize(image, (width, height))

        # Perform image processing (calls a Julia ODE solver)
        processed_image = cnnCall(image, fields.get('mode', 'default_mode_'))

        # Encode the processed image back to PNG format
        success, encoded_image = cv2.imencode('.png', processed_image)
        if not success:
            return web.Response(status=500, text="Failed to encode image")

        print("Image successfully processed and encoded")

        # Cleanup processed image
        del processed_image, image
        gc.collect()

        # Return the processed image as a response
        return web.Response(body=encoded_image.tobytes(), content_type='image/png')

    except Exception as e:
        print(f"Error: {str(e)}")
        return web.Response(status=500, text="Internal server error")
    
async def handle_stx_save(request):
    try:
        # Reading the multipart data
        data = await request.json()
        
        arr_1d_ctrl = np.array(data['ctrl'])
        arr_1d_fdb = np.array(data['fdb'])
        radius = data['radius']
        size = int(radius) * 2 + 1
        print(data['bias'])
        arr_2d_ctrl = arr_1d_ctrl[:size*size].reshape(size, size)
        arr_2d_fdb = arr_1d_fdb[:size*size].reshape(size, size)
        t_span = np.linspace(0, int(data['tspan']), int(data['steps']))
   
        print(f"A: {arr_2d_ctrl}, B: {arr_2d_fdb}, t: {t_span}, Ib: {data['bias']}, init: {data['initial']}")
        message = save_saved(A=arr_2d_fdb, B=arr_2d_ctrl, t=t_span, Ib=data['bias'], init=data['initial'])
        return web.Response(body="message", content_type="text")
    except Exception as e:
        print(f"Error: {str(e)}")
        return web.Response(status=500, text="Internal server error")


async def start_backend(port):
    app = web.Application()
    app.router.add_post('/upload', handle_upload)
    
    # The webrtc functionality is currently turned off 
    # app.router.add_post('/offer', handle_offer)
    # app.on_shutdown.append(on_shutdown)
    
    app.router.add_post('/stx', handle_stx_save)
    app['pcs'] = set()

    # Serve static assets from dist/assets
    app.router.add_static('/assets/', path=dist_path / 'assets', name='assets')


    # Serve the main index.html
    async def handle_index(request):
        return web.FileResponse(dist_path / "index.html")

    app.router.add_get('/', handle_index)

    # Start frame processing
    # asyncio.create_task(process_frames())

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', port)
    await site.start()

async def main():
    # Start backend servers
    for i, port in enumerate([8001, 8002, 8003]):
        asyncio.create_task(start_backend(port))
        print(f"Backend server {i+1} started on port {port}")

    # Start load balancer
    load_balancer_app = web.Application()
    load_balancer_app.router.add_route('*', '/{path:.*}', load_balancer)

    runner = web.AppRunner(load_balancer_app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8000)
    await site.start()
    print("Load balancer started on port 8000")

    # Keep the main coroutine running
    while True:
        await asyncio.sleep(3600)

if __name__ == '__main__':
    asyncio.run(main())