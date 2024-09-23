import asyncio
from aiohttp import web
from aiohttp import ClientSession
import random
from pathlib import Path

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

async def start_backend(port):
    app = web.Application()
    app.router.add_post('/upload', handle_upload)
    app.router.add_post('/offer', handle_offer)
    app.router.add_post('/stx', handle_stx_save)
    app['pcs'] = set()

    # Serve static assets from dist/assets
    app.router.add_static('/assets/', path=dist_path / 'assets', name='assets')

    app.on_shutdown.append(on_shutdown)

    # Serve the main index.html
    async def handle_index(request):
        return web.FileResponse(dist_path / "index.html")

    app.router.add_get('/', handle_index)

    # Start frame processing
    asyncio.create_task(process_frames())

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