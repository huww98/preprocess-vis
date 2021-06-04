from io import BytesIO
from os import name
from pathlib import Path
from typing import Callable
import uuid
import dataclasses
from PIL import Image
from aiohttp import web
import numpy as np

import methods.dali

@dataclasses.dataclass
class PreprocessMethodDef:
    handler: Callable[..., Callable[[Path], np.ndarray]]
    arguments_html: str = dataclasses.field(default="")
    default_args: dict = dataclasses.field(default_factory=dict)

uploaded_path = Path('uploaded')
preprocess_methods = {
    'dali.color_jitter': PreprocessMethodDef(
        handler=methods.dali.ColorJitter,
        arguments_html="""
<label for="brightness">Brightness<label><input type="number" name="brightness" id="brightness" value="{brightness}" step="0.01" min="0"/>
<label for="contrast">Contrast<label><input type="number" name="contrast" id="contrast" value="{contrast}" step="0.01"  min="0"/>
<label for="saturation">Saturation<label><input type="number" name="saturation" id="saturation" value="{saturation}" step="0.01"/>
<label for="hue">HUE<label><input type="number" name="hue" id="hue" value="{hue}" step="0.01"/>
""",
        default_args={'brightness': 1., 'contrast': 1., 'saturation': 1., 'hue': 0.},
    ),
    'dali.gray_scale': PreprocessMethodDef(handler=methods.dali.GrayScale),
    'dali.solorize': PreprocessMethodDef(
        handler=methods.dali.Solorize,
        arguments_html="""
<label for="threshold">Threshold<label><input type="number" name="threshold" id="threshold" value="{threshold}"/>
""",
        default_args={'threshold': 128},
    ),
    'dali.gaussian_blur': PreprocessMethodDef(
        handler=methods.dali.GaussianBlur,
        arguments_html="""
<label for="sigma">Sigma<label><input type="number" name="sigma" id="sigma" value="{sigma}" step="0.01"/>
""",
        default_args={'sigma': 5.},
    ),
}

routes = web.RouteTableDef()

@routes.get('/')
async def index(request: web.Request):
    return web.Response(text=f'''
<!DOCTYPE html>
<html lang="en">
    <head>
        <title>Preprocess Visualization</title>
    </head>
    <body>
        <form action="{request.app.router['upload_image'].url_for()}" method="post" enctype="multipart/form-data">
            <label for="img">Image</label>
            <input type="file" name="img" id="img"/>
            <button type="submit">Upload</button>
        </form>
    </body>
</html>
''', content_type='text/html')

@routes.post('/images', name='upload_image')
async def upload_image(request: web.Request):
    data = await request.post()
    content = data['img'].file.read()
    img_id = str(uuid.uuid4())
    with (uploaded_path/img_id).open('wb') as f:
        f.write(content)
    raise web.HTTPFound(request.app.router['preprocess_list'].url_for(id=img_id))


@routes.get('/images/{id}', name='preprocess_list')
async def preprocess_list(request: web.Request):
    img_id = request.match_info['id']
    def tags_for(m: str):
        href = request.app.router['preprocess'].url_for(method=m, id=img_id)
        return f'<li><a href="{href}">{m}</a></li>'
    return web.Response(text=f'''
<!DOCTYPE html>
<html lang="en">
    <head>
        <title>Preprocess List</title>
    </head>
    <body>
        <p>
            Avaliable preprocesses:
            <ul>
                {''.join(tags_for(m) for m in preprocess_methods)}
            </ul>
        </p>
    </body>
</html>
''', content_type='text/html')


@routes.get('/images/{id}/preprocess/{method}', name='preprocess')
async def preprocess(request: web.Request):
    img_id = request.match_info['id']
    m = request.match_info['method']
    m_def = preprocess_methods[m]
    m_args = dict(m_def.default_args)
    m_args.update(request.query)
    result_src = request.app.router['preprocess_result'].url_for(method=m, id=img_id).with_query(m_args)
    return web.Response(text=f'''
<!DOCTYPE html>
<html lang="en">
    <head>
        <title>{m}</title>
    </head>
    <body>
        <h1>{m}</h1>
        <form>
            {m_def.arguments_html.format_map(m_args)}
            <button type="submit">Process</button>
        <form/>
        <div>
            <img src="{result_src}"/>
        </div>
    </body>
</html>
''', content_type='text/html')

@routes.get('/images/{id}/preprocess/{method}/result', name='preprocess_result')
async def preprocess_result(request: web.Request):
    img_id = request.match_info['id']
    m = request.match_info['method']
    m_def = preprocess_methods[m]
    p = m_def.handler(**request.query)
    img = p(uploaded_path / img_id)
    img = Image.fromarray(img)
    body = BytesIO()
    img.save(body, 'jpeg')
    return web.Response(body=body.getvalue(), content_type='image/jpeg')


if __name__ == '__main__':
    uploaded_path.mkdir(exist_ok=True)

    app = web.Application()
    app.add_routes(routes)
    web.run_app(app)
