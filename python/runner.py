"""在临时工作目录执行源码，将结果交给 Rust；不提供 HTTP 服务。"""
from __future__ import annotations

from importlib.metadata import version
import json
import os
from pathlib import Path
import runpy
import sys
import traceback
from typing import Any

def versions() -> dict[str, str]:
    return {name: version(name) for name in ('jax', 'jaxlib', 'libtpu', 'tpuasm')}

def main() -> None:
    if sys.argv[1:] == ['--info']:
        print(json.dumps(versions()))
        return
    source = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    os.environ['HOATA_SOURCE'] = str(source)
    import hoata
    result: dict[str, Any]
    try:
        runpy.run_path(str(source), run_name='__main__')
        if hoata._result is None:
            raise ValueError('No compilation requested. Add hoata.compile(function, *abstract_inputs) to your script.')
        result = {'ok': True, **hoata._result, 'versions': versions(), 'diagnostics': []}
    except BaseException as error:
        frames = traceback.extract_tb(error.__traceback__)
        location = next((frame for frame in reversed(frames) if frame.filename == str(source)), None)
        line = error.lineno if isinstance(error, SyntaxError) and error.filename == str(source) else location.lineno if location else None
        message = ''.join(traceback.format_exception_only(type(error), error)).strip().replace(str(source), 'kernel.py')
        result = {'ok': False, 'programs': [], 'diagnostics': [{'message': message, 'line': line}], 'traceback': traceback.format_exc().replace(str(source), 'kernel.py')}
    output.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')

if __name__ == '__main__':
    main()
