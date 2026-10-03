# FIT Studio third-party notices

FIT Studio uses **llmfit** through its public executable JSON interface. It
does not copy the Rust fitting engine, model catalog or web application.
llmfit is optional for a source installation; the Windows build includes the
installed llmfit executable when present in the build environment.

Upstream: https://github.com/AlexsJones/llmfit

The Windows build bundles the Python runtime; its PSF license text accompanies
the build under `licenses/python/`. It also uses PyWebView, pythonnet, clr-loader, psutil, PyYAML,
and their dependencies. Their installed package license metadata is copied
into `dist/FIT-Studio/licenses/` by the build script. The application runs in
the installed Microsoft Edge WebView2 runtime; that runtime is not bundled.

The bundled PyInstaller bootloader license and its distribution exception are
included under `licenses/PyInstaller/`. Runtime DLL notices from the build's
Conda packages are retained under `licenses/runtime-*/`. Only the liblzma
library from XZ is bundled; its license is 0BSD. The proxy-tools 0.1.0 wheel
omits its MIT license file, so the build supplies the upstream file from
https://github.com/jtushman/proxy_tools/blob/master/LICENSE.txt
(Git blob `078411c7399fbc0454f36cbe8c8cdbaaba7ebc95`).

## llmfit — MIT License

Copyright (c) 2026 Alex Jones

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
