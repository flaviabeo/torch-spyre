"""Render every <svg> figure in index.html to _light_N.png and _dark_N.png.

The page styles its SVGs with CSS classes and custom properties, which an
SVG rasterizer cannot resolve. So for each theme this script reads the CSS
variables from index.html, substitutes them into the SVG rules, inlines the
result as a <style> element in each SVG, and rasterizes it with resvg at 2x
(1920 px for the 960-wide viewBox) on the figure's surface colour.

Fonts: IBM Plex Sans / Mono TTFs are fetched once into fonts/ and loaded
explicitly, with system fonts skipped, so the output does not depend on the
machine.

Needs: pip install resvg-py   (pure wheel, no system libraries)
Usage: python render_png.py [index.html]
"""

import re
import sys
import urllib.request
from pathlib import Path

import resvg_py

HERE = Path(__file__).parent
FONTS = HERE / "fonts"
FONT_URLS = {
    "IBMPlexSans-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/ibmplexsans/IBMPlexSans%5Bwdth%2Cwght%5D.ttf",
    "IBMPlexMono-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/ibmplexmono/IBMPlexMono-Regular.ttf",
    "IBMPlexMono-Medium.ttf": "https://github.com/google/fonts/raw/main/ofl/ibmplexmono/IBMPlexMono-Medium.ttf",
    "IBMPlexMono-SemiBold.ttf": "https://github.com/google/fonts/raw/main/ofl/ibmplexmono/IBMPlexMono-SemiBold.ttf",
}
SCALE = 2


def fetch_fonts():
    FONTS.mkdir(exist_ok=True)
    for name, url in FONT_URLS.items():
        dst = FONTS / name
        if not dst.exists():
            print("fetching", name)
            urllib.request.urlretrieve(url, dst)
    return [str(p) for p in sorted(FONTS.glob("*.ttf"))]


def parse_vars(block):
    return dict(re.findall(r"--([\w-]+):\s*([^;]+);", block))


def themes(css):
    light = parse_vars(css[css.index(":root {") : css.index("}", css.index(":root {"))])
    i = css.index(':root[data-theme="dark"]')
    dark = dict(light, **parse_vars(css[i : css.index("}", i)]))
    return {"light": light, "dark": dark}


def svg_css(css, var):
    # keep the rules that style SVG elements: from the "SVG roles" comment up to
    # the first HTML-only rule after it
    a = css.index("/* SVG roles */")
    b = css.index("ol.steps", a)
    rules = css[a:b]
    rules = re.sub(r"var\(--([\w-]+)\)", lambda m: var[m.group(1)].strip(), rules)
    # resvg does not support paint-order on text; the stroke halo would then be
    # painted over the glyphs. Drop the halo instead.
    rules = re.sub(
        r"paint-order:[^;]+;\s*stroke:[^;]+;\s*stroke-width:[^;]+;\s*stroke-linejoin:[^;]+;",
        "",
        rules,
    )
    return rules


def main():
    page = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "index.html"
    html = page.read_text()
    css = html[html.index("<style>") + 7 : html.index("</style>")]
    svgs = re.findall(r"<svg\b.*?</svg>", html, flags=re.S)
    fonts = fetch_fonts()
    for theme, var in themes(css).items():
        style = f"<style>{svg_css(css, var)}</style>"
        for n, svg in enumerate(svgs):
            w = float(
                re.search(r'viewBox="[\d.\s-]+?\s([\d.]+)\s[\d.]+"', svg).group(1)
            )
            s = re.sub(
                r"<svg\b", '<svg xmlns="http://www.w3.org/2000/svg"', svg, count=1
            )
            s = s.replace(">", ">" + style, 1)
            png = resvg_py.svg_to_bytes(
                svg_string=s,
                width=int(w * SCALE),
                background=var["surface"].strip(),
                skip_system_fonts=True,
                font_files=fonts,
                sans_serif_family="IBM Plex Sans",
                monospace_family="IBM Plex Mono",
                font_family="IBM Plex Sans",
            )
            out = HERE / f"_{theme}_{n}.png"
            out.write_bytes(bytes(png))
            print("wrote", out.name)


if __name__ == "__main__":
    main()
