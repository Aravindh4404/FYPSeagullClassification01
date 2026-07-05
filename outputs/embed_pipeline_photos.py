"""
Task: embed real GWG/SBG photos into the pipeline schematic (pipeline_figure.svg),
replacing the two dashed placeholder boxes, then rasterize to a 300 DPI PNG.

Source SVG: Report_Images/pipeline_figure.svg (1200x900 user units == 12.5in x 9.375in
at the standard 96 CSS px/inch, i.e. the "~12x9in" hi-res export size).

No native cairo/rsvg library is available in this environment, so rendering is done via
headless Microsoft Edge (Chromium), which ships with Windows and renders SVG faithfully.
"""
import base64
import subprocess
from pathlib import Path
from PIL import Image

REPO = Path(r"D:\FYPSeagullClassification01")
SVG_IN = REPO / "Report_Images" / "pipeline_figure.svg"
OUT_DIR = REPO / "outputs"
SVG_OUT = OUT_DIR / "pipeline_figure_with_photos.svg"
PNG_OUT = OUT_DIR / "pipeline_figure_with_photos.png"

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# Same birds used in task1_gradcam_comparison.png, for cross-figure consistency.
GWG_SRC = REPO / "Test_Results" / "Test_Data" / "Glaucous_Winged_Gull" / "0Z2A0606.JPG"
SBG_SRC = REPO / "Test_Results" / "Test_Data" / "Slaty_Backed_Gull" / "mod-0R8A5217.JPG"

# Placeholder box geometry from the SVG (x, y, width, height)
GWG_BOX = (430, 80, 150, 112)
SBG_BOX = (620, 80, 150, 112)

# Render the 1200x900 SVG canvas (12.5in x 9.375in @ 96 CSS px/in) at 300 DPI.
SCALE = 300 / 96  # 3.125
SVG_W, SVG_H = 1200, 1120
PNG_W, PNG_H = round(SVG_W * SCALE), round(SVG_H * SCALE)


def center_crop_to_box(src_path: Path, box_w: int, box_h: int, upscale: int = 8) -> str:
    """Center-crop (no distortion) to the box aspect ratio, return a base64 PNG data URI.
    Rendered at `upscale`x the box's pixel size so it stays sharp after the SVG->PNG
    rasterization scales it back up to 300 DPI."""
    im = Image.open(src_path).convert("RGB")
    w, h = im.size
    target_ratio = box_w / box_h
    src_ratio = w / h
    if src_ratio > target_ratio:
        # image relatively wider than target -> crop left/right
        new_w = round(h * target_ratio)
        left = (w - new_w) // 2
        im = im.crop((left, 0, left + new_w, h))
    else:
        # image relatively taller than target -> crop top/bottom
        new_h = round(w / target_ratio)
        top = (h - new_h) // 2
        im = im.crop((0, top, w, top + new_h))
    im = im.resize((box_w * upscale, box_h * upscale), Image.LANCZOS)
    buf_path = OUT_DIR / f"_tmp_{src_path.stem}.png"
    im.save(buf_path, "PNG")
    data = base64.b64encode(buf_path.read_bytes()).decode("ascii")
    buf_path.unlink()
    return f"data:image/png;base64,{data}"


def make_image_tag(box, data_uri):
    x, y, w, h = box
    return (f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
            f'href="{data_uri}" preserveAspectRatio="none" '
            f'clip-path="url(#clip-{x}-{y})"/>')


def build_svg():
    svg = SVG_IN.read_text(encoding="utf-8")

    gwg_uri = center_crop_to_box(GWG_SRC, *GWG_BOX[2:])
    sbg_uri = center_crop_to_box(SBG_SRC, *SBG_BOX[2:])

    # Add rounded-rect clip paths matching the original placeholder boxes (rx=6)
    clip_defs = ""
    for (x, y, w, h) in (GWG_BOX, SBG_BOX):
        clip_defs += (f'<clipPath id="clip-{x}-{y}"><rect x="{x}" y="{y}" '
                       f'width="{w}" height="{h}" rx="6"/></clipPath>\n')
    svg = svg.replace("</defs>", clip_defs + "</defs>")

    # GWG: replace the placeholder rect + "insert photo" text with the photo.
    old_gwg = ('<rect x="430" y="80" width="150" height="112" rx="6" fill="#f1efe8" '
               'stroke="#b4b2a9" stroke-width="1.2" stroke-dasharray="5 4"/>\n'
               '<text x="505" y="128" text-anchor="middle" font-size="12" fill="#888780">insert photo</text>')
    new_gwg = make_image_tag(GWG_BOX, gwg_uri)
    assert old_gwg in svg, "GWG placeholder block not found verbatim"
    svg = svg.replace(old_gwg, new_gwg)

    old_sbg = ('<rect x="620" y="80" width="150" height="112" rx="6" fill="#f1efe8" '
               'stroke="#b4b2a9" stroke-width="1.2" stroke-dasharray="5 4"/>\n'
               '<text x="695" y="128" text-anchor="middle" font-size="12" fill="#888780">insert photo</text>')
    new_sbg = make_image_tag(SBG_BOX, sbg_uri)
    assert old_sbg in svg, "SBG placeholder block not found verbatim"
    svg = svg.replace(old_sbg, new_sbg)

    # Thin border around each embedded photo (visually replaces the dashed placeholder outline)
    svg = svg.replace(new_gwg, new_gwg + f'\n<rect x="{GWG_BOX[0]}" y="{GWG_BOX[1]}" width="{GWG_BOX[2]}" height="{GWG_BOX[3]}" rx="6" fill="none" stroke="#b4b2a9" stroke-width="1.2"/>')
    svg = svg.replace(new_sbg, new_sbg + f'\n<rect x="{SBG_BOX[0]}" y="{SBG_BOX[1]}" width="{SBG_BOX[2]}" height="{SBG_BOX[3]}" rx="6" fill="none" stroke="#b4b2a9" stroke-width="1.2"/>')

    SVG_OUT.write_text(svg, encoding="utf-8")
    print(f"Saved: {SVG_OUT}")


def render_png():
    cmd = [
        EDGE, "--headless", "--disable-gpu", "--no-sandbox",
        f"--screenshot={PNG_OUT}",
        f"--window-size={SVG_W},{SVG_H}",
        f"--force-device-scale-factor={SCALE}",
        "--default-background-color=FFFFFFFF",
        SVG_OUT.as_uri(),
    ]
    subprocess.run(cmd, check=True)
    im = Image.open(PNG_OUT)
    print(f"Rendered: {PNG_OUT} size={im.size} (target {PNG_W}x{PNG_H})")


if __name__ == "__main__":
    build_svg()
    render_png()
