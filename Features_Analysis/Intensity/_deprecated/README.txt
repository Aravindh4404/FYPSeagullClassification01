DEPRECATED FILE: Real_Dark_Wingtip_Pixel_Analysis.py
=====================================================

Original location: Features_Analysis\Intensity\Real_Dark_Wingtip_Pixel_Analysis.py
Moved here on:      2026-07-02
Reason:             broken placeholder, confirmed dead code.

This copy's extract_region(image, segmentation, region_type) function
(originally around lines 47-62) is a stub that always returns
np.zeros(...) masks for both "wing" and "wingtip" regions, regardless of
input:

    def extract_region(image, segmentation, region_type):
        # This function should match your existing extract_region function
        # For now, I'll provide a placeholder - you'll need to replace this
        # with your actual implementation
        ...
        wing_mask = np.zeros(segmentation.shape[:2], dtype=np.uint8)
        ...
        wingtip_mask = np.zeros(segmentation.shape[:2], dtype=np.uint8)

This local definition shadows the real, working extract_region() pulled
in earlier via `from Features_Analysis.config import *`. Any call to
extract_region() in this file therefore always received empty masks,
which means wing_pixels/wingtip_pixels were always empty and the
pipeline hit its `if len(wing_pixels) == 0: return None` guard on every
single image. This script could not have produced any real numeric
output, ever.

It is NOT the source of dark_pixel_results_all_images.csv or any other
result used in the paper.

The real, working version — confirmed as the sole source of the
authoritative dark_pixel_results_all_images.csv (by grepping the repo
for output column names unique to its results dict, e.g.
dark_cluster_center_method3 / white_cluster_center_method3, which only
appear in this one script) — lives at:

    Features_Analysis\Intensity\Data_Analysis\Real_Dark_Wingtip_Pixel_Analysis.py

Use that one. This deprecated copy is kept here only for repo history /
paper-trail purposes, not for use.
