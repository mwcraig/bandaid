# Why bandaid measures at Gaia positions

bandaid does *forced* photometry: it projects a magnitude-limited Gaia catalog
into each frame's pixel coordinates and measures every catalog star, instead of
detecting sources in the image. The reasons:

- **Stable star identity.** A catalog row is the same star in every frame, so a
    light curve is a column of the output, not the result of matching detections
    across frames.
- **No detection threshold.** A star is measured whether or not it would have
    been detected on that frame. Detection is used only to estimate the FWHM and to
    solve the WCS.
- **Faint stars are measured consistently.** A star near the detection limit
    does not appear and vanish from frame to frame; it is always measured, and
    its signal-to-noise says how good the measurement is. Positions for faint
    stars are modelled rather than centroided
    ([how star positions are chosen](star_positions.md)).
- **Comparable star lists.** The list comes from the catalog, not the frame, so
    runs on different nights or telescopes cover the same stars and can be
    compared directly.

The cost is that the result depends on a correct WCS solve, which is why a frame
with a wrong-scale or mispointed solution is skipped instead of measured.
