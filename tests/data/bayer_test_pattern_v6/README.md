# Bayer test pattern v6 conformance images

Synthetic Bayer-pattern conformance charts created by Han Kleijn (hnsky.org) and
"kindly offered to the community to help other developers to implement these
extensions in their file input/output" — the `ROWORDER` / `XBAYROFF` /
`YBAYROFF` FITS keywords. They are the de-facto ground truth the demosaicing
implementations (Siril, ASTAP, N.I.N.A.) validate against; bandaid's
`generate_bayer_masks` tests pin the channel masks to them pixel-for-pixel.

- Source: <https://free-astro.org/index.php?title=File:Bayer_test_pattern_v6.tar.gz>
    (download linked from the Siril FITS-orientation documentation).
- License: Public Domain, per the hosting wiki's content license ("Content is
    available under Public Domain unless otherwise noted"; the file page carries
    no other notice).
- Local changes: the eight files covering every `ROWORDER` × `XBAYROFF` ×
    `YBAYROFF` combination were renamed and gzip-compressed, unmodified
    otherwise. The archive's ninth file (which carries no `ROWORDER` keyword) is
    omitted.
