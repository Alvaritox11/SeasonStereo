# SeasonStereo Project Page

Static project page for the SeasonStereo paper. The page is designed to work on
GitHub Pages without a build step.

## Files

- `index.html`: paper-style page structure with abstract, overview, synthetic examples, interactive DSM mesh, results, release, and citation sections.
- `styles.css`: responsive visual design.
- `viewer.js`: interactive Three.js mesh viewer using lightweight untextured Omaha disparity and DSM GLB meshes.
- `assets/`: lightweight release figures used by the page.
- `scripts/export_dsm_mesh.py`: utility used to export DSM GeoTIFFs to decimated untextured GLB meshes.
- `scripts/export_disparity_mesh.py`: utility used to export `.iio` disparity maps to decimated untextured GLB surfaces.

## Teaser Figure

The hero section is already wired to load the paper teaser from:

```text
assets/teaser.png
```

Place the final teaser there when it is ready. Until that file exists, the page
shows a small placeholder in the hero.

## Local Preview

The page imports Three.js from a CDN. You can usually open `index.html`
directly in a browser. A local server is still useful if you want to preview it
exactly like GitHub Pages:

```bash
python -m http.server 8000 --directory repo_release/others/project_page
```

Then open:

```text
http://localhost:8000
```

## Before Publishing

- Add final author list, venue, paper URL, code URL, dataset URL, checkpoint URL, and BibTeX.
- Replace or refine the lightweight Omaha disparity/DSM GLBs if you want a higher-fidelity viewer.
- Decide whether the page will live under `docs/`, a `gh-pages` branch, or another GitHub Pages source.
- Keep `assets/` lightweight. Move heavy videos, meshes, and model files to Hugging Face or another release asset host.
