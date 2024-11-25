# unsupervised-plant-search
Unsupervised computer vision to find target plants from a geo-referenced orthophoto


## Future Improvements
- [ ] Optimize callbacks
    - Precompute & cache computations
- [ ] Debounce & delay callbacks
    - Don't update image until range has been untouched ~300ms or so
- [ ] WebGL rendering
- [ ] Lazy loading of image tiles
- [ ] Auto-select best range for index

## Misc. Testing Notes

Actually usable files:
- spectrum-visualization.py
    - Opens tab in browser, interactive RBG histogram
- streamlit-rgb.py
    - Can filter out RGB spectrum via browser



Useless, mostly:
- find-plants.py
- segmentation-test.py
- dash_app.py
- bokeh_app.py
    - Entirely useless, actually.