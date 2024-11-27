# unsupervised-plant-search
Unsupervised computer vision to find target plants from a geo-referenced orthophoto


## Future Improvements
- [ ] Save to file needs to open a dialogue box, not save to server
- [ ] Optimize callbacks
    - Precompute & cache computations
- [ ] Debounce & delay callbacks
    - Don't update image until range has been untouched ~300ms or so
- [ ] WebGL rendering
- [ ] Lazy loading of image tiles
- [ ] Auto-select best range for index

## Misc. Testing Notes

To upload huge GeoTiff files: 

`bokeh serve --show bokeh-tif.py --websocket-max-message-size=500000000`

Not entirely sure what the default set limit is, but it is much too small for the larger file uploads. We may need to adjust server configs on deploy.




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