# Issues
- [x] Add a new plot to represent the tradeoff between precision and recall

- [ ] CORRECT THE NAMING IN THE CMBINED RESULTS SCRIPT!!!!

- [x] Integrate with the actual inference time results:
    - [ ] See if the results already output a validation metrics
    - [x] adapt mocks for the actual data returned by `ultralytics.utils.benchmarks`
    - [ ] run with and without int8 quantization and half precision

- [x] add mb's, fps and format to the tooltip

- [x] integrate a way to just display top N models by filtering the dataframe before creating the plots

- [x] make legend wrap to next line if it's too long and scrollable if needed (Implemented Top N filtering to solve this)



## Non-urgent issues

- [ ] could be useful to ditch the add_layout for location in the legend, to use the least space consuming possible, and also make it scrollable if needed. 

- [ ] When clicking the legend to remove some of the plotted lines, the tooltip still appears we hovering over it (algthough it's invisible)

- [ ] Scale should auto-adjust dynamically depending on the minimum Y axis value from each series, and also when removing some of the plots by clicking the legend


