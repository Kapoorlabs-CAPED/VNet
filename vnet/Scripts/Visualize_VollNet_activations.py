#!/usr/bin/env python
# coding: utf-8
import os
import glob
from VizOneat import VizOneat
from oneat.NEATUtils.utils import load_json
event_threshold = 0.9
event_confidence = 0.9
normalize = True
nms_function = 'iou'

imagedir = '/mnt/WorkHorse/Mari_Data_Oneat/raw/gt/'
model_dir = '/mnt/WorkHorse/Mari_Models/Oneat/'
model_name = 'Cellsplitdetectorxenopusvolumecnn_d101'

division_categories_json = model_dir + 'Cellsplitdiamondcategoriesxenopus.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'Cellsplitdiamondcordxenopus.json'
cordconfig = load_json(division_cord_json)

Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)

for imagename in X:
     print(imagename)  
     viz_activations = VizOneat(None, imagename, model_dir, model_name, catconfig = catconfig, cordconfig = cordconfig, oneat_vollnet = True)
     viz_activations.PrepareNet()
     viz_activations.VizVollNet()
     viz_activations.VizActivations()

