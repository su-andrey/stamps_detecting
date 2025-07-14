# shape of input image to YOLO
W, H = 448, 448
# grid size after last convolutional layer of YOLO
S = 7 
# anchors of YOLO model
ANCHORS = [[1.5340836003942058, 1.258424277571925],
 [1.4957766780406023, 2.2319885681948217],
 [1.2508985343739407, 0.8233350471152914]]
# number of anchors boxes
BOX = len(ANCHORS)
# minimal confidence of presence a stamp in the grid cell
OUTPUT_THRESH = 0.7
# maximal iou score to consider boxes different
IOU_THRESH = 0.3
