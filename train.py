from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

model = YOLO(r'/projects/ultralytics/ultralytics/cfg/models/v8/yolov8-small-object.yaml')  # build from YAML and transfer weights

# Train the model
model.train(data= "/projects/smoking_exp/yolov5_6.1/data/smoke.yaml", device = [2],batch=16)