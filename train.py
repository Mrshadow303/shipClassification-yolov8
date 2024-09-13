from ultralytics import YOLO

# 训练模型
model = YOLO('E:/myfile/ship-classification-yolo/ultralytics-main-yolov8/runs/detect/train/weights/best.pt')
model.train(data='E:/myfile/ship-classification-yolo/ultralytics-main-yolov8/datasets/yolov8.yaml', epochs=1, imgsz=640, batch=16, device=0)
