from ultralytics import YOLO

def study()
  """обучаем YOLO модель на нашем датасете"""
  dataset_path = "data.yaml"
  model = YOLO("yolo11n.pt")
  model.train(data=dataset_path, epochs=150, batch=16, imgsz=640)


def testCase():
  """Проверяем новую модель"""
  model = YOLO("runs/detect/train7/weights/best.pt")
  result = model("DataSet/train/images/1_png.rf.5a8f38f04e9bfa722ef812a1d12ee081.jpg", conf=0.1)
  result[0].show()
