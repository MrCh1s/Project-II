from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    
    # Passing workers=0 disables multiprocessing, a common source of CUDA errors on Windows
    # due to dataloader memory issues
    result = model.train(
        data=r"e:\OneDrive\Desktop\Project II\data.yaml",
        epochs=3,
        imgsz=640,
        batch=16,
        device="0",
        project='runs_yolo11',     
        name='plate_detection_test_workers0',
        workers=0
    )
    print("Test training with workers=0 completed successfully!")

if __name__ == '__main__':
    main()
