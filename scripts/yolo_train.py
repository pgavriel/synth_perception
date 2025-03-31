from ultralytics import YOLO, settings
from os.path import join, isfile

# https://docs.ultralytics.com/modes/train/#train-settings 
if __name__ == "__main__":

    training_epochs = 100 # 100
    # Print Settings
    data_folder = "/home/csrobot/synth_perception/data"
    settings.update({"datasets_dir": data_folder})
    print(settings)

    # Load a model
    model = YOLO("yolo11n.pt")
    # Display model information (optional)
    model.info()

    # Train the model
    
    dataset_name = "mustard_detec"
    dataset_file = join(data_folder,dataset_name,"data.yaml")
    print("DATA", "FOUND" if isfile(join(dataset_file)) else "NOT FOUND")
    # dataset_file = "../data/test/data.yaml"
    print(f"Attempting to train data: {dataset_file}")
    results = model.train(data=dataset_file, epochs=training_epochs, imgsz=1080,plots=True)

    print("Done")