from ultralytics import YOLO, settings
from os.path import join, isfile
import argparse

# https://docs.ultralytics.com/modes/train/#train-settings 
if __name__ == "__main__":

    dataset_name = "atb1_test2"
    parser = argparse.ArgumentParser(description='Process one primary argument and a list of secondary arguments.')
    parser.add_argument('-n','--name', type=str, default=dataset_name,
                        help='Name of the dataset folder to use for training.')
    parser.add_argument('-e','--epochs', type=int, default=100,
                        help='Number of epochs to train for.')
    args = parser.parse_args()
    dataset_name = args.name

    training_epochs = args.epochs # 100
    # Print Settings
    data_folder = "/home/csrobot/synth_perception/data"
    settings.update({"datasets_dir": data_folder})
    print(settings)

    # Load a model
    model = YOLO("yolo11n.pt")
    # Display model information (optional)
    model.info()

    # Train the model
    
    dataset_file = join(data_folder,dataset_name,"data.yaml")
    print("DATA", "FOUND" if isfile(join(dataset_file)) else "NOT FOUND")
    # dataset_file = "../data/test/data.yaml"
    print(f"Attempting to train data: {dataset_file}")
    results = model.train(data=dataset_file, epochs=training_epochs, imgsz=1080,plots=True,name=dataset_name)

    print("Done")