import warnings
warnings.filterwarnings("ignore")
from hyperpara_tuning import hyperparameter_tuning
from train_utils import train_model
from evaluate_utils import evaluate
from visualization import visualise_model

def main():
    data_dir = r'/home/yue/Desktop/MSc_Project_for_CS+/DeepLearning/train'
    csv_file = r'/home/yue/Desktop/MSc_Project_for_CS+/DeepLearning/train.csv'
    image_path = r'/home/yue/Desktop/MSc_Project_for_CS+/DeepLearning/train/25.png'

    # Suppressing warnings for cleaner output
    warnings.filterwarnings("ignore")

    input('Press Enter to start')

    # Hyperparameter tuning
    best_trial = hyperparameter_tuning(data_dir, csv_file)
    print('Finished hyperparameter tuning')
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial training loss: {}".format(best_trial.last_result["train_loss"]))
    print("Best trial validation loss: {}".format(best_trial.last_result["val_loss"]))
    print("Best trial validation accuracy: {}".format(best_trial.last_result["val_accuracy"]))
    
    # Training
    input("Press Enter to continue to the next step...")
    model=train_model(best_trial.config, data_dir, csv_file)
    print(model)
    print("Finished training the model.")

    # Evaluation
    input("Press Enter to continue to the next step...")
    evaluate(model, best_trial.config, data_dir, csv_file)
    print("Finished model evaluation.")
    
    # Visualization
    input("Press Enter to continue to the next step...")
    visualise_model(model, image_path)
    print("Finished visualization.")

if __name__ == "__main__":
    main()
