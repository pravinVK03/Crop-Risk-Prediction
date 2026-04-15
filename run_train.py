from src.train import train_model


if __name__ == "__main__":
    output = train_model()
    metrics = output["metrics"]
    print("Training completed.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
