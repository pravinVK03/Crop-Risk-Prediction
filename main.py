from src.data_loader import load_data
from src.predict import load_model, predict_from_api_payload
from src.train import train_model


def main():
    df = load_data()
    train_output = train_model(df)

    bundle = load_model()
    api_payload = {
        "location": {"state": "karnataka", "district": "mysore"},
        "crop_type": "rice",
    }

    result = predict_from_api_payload(api_payload, bundle=bundle)
    prediction = result["prediction"]

    print("TabNet tabular training finished.")
    print(f"Test Accuracy: {train_output['metrics']['accuracy']:.4f}")
    print(f"Predicted Risk: {prediction['risk_label']} (class {prediction['risk_class']})")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print(f"Inferred Fields: {', '.join(result['inferred_fields'])}")

    print("\nTop Reasons:")
    for item in prediction["reasons"]:
        print(f"- {item['reason']}")

    print("\nPrecautions:")
    for item in prediction["precautions"]:
        print(f"- {item}")


if __name__ == "__main__":
    main()
