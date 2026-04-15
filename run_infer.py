import argparse
import json

from src.predict import load_model, predict_from_api_payload


def parse_args():
    parser = argparse.ArgumentParser(description="Run crop risk inference from crop + location.")
    parser.add_argument("--state", required=False, help="State name")
    parser.add_argument("--district", required=False, help="District name")
    parser.add_argument("--crop", required=False, help="Crop name")
    return parser.parse_args()


def main():
    args = parse_args()
    state = args.state or input("Enter State: ").strip()
    district = args.district or input("Enter District: ").strip()
    crop = args.crop or input("Enter Crop: ").strip()

    payload = {
        "location": {"state": state, "district": district},
        "crop_type": crop,
    }

    bundle = load_model()
    result = predict_from_api_payload(payload, bundle=bundle)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
