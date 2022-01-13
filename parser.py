from argparse import ArgumentParser


def create_parser():
    parser = ArgumentParser(description="Training object detection models")

    parser.add_argument(
        "--model",
        type=str,
        default="fasterrcnn",
        choices=["fasterrcnn"],
        help="Choose the model to be used",
    )
    parser.add_argument(
        "--root", type=str, help="Path to your dataset", required=True
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch Size")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning Rate")
    parser.add_argument("--optim", type=str,
                        choices=["SGD", "Adam"], default="SGD", help="Optimizer")
    parser.add_argument("--gpu", action="store_true",
                        dest="gpu", help="Use GPU")
    parser.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="Do not use a pretrained model",
    )
    parser.add_argument(
        "--num-classes", type=int, help="number of classes", required=True
    )
    parser.add_argument(
        "--save-model", action="store_true", help="Save model?")
    parser.set_defaults(gpu=False, pretrained=True, save_model=False)
    return parser.parse_args()
