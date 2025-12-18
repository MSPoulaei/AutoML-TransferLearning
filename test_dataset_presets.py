"""
Test script to demonstrate the new dataset preset functionality.
This shows the available presets without requiring full imports.
"""


def show_dataset_presets():
    """Display available dataset presets."""
    presets = {
        "cifar10": {
            "num_classes": 10,
            "num_samples": 50000,
            "image_size": "32x32",
            "domain": "natural",
            "description": "CIFAR-10: 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)",
        },
        "cifar100": {
            "num_classes": 100,
            "num_samples": 50000,
            "image_size": "32x32",
            "domain": "fine_grained",
            "description": "CIFAR-100: 60,000 32x32 color images in 100 fine-grained classes",
        },
        "mnist": {
            "num_classes": 10,
            "num_samples": 60000,
            "image_size": "28x28",
            "domain": "document",
            "description": "MNIST: 70,000 28x28 grayscale images of handwritten digits",
        },
        "fashion_mnist": {
            "num_classes": 10,
            "num_samples": 60000,
            "image_size": "28x28",
            "domain": "natural",
            "description": "Fashion-MNIST: 70,000 28x28 grayscale images of fashion items",
        },
        "svhn": {
            "num_classes": 10,
            "num_samples": 73257,
            "image_size": "32x32",
            "domain": "document",
            "description": "SVHN: Street View House Numbers - real-world images of digits",
        },
        "imagenet": {
            "num_classes": 1000,
            "num_samples": 1281167,
            "image_size": "224x224",
            "domain": "natural",
            "description": "ImageNet: Large-scale dataset with 1000 object categories",
        },
    }

    print("\n" + "=" * 80)
    print("AVAILABLE DATASET PRESETS")
    print("=" * 80)

    for name, info in presets.items():
        print(f"\n{name.upper()}")
        print("-" * 40)
        print(f"  Classes:     {info['num_classes']}")
        print(f"  Samples:     {info['num_samples']:,}")
        print(f"  Image Size:  {info['image_size']}")
        print(f"  Domain:      {info['domain']}")
        print(f"  Description: {info['description']}")

    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    print("\n1. Run with CIFAR-10 preset (simulation mode):")
    print("   python main.py run --dataset cifar10 --budget 5 --simulation")

    print("\n2. Run with Fashion-MNIST preset:")
    print("   python main.py run --dataset fashion_mnist --budget 10 --simulation")

    print("\n3. Run with custom dataset:")
    print("   python main.py run --num-classes 10 --num-samples 5000 \\")
    print("       --domain natural --domain-desc 'My custom dataset' \\")
    print("       --budget 5 --simulation")

    print("\n4. List all available presets:")
    print("   python main.py list-datasets")

    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    show_dataset_presets()
