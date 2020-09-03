import os

dataset_types = ['train', 'val', 'test']
miniImageNet_path = r'D:\Research\Data\miniImageNet'
generated_csv_path = './data/miniImageNet'


def generate_specified_csv(dataset_type):
    path = os.path.join(miniImageNet_path, dataset_type)
    with open(f'{generated_csv_path}/{dataset_type}.csv', 'w') as csv_file:
        csv_file.write("path, label\n")
        for class_name in os.listdir(path):
            class_path = os.path.join(path, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                csv_file.write(f"{image_path}, {class_name}\n")


def main():
    for dataset_type in dataset_types:
        generate_specified_csv(dataset_type)


if __name__ == '__main__':
    main()
