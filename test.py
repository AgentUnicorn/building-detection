from create_data_loaders import create_data_loaders
from split_dataset import split_dataset

target_dir = "./datasets/Rio"
# split_dataset(
#     image_dir="./datasets/AOIs/AOI_1_Rio/3band",
#     geojson_dir="./datasets/AOIs/AOI_1_Rio/geojson",
#     output_root=target_dir,
# )

train_loader, val_loader, test_loader = create_data_loaders(
    target_dir, batch_size=4, num_workers=4
)
print(train_loader)
print(val_loader)
print(test_loader)
print(full_dataset)
