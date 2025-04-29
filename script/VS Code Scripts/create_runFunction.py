import torch
import create_dataframe
import create_model
import warnings

from create_datasetClass import BLIPCaptionDataset
from torch.utils.data import DataLoader
from create_trainingFunction import train_loop_fn
from create_evaluationFunction import eval_loop_fn

# Suppress all warnings
warnings.filterwarnings('ignore')

def run():
    # Set constants
    EPOCHS = 2
    TRAIN_BATCH_SIZE = 2
    VALID_BATCH_SIZE = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(create_model.model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2, T_mult=1, eta_min=1e-6, last_epoch=-1
    )

    for fold in range(1, 6):
        print("*" * 20, f"FOLD NUMBER {fold}", "*" * 20)
        df_train = create_dataframe.df[create_dataframe.df['Fold'] != fold].reset_index(drop=True)
        df_valid = create_dataframe.df[create_dataframe.df['Fold'] == fold].reset_index(drop=True)

        train_dataset = BLIPCaptionDataset(df_train, create_model.processor)
        valid_dataset = BLIPCaptionDataset(df_valid, create_model.processor)

        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=4)

        for epoch in range(EPOCHS):
            print(f"Epoch --> {epoch + 1} / {EPOCHS}")
            print("-------------------------------")
            train_metrics = train_loop_fn(train_loader, create_model.model, create_model.processor, optimizer, DEVICE, scheduler)
            print("Training Loss & Metrics:")
            print(train_metrics)

            val_metrics = eval_loop_fn(val_loader, create_model.model, create_model.processor, DEVICE)
            print("Validation Loss & Metrics:")
            print(val_metrics)

        print("\n")

    torch.save(create_model.model.state_dict(), './best_blip_large_captioning_model.pt')


if __name__ == "__main__":
    run()