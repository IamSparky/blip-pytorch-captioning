from PIL import Image
from torch.utils.data import Dataset

# create dataframe
class BLIPCaptionDataset(Dataset):
    def __init__(self, dataframe, processor, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.prefix_text = "a picture of" 
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.loc[idx, "Image File"]
        caption = self.data.loc[idx, "Caption"]
        image = Image.open(image_path).convert("RGB")

        # 🔥 THIS is the correct usage for BLIP
        processed = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        # Labels are still the actual caption
        label_encoding = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )


        return {
            "pixel_values": processed["pixel_values"].squeeze(0),
            "input_ids": processed["input_ids"].squeeze(0),   # constant prompt
            "labels": label_encoding["input_ids"].squeeze(0)  # actual caption
        }