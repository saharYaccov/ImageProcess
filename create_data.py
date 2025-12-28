import os
from datasets import load_dataset

# 1. טעינת הדאטה וערבוב (בוחרים 1200 כדי לחלק בצורה נוחה)
dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train")
dataset = dataset.shuffle(seed=42)
subset_size = 3000
small_subset = dataset.select(range(subset_size))

train_float , val_float = 0.8 , 0.1
# 2. הגדרת יחסי חלוקה
train_end = int(subset_size * train_float)  # 840 תמונות
val_end = train_end + int(subset_size * val_float)  # 180 תמונות

# 3. יצירת מבנה התיקיות
base_dir = "data"
splits = ['train', 'val', 'test']
categories = ['ai', 'real']

for s in splits:
    for cat in categories:
        os.makedirs(f"{base_dir}/{s}/{cat}", exist_ok=True)

# 4. מעבר על התמונות ושמירתן במקום הנכון
for i, item in enumerate(small_subset):
    image = item['image']
    label = item['label']  # 0=Real, 1=Fake/AI

    # קביעת הפיצול (Split)
    if i < train_end:
        current_split = 'train'
    elif i < val_end:
        current_split = 'val'
    else:
        current_split = 'test'

    # קביעת הקטגוריה (Category)
    # שים לב: במערך הזה 1 הוא Fake (AI) ו-0 הוא Real
    category = "ai" if label == 1 else "real"

    # שמירת התמונה
    file_path = f"{base_dir}/{current_split}/{category}/img_{i}.png"
    image.save(file_path)

print(f"הסתיים! המבנה נוצר בתיקיית '{base_dir}':")
print("- Train: 840 images")
print("- Val: 180 images")
print("- Test: 180 images")
