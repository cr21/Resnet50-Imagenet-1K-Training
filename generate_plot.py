import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV files
train_df = pd.read_csv('logs/resnet50_imagenet_1k_onecycleLr/csv_logger/training_log.csv')
test_df = pd.read_csv('logs/resnet50_imagenet_1k_onecycleLr/csv_logger/test_log.csv')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create figure with subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
fig.suptitle('Training and Validation Metrics', fontsize=16)

# Plot Loss
ax1.plot(train_df['epoch'], train_df['loss'], label='Train Loss')
ax1.plot(test_df['epoch'], test_df['loss'], label='Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Plot Top-1 Accuracy
ax2.plot(train_df['epoch'], train_df['accuracy'], label='Train Accuracy')
ax2.plot(test_df['epoch'], test_df['accuracy'], label='Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Top-1 Accuracy (%)')
ax2.legend()
ax2.grid(True)

# Plot Top-5 Accuracy
ax3.plot(train_df['epoch'], train_df['accuracy_top5'], label='Train Top-5 Accuracy')
ax3.plot(test_df['epoch'], test_df['accuracy_top5'], label='Test Top-5 Accuracy')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Top-5 Accuracy (%)')
ax3.legend()
ax3.grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig('static/training_metrics.png')
plt.close()