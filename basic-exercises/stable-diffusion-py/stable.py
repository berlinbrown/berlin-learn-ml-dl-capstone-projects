import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------------------------------
# Create a simple image
# ---------------------------------------

SIZE = 32

image = np.zeros((SIZE, SIZE), dtype=np.float32)
image[10:22, 10:22] = 1.0

image_tensor = torch.tensor(image).flatten()

# ---------------------------------------
# Tiny Neural Network
# ---------------------------------------

model = nn.Sequential(
    nn.Linear(SIZE * SIZE, 512),
    nn.ReLU(),
    nn.Linear(512, SIZE * SIZE)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------
# Train
# ---------------------------------------

epochs = 2000

for epoch in range(epochs):

    # Generate new random noise every iteration
    noise = torch.randn(SIZE * SIZE)

    # Add noise to image
    noisy_image = image_tensor + noise * 0.5

    # Predict the noise
    predicted_noise = model(noisy_image)

    # Compare prediction against actual noise
    loss = ((predicted_noise - noise) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} Loss {loss.item():.4f}")

# ---------------------------------------
# Test
# ---------------------------------------

test_noise = torch.randn(SIZE * SIZE)
test_noisy = image_tensor + test_noise * 0.5

with torch.no_grad():
    predicted = model(test_noisy)
    reconstructed = test_noisy - predicted * 0.5

# ---------------------------------------
# Display
# ---------------------------------------

fig, ax = plt.subplots(1,3, figsize=(10,4))

ax[0].imshow(image.reshape(SIZE,SIZE), cmap="gray")
ax[0].set_title("Original")

ax[1].imshow(test_noisy.reshape(SIZE,SIZE), cmap="gray")
ax[1].set_title("Noisy")

# After millions of examples, the model develops an internal understanding of ideas like:
# What makes something look like a bunny (ears, fur, body proportions).
# What makes something look like a guitar (neck, strings, body).
# What "playing" usually looks like (holding the guitar, hand positions, posture).
# Concept: Bunny
# Concept: Guitar
#         +
# Concept: Playing
#         ↓
# Generate an image satisfying all three.


ax[2].imshow(reconstructed.reshape(SIZE,SIZE), cmap="gray")
ax[2].set_title("Reconstructed")

for a in ax:
    a.axis("off")

plt.show()