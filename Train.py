import os
import torch
import Config
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
import torch.optim as optim
from Generator import Generator
from Dataloader import MapDataset
from Discriminator import Discriminator
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from Utils import save_images_of_batch, save_checkpoint, load_checkpoint, save_history


def train_one_epoch(discriminator, generator, train_loader, discriminator_optimizer, generator_optimizer, bce_loss, l1_loss):
    loop = tqdm(train_loader, leave=True)
    disc_loss = list()
    gen_loss = list()
    discriminator.train()
    generator.train()
    for idx, (x, y) in enumerate(loop):
        x = x.to(Config.DEVICE).float()
        y = y.to(Config.DEVICE).float()

        y_fake = generator(x)
        D_real = discriminator(x, y)
        
        D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
        D_fake = discriminator(x, y_fake.detach())
        D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        disc_loss.append(D_loss.item())
        discriminator_optimizer.zero_grad()
        D_loss.backward()
        discriminator_optimizer.step()


        D_fake = discriminator(x, y_fake)
        G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(y_fake, y)
        G_loss = (G_fake_loss + L1 * Config.L1_LAMBDA) / 2
        gen_loss.append(G_loss.item())


        generator_optimizer.zero_grad()
        G_loss.backward()
        generator_optimizer.step()
    
    return np.mean(disc_loss), np.mean(gen_loss)


def validation_fn(generator, validation_loader, epoch):
    generator.eval()
    for x, y in validation_loader:
        fname = os.path.join(Config.SAVE_EXAMPLE_PATH, str(epoch) + '.png')
        with torch.no_grad():
            predictions = generator(x.to(Config.DEVICE).float())
        save_images_of_batch(x, y, predictions, fname)



def main():
    discriminator = Discriminator(in_channels=3, features=[64, 128, 256, 512])
    generator = Generator(in_channels=3, features=64)

    discriminator = discriminator.to(Config.DEVICE)
    generator = generator.to(Config.DEVICE)

    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=Config.LEARNING_RATE, betas=(0.5, 0.999))
    generator_optimizer = optim.Adam(generator.parameters(), lr=Config.LEARNING_RATE, betas=(0.5, 0.999))

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    

    if Config.LOAD_CHECKPOINT:
        load_checkpoint(generator, generator_optimizer, discriminator, discriminator_optimizer, learning_rate=Config.LEARNING_RATE, fname=Config.MODEL_CHECKPOINT_PATH)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

        
    train_dataset = MapDataset(Config.TRAIN_DATA_DIR, Config.IMG_SIZE, transform=transform)
    validation_dataset = MapDataset(Config.VAL_DATA_DIR, Config.IMG_SIZE, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    validation_loader = DataLoader(validation_dataset, batch_size=Config.VAL_BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

    
    disc_loss_min = float('inf')
    gen_loss_min = float('inf')

    disc_loss_list = list()
    gen_loss_list = list()

    for epoch in range(47, Config.NUM_EPOCHS):
        disc_loss, gen_loss = train_one_epoch(discriminator, generator, train_loader, discriminator_optimizer, generator_optimizer, bce_loss, l1_loss)
        disc_loss_list.append(disc_loss)
        gen_loss_list.append(gen_loss)
        print('Train epoch ' + str(epoch) + ' Disc Loss ', str(disc_loss), ' Generator Loss ', str(gen_loss))
        if disc_loss < disc_loss_min and gen_loss < gen_loss_min:
            print('Saving checkpoint disc_loss ' + str(disc_loss_min) + ' -> ' + str(disc_loss) + ' gen_loss ' + str(gen_loss_min) + ' -> ' + str(gen_loss))
            disc_loss_min = disc_loss
            gen_loss_min = gen_loss
            save_checkpoint(generator, generator_optimizer, discriminator, discriminator_optimizer, Config.MODEL_CHECKPOINT_PATH)
        validation_fn(generator, validation_loader, epoch)
    
    save_history(disc_loss_list, gen_loss_list, fname="historynn")

if __name__ == "__main__":
    main()

