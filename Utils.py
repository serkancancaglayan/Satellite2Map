import torch 
import cv2 as cv
import Config
import numpy as np
from scipy.io import savemat
from matplotlib import pyplot as plt

def save_images_of_batch(inputs, targets, predictions, fname):
    rows = list()
    for input_, target, prediction in zip(inputs, targets, predictions):
        prediction = (prediction * 0.5 + 0.5) * 255
        target = (target * 0.5 + 0.5) * 255
        input_ = (input_ * 0.5 + 0.5) * 255
        prediction = prediction.permute(1, 2, 0)
        target = target.permute(1, 2, 0)
        input_ = input_.permute(1, 2, 0)

        row =  np.hstack([input_.cpu().numpy(), target.cpu().numpy(), prediction.cpu().numpy()])
        rows.append(row)
    final_grid = rows[0]
    for row in rows[1:]:
        final_grid = cv.vconcat([final_grid, row])
    cv.imwrite(fname, final_grid)


def save_checkpoint(generator_model, generator_optimizer, discriminator_model, discriminator_optimizer, fname):
    model_dict = {
        "generator_model": generator_model.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_model": discriminator_model.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict()
    }
    torch.save(model_dict, fname)


def load_checkpoint(generator_model, generator_optimizer, discriminator_model, discriminator_optimizer, learning_rate, fname):
    model_dict = torch.load(fname, map_location=Config.DEVICE)
    generator_model.load_state_dict(model_dict["generator_model"])
    generator_optimizer.load_state_dict(model_dict["generator_optimizer"])
    discriminator_model.load_state_dict(model_dict["discriminator_model"])
    discriminator_optimizer.load_state_dict(model_dict["discriminator_optimizer"])

    for param_group in generator_optimizer.param_groups:
        param_group["lr"] = learning_rate

    for param_group in discriminator_optimizer.param_groups: 
        param_group["lr"] = learning_rate


def save_history(discriminator_loss, generator_loss, fname="history "):
    history_dict = {
        "discriminator_loss": discriminator_loss,
        "generator_loss": generator_loss
    }
    savemat(fname + '.mat', history_dict)
    plt.figure(figsize=(19.2, 10.9))
    plt.plot(discriminator_loss, label='Discriminator Loss')
    plt.plot(generator_loss, label='Generator Loss')
    plt.legend()
    plt.savefig(fname + 'png')


         

    

    

