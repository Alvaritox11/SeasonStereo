import wandb 
from torchvision.transforms.functional import to_tensor

def get_tb_tracker(accelerator):
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            return tracker
    return None


def get_wandb_tracker(accelerator):
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            return tracker
    return None


def log_images(accelerator, tb_tracker, wandb_tracker, images_dict, step):
    if tb_tracker is not None:
        for name, img in images_dict.items():
            tb_tracker.writer.add_image(name, to_tensor(img), step)
        tb_tracker.writer.flush()

    if wandb_tracker is not None:
        import wandb
        wandb_tracker.log(
            {
                name: wandb.Image(img)
                for name, img in images_dict.items()
            },
            step=step,
        )