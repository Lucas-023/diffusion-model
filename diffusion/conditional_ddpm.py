import torch
from tqdm import tqdm
import logging

class Diffusion_conditional:

    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=32,
        device="cuda"
    ):

        self.noise_steps = noise_steps

        self.beta_start = beta_start
        self.beta_end = beta_end

        self.img_size = img_size

        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)

        self.alpha = 1. - self.beta

        self.alpha_hat = torch.cumprod(
            self.alpha,
            dim=0
        )

    def prepare_noise_schedule(self):

        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.noise_steps
        )

    def noise_images(self, x, t):

        sqrt_alpha_hat = torch.sqrt(
            self.alpha_hat[t]
        )[:, None, None, None]

        sqrt_one_minus_alpha_hat = torch.sqrt(
            1 - self.alpha_hat[t]
        )[:, None, None, None]

        epsilon = torch.randn_like(x)

        return (
            sqrt_alpha_hat * x +
            sqrt_one_minus_alpha_hat * epsilon,
            epsilon
        )

    def sample_timesteps(self, n):

        return torch.randint(
            1,
            self.noise_steps,
            size=(n,)
        )

    def sample(
        self,
        model,
        n,
        context=None,
        channels=4,
        cfg_scale=3.0
    ):

        logging.info(f"Sampling {n} images...")

        model.eval()

        with torch.no_grad():

            x = torch.randn(
                (
                    n,
                    channels,
                    self.img_size,
                    self.img_size
                )
            ).to(self.device)

            for i in tqdm(
                reversed(range(1, self.noise_steps)),
                position=0
            ):

                t = (
                    torch.ones(n) * i
                ).long().to(self.device)

                if context is not None:

                    noise_pred_cond = model(
                        x,
                        t,
                        context=context
                    )

                    noise_pred_uncond = model(
                        x,
                        t,
                        context=torch.zeros_like(context)
                    )

                    predicted_noise = (
                        noise_pred_uncond +
                        cfg_scale * (
                            noise_pred_cond -
                            noise_pred_uncond
                        )
                    )

                else:

                    predicted_noise = model(
                        x,
                        t,
                        context=torch.zeros(
                            n,
                            512,
                            40
                        ).to(self.device)
                    )

                alpha = self.alpha[t][:, None, None, None]

                alpha_hat = self.alpha_hat[t][:, None, None, None]

                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (
                    1 / torch.sqrt(alpha)
                ) * (
                    x - (
                        (
                            1 - alpha
                        ) / torch.sqrt(
                            1 - alpha_hat
                        )
                    ) * predicted_noise
                ) + torch.sqrt(beta) * noise

        model.train()

        return x