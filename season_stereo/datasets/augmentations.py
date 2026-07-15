import random
import torch
import torch.nn.functional as F
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF


class StereoAugmentor:
    def __init__(
        self,
        crop_size,
        min_scale=-0.2,
        max_scale=0.5,
        stretch_prob=0.8,
        max_stretch=0.2,
        do_flip=True,
        h_flip_prob=0.5,
        v_flip_prob=0.1,
        yjitter=False,
        saturation_range=(0.6, 1.4),
        brightness=0.4,
        contrast=0.4,
        hue=0.159,
        gamma_range=(1.0, 1.0, 1.0, 1.0),
        asymmetric_prob=0.2,
        eraser_prob=0.5,
        # ── Overlap-shift augmentation ────────────────────────────────────
        # Simulates reduced stereo overlap by shifting the right image
        # leftward, zeroing its right columns, and correcting GT disparity.
        #
        # shift_p_apply : probability that *any* shift is applied (rest → 0).
        # shift_scale   : mean of the exponential draw (pixels).
        # shift_max     : hard cap.  None → W // 2 at runtime.
        shift_p_apply=0.5,
        shift_scale=100.0,
        shift_max=None,
    ):
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.stretch_prob = stretch_prob
        self.max_stretch = max_stretch
        self.do_flip = do_flip
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.yjitter = yjitter
        self.asymmetric_prob = asymmetric_prob
        self.gamma_range = gamma_range
        self.eraser_prob = eraser_prob
        self.shift_p_apply = shift_p_apply
        self.shift_scale = shift_scale
        self.shift_max = shift_max

        self.color_jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=tuple(saturation_range),
            hue=hue,
        )

    # ── Overlap-shift helpers ─────────────────────────────────────────────────

    def _sample_overlap_shift(self, W: int) -> int:
        """
        Sample a horizontal shift for the right image.

        Returns 0 with probability (1 - shift_p_apply).
        Otherwise draws from Exponential(shift_scale), capped at
        shift_max (defaults to W // 2).
        """
        if random.random() >= self.shift_p_apply:
            return 0
        cap = self.shift_max if self.shift_max is not None else W // 2
        shift = random.expovariate(1.0 / self.shift_scale)
        return min(int(shift), cap)

    def overlap_shift_aug(self, right_img, disp_items, valid_items):
        """
        Shift the right image leftward by ``shift`` pixels.

        Right image
            Columns [shift, W) move to [0, W-shift); the last ``shift``
            columns are zeroed (no content).

        GT disparity
            ``d_new = d + shift`` — same 3-D point now appears ``shift``
            pixels further left in the right frame.

        Valid mask
            The match for left pixel at column *x* lands at column
            ``x - d_new`` in the shifted right image.  That position must
            lie inside the content region [0, W-shift):

                0  ≤  x - d_new  <  W - shift

            The left bound (≥ 0) catches matches that fall off the left
            edge.  The right bound (< W-shift) catches matches that land
            in the black / zero-filled zone — which matters when
            disparities are negative (e.g. after horizontal flip).
        """
        _, H, W = right_img.shape
        shift = self._sample_overlap_shift(W)
        if shift == 0:
            return right_img, disp_items, valid_items

        # ── Shift right image ─────────────────────────────────────────────
        right_shifted = torch.zeros_like(right_img)
        if shift < W:
            right_shifted[:, :, : W - shift] = right_img[:, :, shift:]

        # ── Update disparities ────────────────────────────────────────────
        new_disp_items = None
        if disp_items is not None:
            new_disp_items = [d + shift for d in disp_items]

        # ── Update valid masks ────────────────────────────────────────────
        new_valid_items = None
        if valid_items is not None:
            # x-coordinate of each pixel: (1, 1, W), broadcasts to (1, H, W)
            x_coords = torch.arange(W, device=valid_items[0].device).float().view(1, 1, W)

            new_valid_items = []
            for idx, v in enumerate(valid_items):
                if new_disp_items is not None and idx < len(new_disp_items):
                    match_pos = x_coords - new_disp_items[idx]   # (1, H, W)
                    in_content = ((match_pos >= 0) & (match_pos < W - shift)).float()
                    new_valid_items.append(v * in_content)
                else:
                    new_valid_items.append(v)

        return right_shifted, new_disp_items, new_valid_items

    # ── Existing augmentations (unchanged) ───────────────────────────────────

    def random_scaling_and_stretch(
        self,
        left_items,
        right_items,
        disp_items=None,
        valid_items=None,
    ):
        scale = 2 ** random.uniform(self.min_scale, self.max_scale)
        scale_x, scale_y = scale, scale

        if random.random() < self.stretch_prob:
            scale_x *= 2 ** random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** random.uniform(-self.max_stretch, self.max_stretch)

        h, w = left_items[0].shape[-2:]
        nh = int(h * scale_y)
        nw = int(w * scale_x)

        left_items = [
            TF.resize(x, (nh, nw), interpolation=TF.InterpolationMode.BILINEAR)
            for x in left_items
        ]
        right_items = [
            TF.resize(x, (nh, nw), interpolation=TF.InterpolationMode.BILINEAR)
            for x in right_items
        ]

        if disp_items is not None:
            disp_items = [
                F.interpolate(
                    x.unsqueeze(0),
                    size=(nh, nw),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0) * scale_x
                for x in disp_items
            ]

        if valid_items is not None:
            valid_items = [
                F.interpolate(
                    x.unsqueeze(0),
                    size=(nh, nw),
                    mode="nearest",
                ).squeeze(0)
                for x in valid_items
            ]

        return left_items, right_items, disp_items, valid_items

    def random_flip(
        self,
        left_items,
        right_items,
        disp_items=None,
        valid_items=None,
    ):
        if not self.do_flip:
            return left_items, right_items, disp_items, valid_items

        if random.random() < self.h_flip_prob:
            left_items = [TF.hflip(x) for x in left_items]
            right_items = [TF.hflip(x) for x in right_items]

            if disp_items is not None:
                disp_items = [TF.hflip(x) * -1 for x in disp_items]

            if valid_items is not None:
                valid_items = [TF.hflip(x) for x in valid_items]

        if random.random() < self.v_flip_prob:
            left_items = [TF.vflip(x) for x in left_items]
            right_items = [TF.vflip(x) for x in right_items]

            if disp_items is not None:
                disp_items = [TF.vflip(x) for x in disp_items]

            if valid_items is not None:
                valid_items = [TF.vflip(x) for x in valid_items]

        return left_items, right_items, disp_items, valid_items

    def random_crop(
        self,
        left_items,
        right_items,
        disp_items=None,
        valid_items=None,
    ):
        h, w = left_items[0].shape[-2:]
        th, tw = self.crop_size, self.crop_size

        if h < th or w < tw:
            pad_h = max(th - h, 0)
            pad_w = max(tw - w, 0)

            left_items = [
                F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") for x in left_items
            ]
            right_items = [
                F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") for x in right_items
            ]

            if disp_items is not None:
                disp_items = [
                    F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
                    for x in disp_items
                ]

            if valid_items is not None:
                valid_items = [
                    F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
                    for x in valid_items
                ]

            h, w = left_items[0].shape[-2:]

        i_left = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        if self.yjitter and random.random() < 0.5:
            y_shift = random.randint(-2, 2)
            i_right = max(0, min(h - th, i_left + y_shift))
        else:
            i_right = i_left

        left_items = [x[:, i_left:i_left + th, j:j + tw] for x in left_items]
        right_items = [x[:, i_right:i_right + th, j:j + tw] for x in right_items]

        if disp_items is not None:
            disp_items = [x[:, i_left:i_left + th, j:j + tw] for x in disp_items]

        if valid_items is not None:
            valid_items = [x[:, i_left:i_left + th, j:j + tw] for x in valid_items]

        return left_items, right_items, disp_items, valid_items

    def photometric_aug(self, img1, img2):
        if random.random() < self.asymmetric_prob:
            img1 = self.color_jitter(img1)
            img2 = self.color_jitter(img2)
        else:
            stack = torch.stack([img1, img2], dim=0)
            stack = self.color_jitter(stack)
            img1, img2 = stack[0], stack[1]

        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        gain = random.uniform(self.gamma_range[2], self.gamma_range[3])

        img1 = TF.adjust_gamma(img1, gamma, gain).clamp(0, 1)
        img2 = TF.adjust_gamma(img2, gamma, gain).clamp(0, 1)
        return img1, img2

    def eraser_aug(self, img):
        if random.random() < self.eraser_prob:
            _, h, w = img.shape
            mean_color = img.mean(dim=[1, 2], keepdim=True)

            for _ in range(random.randint(1, 3)):
                x0 = random.randint(0, w)
                y0 = random.randint(0, h)
                dx = random.randint(50, 100)
                dy = random.randint(50, 100)
                x1 = min(w, x0 + dx)
                y1 = min(h, y0 + dy)
                img[:, y0:y1, x0:x1] = mean_color

        return img

    def __call__(self, sample):
        left_items = [sample["left"]]
        right_items = [sample["right"]]

        left_mask_idx = None
        right_mask_idx = None
        left_building_idx = None
        right_building_idx = None

        if "left_mask" in sample:
            left_mask_idx = len(left_items)
            left_items.append(sample["left_mask"])
        if "right_mask" in sample:
            right_mask_idx = len(right_items)
            right_items.append(sample["right_mask"])

        if "left_building_mask" in sample:
            left_building_idx = len(left_items)
            left_items.append(sample["left_building_mask"])
        if "right_building_mask" in sample:
            right_building_idx = len(right_items)
            right_items.append(sample["right_building_mask"])

        has_real = "left_real" in sample and "right_real" in sample
        if has_real:
            left_real_idx  = len(left_items)
            right_real_idx = len(right_items)
            left_items.append(sample["left_real"])
            right_items.append(sample["right_real"])

        disp_keys, disp_items = [], []
        for k in ("disparity", "disparity_no_trees"):
            if k in sample:
                disp_keys.append(k)
                disp_items.append(sample[k])
        disp_items = disp_items if disp_items else None

        valid_keys, valid_items = [], []
        for k in ("valid", "valid_no_trees"):
            if k in sample:
                valid_keys.append(k)
                valid_items.append(sample[k])
        valid_items = valid_items if valid_items else None

        left_items, right_items, disp_items, valid_items = self.random_scaling_and_stretch(
            left_items, right_items, disp_items, valid_items
        )
        left_items, right_items, disp_items, valid_items = self.random_flip(
            left_items, right_items, disp_items, valid_items
        )
        left_items, right_items, disp_items, valid_items = self.random_crop(
            left_items, right_items, disp_items, valid_items
        )

        # ── Overlap-shift: applied after crop so shift is in final-res pixels.
        # Only the right image, GT disparity, and valid mask are modified.
        # No right_mask update needed — losses penalise on the left-aligned
        # disparity, not on the right image directly.
        right_items[0], disp_items, valid_items = self.overlap_shift_aug(
            right_items[0], disp_items, valid_items
        )

        left_img  = left_items[0]
        right_img = right_items[0]

        left_img, right_img = self.photometric_aug(left_img, right_img)
        right_img = self.eraser_aug(right_img)

        sample["left"]  = left_img
        sample["right"] = right_img

        if left_mask_idx is not None:
            sample["left_mask"] = (left_items[left_mask_idx] > 0.5).float()
        if right_mask_idx is not None:
            sample["right_mask"] = (right_items[right_mask_idx] > 0.5).float()

        if left_building_idx is not None:
            sample["left_building_mask"] = (left_items[left_building_idx] > 0.5).float()
        if right_building_idx is not None:
            sample["right_building_mask"] = (right_items[right_building_idx] > 0.5).float()

        if has_real:
            sample["left_real"]  = left_items[left_real_idx]
            sample["right_real"] = right_items[right_real_idx]

        if disp_items is not None:
            for k, x in zip(disp_keys, disp_items):
                sample[k] = x

        if valid_items is not None:
            for k, x in zip(valid_keys, valid_items):
                sample[k] = (x > 0.5).float()

        return sample
