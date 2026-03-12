import os
import re
import glob
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image, TiffImagePlugin
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

_FRAME_REGEX = re.compile(r"^vid_(\d+)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
_MEL_EXTS = (".tiff", ".tif")


def _parse_frame_filename(path: str) -> Optional[Tuple[str, int, str]]:
    """
    Returns (video_id_str, frame_idx_int, stem_without_ext) or None if not matching.
    Example: '.../vid_004_093.png' -> ('004', 93, 'vid_004_093')
    """
    name = os.path.basename(path)
    m = _FRAME_REGEX.match(name)
    if not m:
        return None
    vid_str = m.group(1)     # keep zero-padded string, e.g., '004'
    frame_idx = int(m.group(2))
    stem = os.path.splitext(name)[0]
    return vid_str, frame_idx, stem


def _find_mel_for_stem(mels_dir: str, stem: str) -> Optional[str]:
    """Find a .tiff/.tif that matches the given stem (e.g., 'vid_004_098')."""
    for ext in _MEL_EXTS:
        cand = os.path.join(mels_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    return None


def _is_consecutive(ints: List[int]) -> bool:
    return all(b - a == 1 for a, b in zip(ints[:-1], ints[1:]))





# In data_lipsync.py (top-level, not inside another function)
class CollatePadMel:
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value
    def __call__(self, batch):
        return lipsync_collate_pad_mel(batch, pad_value=self.pad_value)





class LipSyncDataset(Dataset):
    """
    Yields:
        imgs: Tensor (L, C, H, W) of L consecutive frames (last aligns with mel)
        mel:  Tensor (1, H_mel, W_mel), dtype=float32
        meta: dict with paths/ids

    Assumptions:
      - Frames follow naming 'vid_{VID}_{FRAME}.png', e.g. vid_004_093.png
      - Mels are named identically but with .tiff or .tif
      - Sequences do not cross video boundaries.
    """

    def __init__(
        self,
        frames_dir: str,
        mels_dir: Optional[str] = None,
        sequence_length: int = 5,
        image_extensions: Tuple[str, ...] = (".png",),
        transform: Optional[Any] = None,
        mel_transform: Optional[Any] = None,
        enforce_consecutive: bool = True,
        mel_height_target: Optional[int] = None,
        mel_pad_value: float = 0.0,
        return_paths: bool = False,
        recursive: bool = True,
    ):
        """
        Args:
            frames_dir: Directory containing frame images.
            mels_dir:   Directory containing mel .tiff/.tif (defaults to frames_dir if None).
            sequence_length: Number of consecutive frames (last aligns with mel).
            image_extensions: Frame extensions to include.
            transform:   Transform for images (default = ToTensor()).
            mel_transform: Transform for mel tensor (applied after conversion to (1,H,W) float32).
            enforce_consecutive: Require frame indices to differ by exactly +1.
            mel_height_target: If set, crop/pad mel height to this value.
            mel_pad_value: Value used for mel padding.
            return_paths: Include full paths in meta.
            recursive:   Recurse into subdirectories.
        """
        super().__init__()
        self.frames_dir = frames_dir
        self.mels_dir = mels_dir or frames_dir
        self.sequence_length = int(sequence_length)
        assert self.sequence_length >= 1, "sequence_length must be >= 1"

        self.image_extensions = tuple(ext.lower() for ext in image_extensions)
        self.transform = transform or T.ToTensor()
        self.mel_transform = mel_transform
        self.enforce_consecutive = enforce_consecutive
        self.mel_height_target = mel_height_target
        self.mel_pad_value = float(mel_pad_value)
        self.return_paths = return_paths
        self.recursive = recursive

        self.samples = self._build_index()

    def _gather_frame_files(self) -> List[str]:
        if self.recursive:
            # recursively gather all matching files
            frame_files = []
            for ext in self.image_extensions:
                frame_files.extend(
                    glob.glob(os.path.join(self.frames_dir, "**", f"*{ext}"), recursive=True)
                )
        else:
            frame_files = []
            for ext in self.image_extensions:
                frame_files.extend(glob.glob(os.path.join(self.frames_dir, f"*{ext}")))

        # filter to only files matching 'vid_..._....EXT' pattern
        filtered = []
        for p in frame_files:
            parsed = _parse_frame_filename(p)
            if parsed is not None:
                filtered.append(p)
        return sorted(filtered)

    def _build_index(self) -> List[Dict[str, Any]]:
        """
        Build a list of valid samples:
          Each sample contains keys:
            - 'video_id' (str), e.g. '004'
            - 'frame_indices' (List[int])
            - 'frame_paths' (List[str])
            - 'mel_path' (str)
            - 'last_frame_idx' (int)
        """
        files = self._gather_frame_files()

        # Group frames by video id
        by_video: Dict[str, Dict[int, str]] = {}
        for path in files:
            parsed = _parse_frame_filename(path)
            if parsed is None:
                continue
            vid_str, frame_idx, stem = parsed
            by_video.setdefault(vid_str, {})[frame_idx] = path

        samples: List[Dict[str, Any]] = []
        L = self.sequence_length

        for vid_str, idx2path in by_video.items():
            sorted_indices = sorted(idx2path.keys())

            # Slide over possible end indices
            for end_idx in sorted_indices:
                start_idx = end_idx - (L - 1)
                # Collect the candidate sequence indices:
                seq_idxs = list(range(start_idx, end_idx + 1))
                if seq_idxs[0] not in idx2path:
                    # skip if start doesn't exist
                    continue
                # Ensure all indices exist as frames
                if not all((i in idx2path) for i in seq_idxs):
                    continue
                # Optionally enforce strict consecutiveness
                if self.enforce_consecutive and not _is_consecutive(seq_idxs):
                    continue

                # Find mel for last frame (end_idx)
                stem = f"vid_{vid_str}_{end_idx:03d}"
                mel_path = _find_mel_for_stem(self.mels_dir, stem)
                if mel_path is None:
                    # no mel for this end frame -> skip
                    continue

                frame_paths = [idx2path[i] for i in seq_idxs]
                samples.append({
                    "video_id": vid_str,
                    "frame_indices": seq_idxs,
                    "frame_paths": frame_paths,
                    "mel_path": mel_path,
                    "last_frame_idx": end_idx,
                })

        # You may want to sort samples by (video_id, last_frame_idx)
        samples.sort(key=lambda s: (s["video_id"], s["last_frame_idx"]))
        if len(samples) == 0:
            raise RuntimeError(
                "No valid sequences found. "
                "Check file naming, directories, and sequence_length."
            )
        return samples

    @staticmethod
    def _load_image_as_tensor(path: str, transform: Any) -> torch.Tensor:
        img = Image.open(path).convert("RGB")  # 512x512 per your setup
        tensor = transform(img) if transform else T.ToTensor()(img)
        # Ensure float32 and shape (C,H,W)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        return tensor

    def _load_mel_tiff(self, path: str) -> np.ndarray:
        # Handle multi-page TIFF by taking the first frame
        with Image.open(path) as im:
            # If it’s a multi-page tiff, load the first page
            try:
                # Some TIFFs can have multiple frames/pages
                if hasattr(im, "n_frames") and im.n_frames > 1:
                    im.seek(0)
            except Exception:
                pass
            # Convert to numpy. Mode can be 'F', 'I;16', 'L', etc.
            arr = np.array(im, dtype=np.float32)
        # Expect (H, W). If there's a trailing channel by accident, handle it
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"Unexpected mel array shape for {path}: {arr.shape}")
        return arr  # float32, (H, W)

    def _maybe_resize_mel_height(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (1, H, W) float32. If mel_height_target is set:
           - if H < target: pad at bottom with mel_pad_value
           - if H > target: center-crop to target
        """
        if self.mel_height_target is None:
            return mel
        _, H, W = mel.shape
        Tgt = self.mel_height_target
        if H == Tgt:
            return mel
        if H < Tgt:
            pad_amt = Tgt - H
            pad = torch.full((1, pad_amt, W), self.mel_pad_value, dtype=mel.dtype)
            return torch.cat([mel, pad], dim=1)  # pad at bottom
        else:
            # center-crop along height
            start = (H - Tgt) // 2
            end = start + Tgt
            return mel[:, start:end, :]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        frame_paths = rec["frame_paths"]
        mel_path = rec["mel_path"]

        # Load images
        imgs = [self._load_image_as_tensor(p, self.transform) for p in frame_paths]
        # (L, C, H, W)
        imgs = torch.stack(imgs, dim=0)

        # Load mel
        mel_np = self._load_mel_tiff(mel_path)  # (H, W) float32
        mel = torch.from_numpy(mel_np).unsqueeze(0)  # (1, H, W)
        mel = mel[..., :86] if mel.shape[-1] > 86 else mel
        if self.mel_transform:
            mel = self.mel_transform(mel)
        mel = mel.to(torch.float32)
        mel = self._maybe_resize_mel_height(mel)

        meta = {
            "video_id": rec["video_id"],
            "frame_indices": rec["frame_indices"],
            "last_frame_idx": rec["last_frame_idx"],
        }
        if self.return_paths:
            meta["frame_paths"] = frame_paths
            meta["mel_path"] = mel_path

        return imgs, mel, meta


def lipsync_collate_pad_mel(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]], pad_value: float = 0.0):
    """
    Pads mel height to the maximum in the batch, stacks everything.
    Args:
        batch: list of (imgs[L,C,H,W], mel[1,Hm,Wm], meta)
    Returns:
        imgs: (B, L, C, H, W)
        mel:  (B, 1, Hm_max, Wm)  # assumes all mel widths are equal (e.g., 256)
        meta: list of dicts
    """
    imgs_list, mel_list, meta_list = zip(*batch)

    # Stack imgs (all same size)
    imgs = torch.stack(imgs_list, dim=0)  # (B, L, C, H, W)

    # Determine max mel height; assume equal width
    heights = [m.shape[-2] for m in mel_list]
    widths = [m.shape[-1] for m in mel_list]
    if len(set(widths)) != 1:
        raise ValueError(f"Mel widths vary within batch: {set(widths)}. "
                         "This implementation assumes constant width.")
    Hmax = max(heights)
    W = widths[0]

    padded = []
    for m in mel_list:
        _, H, Wm = m.shape
        if H == Hmax:
            padded.append(m)
        else:
            pad_amt = Hmax - H
            pad = torch.full((1, pad_amt, Wm), float(pad_value), dtype=m.dtype)
            padded.append(torch.cat([m, pad], dim=1))
    mel = torch.stack(padded, dim=0)  # (B, 1, Hmax, W)

    return imgs, mel, list(meta_list)
