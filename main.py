#!/usr/bin/env python3

from parsecli import *

from PicoscenesToolbox.picoscenes import Picoscenes
from tqdm import tqdm
import numpy as np

from pathlib import Path


def picoFrame2numpy(
  frameRaw: dict, dataType: str, interpolate: bool
) -> np.datetime64 | np.ndarray:
  assert dataType in ("csi", "amp", "phase", "timestamp")

  # Parse timestamp
  if dataType == "timestamp":
    rawTimesteampNs: int = frameRaw["RxSBasic"]["systemns"]
    timestamp = np.datetime64(rawTimesteampNs, "ns")
    return timestamp

  # Parse CSI
  else:
    typeMap = {"csi": "CSI", "amp": "Mag", "phase": "Phase"}
    cbw: int = frameRaw["CSI"]["CBW"]

    nS: int = frameRaw["CSI"]["numTones"]
    nTx: int = frameRaw["CSI"]["numTx"]
    nRx: int = frameRaw["CSI"]["numRx"]
    shape: tuple = (nS, nTx, nRx)

    ## PicoscenesToolbox always interpolate the missing subcarrier
    picoCsi = np.array(frameRaw["CSI"][typeMap[dataType]]).reshape(shape)

    if interpolate:
      return picoCsi
    else:
      picoSubcarrierIdx: list[int] = frameRaw["CSI"]["SubcarrierIndex"]
      interpolatedIdx: frozenset = frozenset((-1, 0, 1) if cbw == 40 else (0,))

      realCsi: list[np.ndarray] = []
      for idx in picoSubcarrierIdx:
        if idx not in interpolatedIdx:
          realCsi.append(picoCsi[picoSubcarrierIdx.index(idx)])
      return np.array(realCsi)


def pico2Numpy(
  picoRaw: list[dict], types: frozenset, interpolate: bool = False
) -> dict[str, np.ndarray]:
  outputByType: dict[str, list[np.datetime64 | np.ndarray]] = {}
  outputByTypeNp: dict[str, np.ndarray] = dict.fromkeys(types)

  for dataType in types:
    outputByType[dataType] = []

  # Parse each frame along reversed time axis
  for raw in tqdm(picoRaw, leave=False, desc="Frames"):
    for dataType in types:
      frame = picoFrame2numpy(raw, dataType, interpolate)
      outputByType[dataType].append(frame)

  for dataType in tuple(outputByType.keys()):
    data = outputByType.pop(dataType)
    outputByTypeNp[dataType] = np.array(data)
  return outputByTypeNp


def saveNumpy(inPath: Path, outDir: Path, types: frozenset):
  outDir.mkdir(parents=True, exist_ok=True)

  outputByType = pico2Numpy(Picoscenes(str(inPath)).raw, types)
  for dataType in tuple(outputByType.keys()):
    data = outputByType.pop(dataType)
    filename = inPath.with_suffix(f".{dataType}.npy").name
    np.save(outDir / filename, data)


if __name__ == "__main__":
  scriptPath = Path(__file__).parent

  args = parseCli()
  inDir = Path(args.inDir)
  outDir = Path(args.outDir)

  for rawPath in tqdm(tuple(inDir.glob("*.csi")), desc="Files"):
    saveNumpy(rawPath, outDir, args.types)
