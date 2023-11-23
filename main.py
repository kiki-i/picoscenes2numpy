#!/usr/bin/env python3

from parsecli import parseCli

from PicoscenesToolbox.picoscenes import Picoscenes
from tqdm import tqdm
import numpy as np

from pathlib import Path


def picoFrame2numpy(frameRaw: dict, dataType: str, interpolate: bool) -> np.ndarray:
  assert dataType in ("csi", "amp", "phase", "timestamp")

  # Parse timestamp
  if dataType == "timestamp":
    rawTimesteampNs: int = frameRaw["RxSBasic"]["systemns"]
    timestamp = np.datetime64(rawTimesteampNs, "ns")
    return np.array(timestamp)

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
      picoSubcarrierList = np.array(frameRaw["CSI"]["SubcarrierIndex"])
      interpolatedSubcarrierList = (-1, 0, 1) if cbw == 40 else (0,)
      realSubcarrierIdxList = np.nonzero(
        ~np.in1d(picoSubcarrierList, interpolatedSubcarrierList)
      )

      return picoCsi[realSubcarrierIdxList]


def pico2Numpy(
  picoRaw: list[dict], types: frozenset, interpolate: bool = False
) -> dict[str, np.ndarray]:
  output: dict[str, list[np.ndarray]] = {}

  for dataType in types:
    output[dataType] = []

  # Parse each frame along reversed time axis
  for raw in tqdm(picoRaw, leave=False, desc="Frames"):
    for dataType in types:
      frame = picoFrame2numpy(raw, dataType, interpolate)
      output[dataType].append(frame)

  return {k: np.array(v) for k, v in output.items()}


def saveNumpy(inPath: Path, outDir: Path, types: frozenset):
  outDir.mkdir(parents=True, exist_ok=True)

  output = pico2Numpy(Picoscenes(str(inPath)).raw, types)
  for dataType in output.keys():
    filename = inPath.with_suffix(f".{dataType}.npy").name
    np.save(outDir / filename, output[dataType])


if __name__ == "__main__":
  scriptPath = Path(__file__).parent

  args = parseCli()
  inDir = Path(args.inDir)
  outDir = Path(args.outDir)

  for rawPath in tqdm(tuple(inDir.glob("*.csi")), desc="Files"):
    saveNumpy(rawPath, outDir, args.types)
